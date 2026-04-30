import argparse
import re
import os
import json
from urllib import response
import numpy as np
import pickle
import joblib
import networkx as nx
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def read_sample_file(file_path):
    qa_pairs = []
   
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
   
    lines = content.split('\n')
    current_question = None
    current_answer = None
   
    for line in lines:
        line = line.strip()
       
        if line.startswith('**Question:**'):
            if current_question and current_answer:
                qa_pairs.append((current_question, current_answer))
            current_question = line.replace('**Question:**', '').strip()
            current_answer = None
           
        elif line.startswith('**Answer:**'):
            current_answer = line.replace('**Answer:**', '').strip()
   
    if current_question and current_answer:
        qa_pairs.append((current_question, current_answer))
   
    return qa_pairs

def save_to_markdown(queries, ground_truths, results, output_file):
    output_content = "# Retrieval Results\n\n"
    output_content += "---\n\n"
   
    for i, (query, truth) in enumerate(zip(queries, ground_truths), 1):
        output_content += f"\n### Pair {i}\n"
        output_content += f"**Question:** {query}\n"
        output_content += f"**Ground Truth:** {truth}\n"
        output_content += f"**Retrieved Answer:** {results.get(query, 'No answer available')}\n\n"
        output_content += "---\n"
   
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output_content)

class NetworkXRetriever:
    def __init__(self, max_hop_depth=3):
        self.max_hop_depth = max_hop_depth
        self.chunk_embeddings = {}
        self.chunk_ids = []
        self.embedding_matrix = None
        self.chunk_triplet_mapping = {}
       
        self.G = nx.MultiDiGraph()
        self.vectorizer = None
        self.tfidf_matrix = None
        self.tfidf_entity_list = []

    def load_from_json(self, json_file, embedding_model):
        print("[1/4] Loading JSON data...")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        chunk_texts = {}
        entity_chunk_map = defaultdict(set)

        for item in data:
            cid = item["chunk_id"]
            chunk_texts[cid] = item["chunk_text"]
            triplet_strings = []
            for t in item["triplets"]:
                subj = t["subject"]
                pred = t["predicate"]
                obj  = t["object"]
                triplet_strings.append(f"({subj}, {pred}, {obj})")
                entity_chunk_map[subj].add(cid)
                entity_chunk_map[obj].add(cid)
            self.chunk_triplet_mapping[cid] = triplet_strings

        print(f"[2/4] Encoding {len(chunk_texts)} chunk embeddings...")
        self.chunk_ids = list(chunk_texts.keys())
        texts_ordered = [chunk_texts[cid] for cid in self.chunk_ids]
        embeddings_array = embedding_model.encode(texts_ordered, show_progress_bar=True,
                                                   batch_size=128)
        for idx, cid in enumerate(self.chunk_ids):
            self.chunk_embeddings[cid] = embeddings_array[idx]
        self.embedding_matrix = np.vstack([self.chunk_embeddings[cid]
                                           for cid in self.chunk_ids])

        print("[3/4] Building knowledge graph...")
        cid_to_idx = {cid: i for i, cid in enumerate(self.chunk_ids)}

        for entity, cids in entity_chunk_map.items():
            emb = np.mean([embeddings_array[cid_to_idx[c]] for c in cids], axis=0)
            self.G.add_node(entity,
                            chunk_ids=list(cids),
                            embedding=emb)

        for item in data:
            for t in item["triplets"]:
                subj = t["subject"]
                pred = t["predicate"]
                obj  = t["object"]
                self.G.add_edge(subj, obj, original_predicate=pred)

        print("[4/4] Building TF-IDF index...")
        self.tfidf_entity_list = list(entity_chunk_map.keys())

        entity_documents = []
        for entity in self.tfidf_entity_list:
            cids = entity_chunk_map[entity]
            combined_text = " ".join(chunk_texts[c] for c in cids)
            entity_documents.append(combined_text)

        self.vectorizer = TfidfVectorizer(max_features=50000, stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(entity_documents)

        print(f"    Graph: {self.G.number_of_nodes()} nodes, "
              f"{self.G.number_of_edges()} edges")
        print(f"    Chunks: {len(self.chunk_ids)}")
        print(f"    TF-IDF entities: {len(self.tfidf_entity_list)}")
        print("    Done.\n")

    def load_graph_data(self, graph_file):
        with open(graph_file, "rb") as f:
            self.G = pickle.load(f)

    def load_chunk_data(self, chunk_file):
        with open(chunk_file, "rb") as f:
            data = pickle.load(f)
            self.chunk_triplet_mapping = data["chunk_triplet_mapping"]
            self.chunk_embeddings = data["chunk_embeddings"]
            self.chunk_ids = list(self.chunk_embeddings.keys())
            embeddings_list = [self.chunk_embeddings[cid] for cid in self.chunk_ids]
            self.embedding_matrix = np.vstack(embeddings_list)

    def load_tfidf_data(self, tfidf_file):
        tfidf_data = joblib.load(tfidf_file)
        self.vectorizer = tfidf_data["vectorizer"]
        self.tfidf_matrix = tfidf_data["tfidf_matrix"]
        self.tfidf_entity_list = tfidf_data["entity_list"]

    def _select_chunks(self, query_text, embedding_model, top_k=5):
        query_emb = embedding_model.encode([query_text])[0].reshape(1, -1)
        similarities = cosine_similarity(query_emb, self.embedding_matrix)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [self.chunk_ids[idx] for idx in top_indices]

    def _vector_search_entities(self, query_text, selected_chunks, embedding_model, top_k=5):
        query_emb = embedding_model.encode([query_text])[0].reshape(1, -1)
        scores = []
        for node in self.G.nodes:
            data = self.G.nodes[node]
            if set(data.get('chunk_ids', [])) & set(selected_chunks):
                emb = data.get('embedding')
                if emb is not None:
                    sim = cosine_similarity(query_emb, emb.reshape(1, -1))[0][0]
                    scores.append((node, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [n for n, s in scores[:top_k]]

    def _tfidf_search_entities(self, query_text, selected_chunks, top_k=5):
        query_vec = self.vectorizer.transform([query_text])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        scores = []
        for i, entity in enumerate(self.tfidf_entity_list):
            if entity in self.G.nodes:
                data = self.G.nodes[entity]
                if set(data.get('chunk_ids', [])) & set(selected_chunks):
                    scores.append((entity, similarities[i]))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [n for n, s in scores[:top_k]]

    def _get_hybrid_seeds(self, query_text, embedding_model):
        chunks = self._select_chunks(query_text, embedding_model)
        vec_entities = self._vector_search_entities(query_text, chunks, embedding_model, 5)
        tfidf_entities = self._tfidf_search_entities(query_text, chunks, 5)
        seeds = list(set(vec_entities + tfidf_entities))
        return seeds

    def _collect_multihop_triplets(self, seeds):
        triplets = set()
        for seed in seeds:
            if seed not in self.G.nodes:
                continue
            current = {seed}
            for depth in range(1, self.max_hop_depth + 1):
                next_level = set()
                for node in current:
                    for neighbor in self.G.neighbors(node):
                        for key in self.G[node][neighbor]:
                            data = self.G[node][neighbor][key]
                            p = data['original_predicate']
                            triplet = f"({node}, {p}, {neighbor})"
                            triplets.add(triplet)
                            next_level.add(neighbor)
                current = next_level
        return list(triplets)

    def retrieve_triplets(self, query_text, embedding_model):
        seeds = self._get_hybrid_seeds(query_text, embedding_model)
        triplets = self._collect_multihop_triplets(seeds)
        return triplets

def answer_question(query, triplets, llm):
    context = "\n".join(triplets)
    prompt = f"""
You are an expert analyst given a set of factual triplets extracted from reliable sources.
Your task is to carefully analyze these facts and provide a clear, concise, short to the point answer to the question.
Answer the question as factual type, just the fact, with no description.

Factual context:
{context}

Question: {query}
Answer: """
    response = llm.invoke(prompt)
    return response.strip()

def main():
    parser = argparse.ArgumentParser(description="Knowledge Graph Querying")
    parser.add_argument("--qa-file", default="questions.md", help="Input QA Markdown file")
    parser.add_argument("--output-file", default="output.md", help="Output results Markdown file")
    parser.add_argument("--embedding-model", default="all-mpnet-base-v2", help="Embedding model name")
    parser.add_argument("--llm-model", default="llama3.1:70b", help="Ollama LLM model")

    parser.add_argument("--triplets-json", default=None,
                        help="Input JSON file with chunk-to-triplet mappings. "
                             "When provided, builds graph/embeddings/tfidf from "
                             "this single file (no pickle/joblib needed).")

    parser.add_argument("--graph-file", default="knowledge_graph.pickle",
                        help="Input graph pickle file (ignored when --triplets-json is set)")
    parser.add_argument("--chunk-file", default="chunk_data.pickle",
                        help="Input chunk data pickle file (ignored when --triplets-json is set)")
    parser.add_argument("--tfidf-file", default="tfidf_data.joblib",
                        help="Input TF-IDF joblib file (ignored when --triplets-json is set)")

    args = parser.parse_args()
   
    embedding_model = SentenceTransformer(args.embedding_model)
    llm = Ollama(model=args.llm_model, temperature=0)
   
    qa_pairs = read_sample_file(args.qa_file)
   
    if not qa_pairs:
        print("No QA pairs found. Exiting.")
        return
       
    queries = [q for q, _ in qa_pairs]
    ground_truths = [a for _, a in qa_pairs]
   
    retriever = NetworkXRetriever()

    if args.triplets_json:
        retriever.load_from_json(args.triplets_json, embedding_model)
    else:
        retriever.load_graph_data(args.graph_file)
        retriever.load_chunk_data(args.chunk_file)
        retriever.load_tfidf_data(args.tfidf_file)
   
    results = {}
   
    for i, query in enumerate(queries, 1):
        print(f"[Query {i}/{len(queries)}] {query}")
        triplets = retriever.retrieve_triplets(query, embedding_model)
        answer = "No answer available"
        if triplets:
            answer = answer_question(query, triplets, llm)
        results[query] = answer
        print(f"  -> {answer}\n")
       
    save_to_markdown(queries, ground_truths, results, args.output_file)
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()