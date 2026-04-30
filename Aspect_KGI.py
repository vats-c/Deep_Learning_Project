import argparse
import json
import pickle
from collections import defaultdict

import networkx as nx
import numpy as np
import joblib

from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


_EPS_START = 0.10
_EPS_END = 0.90
_EPS_STEP = 0.05
_BURST_DROP = 0.40
_MIN_TRIPLETS_FOR_CLUSTER = 3


def _adaptive_cluster(embeddings: np.ndarray) -> np.ndarray:
    n = len(embeddings)
    if n == 0:
        return np.array([], dtype=int)
    if n < _MIN_TRIPLETS_FOR_CLUSTER:
        return np.arange(n)

    prev_count = n
    best_labels = np.arange(n)

    for raw_eps in np.arange(_EPS_START, _EPS_END + _EPS_STEP, _EPS_STEP):
        eps = round(float(raw_eps), 4)
        labels = DBSCAN(eps=eps, min_samples=1, metric="cosine").fit_predict(embeddings)

        unique = set(labels)
        n_clusters = max(len(unique) - (1 if -1 in unique else 0), 1)
        drop = (prev_count - n_clusters) / prev_count if prev_count > 0 else 0

        if drop > _BURST_DROP:
            break

        best_labels = labels
        prev_count = n_clusters

    return best_labels


def _build_aspects(triplet_texts: list, embedder: SentenceTransformer) -> list:
    if not triplet_texts:
        return []

    embeddings = embedder.encode(
        triplet_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    embeddings = np.array(embeddings)
    labels = _adaptive_cluster(embeddings)

    centroids = []
    for label in sorted(set(labels)):
        idxs = [i for i, l in enumerate(labels) if l == label]
        vecs = embeddings[idxs]
        centroid = vecs.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroids.append(centroid)

    return centroids


_coref_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a STRICT coreference resolution engine. "
     "Replace every pronoun / reference with the exact entity name. "
     "Do NOT rewrite or summarise."),
    ("human", "Original text:\n{text}\n\nOutput ONLY the resolved text."),
])

_pass1_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a factual information extractor. "
     "Extract every atomic information need from the question."),
    ("human", "Question:\n{chunk}\n\nInformation needs (one per line):"),
])

_pass2_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Convert each information need into a KG triplet: "
     "Subject | Relation | Object\nOne per line. No explanation."),
    ("human", "Information needs:\n{facts}\n\nTriplets:"),
])


def _build_decomposition_chains(model_name: str):
    llm = ChatOllama(model=model_name, temperature=0.0, num_predict=1024)
    parser = StrOutputParser()
    return (
        _coref_prompt | llm | parser,
        _pass1_prompt | llm | parser,
        _pass2_prompt | llm | parser,
    )


def _extract_query_triplets(query: str, chains) -> list:
    coref_c, pass1_c, pass2_c = chains
    try:
        resolved = coref_c.invoke({"text": query})
        facts = pass1_c.invoke({"chunk": resolved})
        raw = pass2_c.invoke({"facts": facts})
    except Exception as exc:
        print(f"  [decompose] LLM call failed ({exc}), falling back to raw query.")
        return [query]

    texts = []
    for line in str(raw).splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 3 and parts[0] and parts[1] and parts[2]:
            texts.append(f"{parts[0]} {parts[1]} {parts[2]}")

    if not texts:
        print("  [decompose] No triplets parsed, falling back to raw query.")
        return [query]

    print(f"  [decompose] {len(texts)} query triplet(s): {texts}")
    return texts


def read_sample_file(file_path):
    qa_pairs = []

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")
    current_question = None
    current_answer = None

    for line in lines:
        line = line.strip()

        if line.startswith("**Question:**"):
            if current_question is not None and current_answer is not None:
                qa_pairs.append((current_question, current_answer))
            current_question = line.replace("**Question:**", "").strip()
            current_answer = None

        elif line.startswith("**Answer:**"):
            current_answer = line.replace("**Answer:**", "").strip()

    if current_question is not None and current_answer is not None:
        qa_pairs.append((current_question, current_answer))

    return qa_pairs


def save_to_markdown(queries, ground_truths, results, output_file):
    output_content = "# Retrieval Results\n\n---\n"

    for i, (query, truth) in enumerate(zip(queries, ground_truths), 1):
        output_content += f"\n### Pair {i}\n"
        output_content += f"**Question:** {query}\n"
        output_content += f"**Ground Truth:** {truth}\n"
        output_content += f"**Retrieved Answer:** {results[i - 1] if i - 1 < len(results) else 'No answer available'}\n\n"
        output_content += "---\n"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output_content)


class NetworkXRetriever:
    def __init__(self, max_hop_depth=3, llm_model="llama3.1:70b"):
        self.max_hop_depth = max_hop_depth
        self.chunk_embeddings = {}
        self.chunk_ids = []
        self.embedding_matrix = None
        self.chunk_triplet_mapping = {}

        self.chunk_aspects = {}

        print(f"[init] Setting up query-decomposition chains (model: {llm_model})...")
        self._decomp_chains = _build_decomposition_chains(llm_model)

        self.G = nx.MultiDiGraph()
        self.vectorizer = None
        self.tfidf_matrix = None
        self.tfidf_entity_list = []

    def load_from_json(self, json_file, embedding_model):
        print("[1/5] Loading JSON data...")
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunk_texts = {}
        chunk_triplet_strings = {}
        entity_chunk_map = defaultdict(set)

        for item in data:
            cid = item["chunk_id"]
            chunk_texts[cid] = item.get("chunk_text", "")

            triplet_strings = []
            chunk_triplet_strings[cid] = []

            for t in item.get("triplets", []):
                subj = t.get("subject", "").strip()
                pred = t.get("predicate", "").strip()
                obj = t.get("object", "").strip()

                if not subj or not pred or not obj:
                    continue

                triplet_strings.append(f"({subj}, {pred}, {obj})")
                chunk_triplet_strings[cid].append(f"{subj} {pred} {obj}")
                entity_chunk_map[subj].add(cid)
                entity_chunk_map[obj].add(cid)

            self.chunk_triplet_mapping[cid] = triplet_strings

        print(f"[2/5] Encoding {len(chunk_texts)} chunk embeddings...")
        self.chunk_ids = list(chunk_texts.keys())
        texts_ordered = [chunk_texts[cid] for cid in self.chunk_ids]
        embeddings_array = embedding_model.encode(
            texts_ordered,
            show_progress_bar=True,
            batch_size=128,
            normalize_embeddings=True,
        )
        embeddings_array = np.array(embeddings_array)

        for idx, cid in enumerate(self.chunk_ids):
            self.chunk_embeddings[cid] = embeddings_array[idx]
        self.embedding_matrix = np.vstack([self.chunk_embeddings[cid] for cid in self.chunk_ids])

        print("[3/5] Building knowledge graph...")
        for entity, cids in entity_chunk_map.items():
            emb = embedding_model.encode([entity], normalize_embeddings=True)[0]
            self.G.add_node(entity, chunk_ids=list(cids), embedding=emb)

        for item in data:
            for t in item.get("triplets", []):
                subj = t.get("subject", "").strip()
                pred = t.get("predicate", "").strip()
                obj = t.get("object", "").strip()
                if subj and pred and obj:
                    self.G.add_edge(subj, obj, original_predicate=pred)

        print(f"[4/5] Building aspect centroids for {len(self.chunk_ids)} chunks...")
        aspect_counts = []
        for cid in self.chunk_ids:
            triplet_texts = chunk_triplet_strings.get(cid, [])
            self.chunk_aspects[cid] = _build_aspects(triplet_texts, embedding_model)
            aspect_counts.append(len(self.chunk_aspects[cid]))
        avg_aspects = float(np.mean(aspect_counts)) if aspect_counts else 0.0
        print(f"      Done. (avg {avg_aspects:.1f} aspects/chunk)")

        print("[5/5] Building TF-IDF index...")
        self.tfidf_entity_list = list(entity_chunk_map.keys())

        entity_documents = []
        for entity in self.tfidf_entity_list:
            cids = entity_chunk_map[entity]
            combined_text = " ".join(chunk_texts[c] for c in cids)
            entity_documents.append(combined_text)

        self.vectorizer = TfidfVectorizer(max_features=50000, stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(entity_documents)

        print(
            f"    Graph: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges"
        )
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
        query_triplet_texts = _extract_query_triplets(query_text, self._decomp_chains)

        Q = embedding_model.encode(query_triplet_texts, normalize_embeddings=True)
        Q = np.array(Q)
        if Q.ndim == 1:
            Q = Q.reshape(1, -1)

        scores = {}
        for cid in self.chunk_ids:
            aspects = self.chunk_aspects.get(cid, [])
            if not aspects:
                scores[cid] = 0.0
                continue

            A = np.array(aspects)
            sim_matrix = cosine_similarity(Q, A)
            best_per_query = sim_matrix.max(axis=1)
            scores[cid] = float(best_per_query.sum())

        return sorted(scores, key=scores.get, reverse=True)[:top_k]

    def _vector_search_entities(self, query_text, selected_chunks, embedding_model, top_k=5):
        query_emb = embedding_model.encode([query_text], normalize_embeddings=True)[0].reshape(1, -1)
        scores = []

        selected_chunks = set(selected_chunks)

        for node in self.G.nodes:
            data = self.G.nodes[node]
            if set(data.get("chunk_ids", [])) & selected_chunks:
                emb = data.get("embedding")
                if emb is not None:
                    sim = cosine_similarity(query_emb, emb.reshape(1, -1))[0][0]
                    scores.append((node, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [n for n, _ in scores[:top_k]]

    def _tfidf_search_entities(self, query_text, selected_chunks, top_k=5):
        if self.vectorizer is None or self.tfidf_matrix is None:
            return []

        query_vec = self.vectorizer.transform([query_text])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        scores = []
        selected_chunks = set(selected_chunks)

        for i, entity in enumerate(self.tfidf_entity_list):
            if entity in self.G.nodes:
                data = self.G.nodes[entity]
                if set(data.get("chunk_ids", [])) & selected_chunks:
                    scores.append((entity, similarities[i]))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [n for n, _ in scores[:top_k]]

    def _get_hybrid_seeds(self, query_text, embedding_model):
        chunks = self._select_chunks(query_text, embedding_model)
        vec_entities = self._vector_search_entities(query_text, chunks, embedding_model, 5)
        tfidf_entities = self._tfidf_search_entities(query_text, chunks, 5)
        return list(set(vec_entities + tfidf_entities))

    def _collect_multihop_triplets(self, seeds):
        triplets = set()

        for seed in seeds:
            if seed not in self.G.nodes:
                continue

            current = {seed}
            for _depth in range(1, self.max_hop_depth + 1):
                next_level = set()
                for node in current:
                    for neighbor in self.G.neighbors(node):
                        for key in self.G[node][neighbor]:
                            data = self.G[node][neighbor][key]
                            p = data["original_predicate"]
                            triplets.add(f"({node}, {p}, {neighbor})")
                            next_level.add(neighbor)
                current = next_level

        return list(triplets)

    def retrieve_triplets(self, query_text, embedding_model):
        seeds = self._get_hybrid_seeds(query_text, embedding_model)
        return self._collect_multihop_triplets(seeds)


def answer_question(query, triplets, llm):
    context = "\n".join(triplets)
    prompt = f"""
You are an expert analyst given a set of factual triplets extracted from reliable sources.
Your task is to carefully analyze these facts and provide a clear, concise, short to the point answer to the question.
Answer the question as factual type, just the fact, with no description.

Factual context:
{context}

Question: {query}
Answer:
""".strip()

    response = llm.invoke(prompt)

    if response is None:
        return "No answer available"

    if hasattr(response, "content"):
        text = response.content
    else:
        text = str(response)

    text = text.strip()
    return text if text else "No answer available"


def main():
    parser = argparse.ArgumentParser(description="Aspect Knowledge Graph Querying")
    parser.add_argument("--qa-file", default="questions.md", help="Input QA Markdown file")
    parser.add_argument("--output-file", default="output.md", help="Output results Markdown file")
    parser.add_argument("--embedding-model", default="all-mpnet-base-v2", help="Embedding model name")
    parser.add_argument("--llm-model", default="llama3.1:70b", help="Ollama LLM model")
    parser.add_argument(
        "--triplets-json",
        default=None,
        help="Input JSON file with chunk-to-triplet mappings. When provided, builds graph/embeddings/tfidf from this single file.",
    )
    parser.add_argument(
        "--graph-file",
        default="knowledge_graph.pickle",
        help="Input graph pickle file (ignored when --triplets-json is set)",
    )
    parser.add_argument(
        "--chunk-file",
        default="chunk_data.pickle",
        help="Input chunk data pickle file (ignored when --triplets-json is set)",
    )
    parser.add_argument(
        "--tfidf-file",
        default="tfidf_data.joblib",
        help="Input TF-IDF joblib file (ignored when --triplets-json is set)",
    )

    args = parser.parse_args()

    embedding_model = SentenceTransformer(args.embedding_model)
    llm = ChatOllama(model=args.llm_model, temperature=0.0, num_predict=1024)

    retriever = NetworkXRetriever(llm_model=args.llm_model)

    if args.triplets_json:
        retriever.load_from_json(args.triplets_json, embedding_model)
    else:
        retriever.load_graph_data(args.graph_file)
        retriever.load_chunk_data(args.chunk_file)
        retriever.load_tfidf_data(args.tfidf_file)

    qa_pairs = read_sample_file(args.qa_file)
    if not qa_pairs:
        print("No QA pairs found. Exiting.")
        return

    queries = [q for q, _ in qa_pairs]
    ground_truths = [a for _, a in qa_pairs]

    results = []

    for i, query in enumerate(queries, 1):
        print(f"[Query {i}/{len(queries)}] {query}")
        triplets = retriever.retrieve_triplets(query, embedding_model)

        answer = "No answer available"
        if triplets:
            answer = answer_question(query, triplets, llm)

        results.append(answer)
        print(f"  -> {answer}\n")

    save_to_markdown(queries, ground_truths, results, args.output_file)
    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()