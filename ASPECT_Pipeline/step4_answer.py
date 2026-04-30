import json
import os
import logging
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import ollama

from config import (
    RETRIEVAL_FILE, TRIPLETS_FILE, ANSWERS_FILE, OUTPUTS_DIR,
    EMBED_MODEL, LLM_MODEL,
    MAX_TRIPLETS_CONTEXT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
log = logging.getLogger(__name__)

ALPHA       = 2.0
BETA        = 1.5
GAMMA       = 3.0
TOP_N_SEEDS = 10
MAX_HOPS    = 3


def build_full_graph(triplets_file: str):
    graph: dict         = defaultdict(list)
    entity_chunks: dict = defaultdict(set)

    log.info("Building full entity graph from all triplets ...")
    count = 0

    with open(triplets_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t        = json.loads(line)
            s        = t.get("subject",  "").lower().strip()
            r        = t.get("relation", "").lower().strip()
            o        = t.get("object",   "").lower().strip()
            text     = t.get("text",     f"{s} {r} {o}")
            chunk_id = t.get("chunk_id", "")

            if not s or not o:
                continue

            graph[s].append({"relation": r, "neighbor": o, "triplet_text": text})
            graph[o].append({"relation": r, "neighbor": s, "triplet_text": text})

            entity_chunks[s].add(chunk_id)
            entity_chunks[o].add(chunk_id)
            count += 1

    log.info(
        f"Full graph: {len(graph)} unique entities, "
        f"{count} triplets."
    )
    return graph, entity_chunks


def entities_from_query_triplets(query_triplets: list) -> set:
    entities = set()
    for qt in query_triplets:
        if "|" in qt:
            parts = [p.strip().lower() for p in qt.split("|") if p.strip()]
            if parts:
                entities.add(parts[0])
            if len(parts) >= 3:
                entities.add(parts[-1])
        else:
            for tok in qt.lower().split():
                if len(tok) > 3:
                    entities.add(tok)
    return entities


def frequency_in_top_chunks(top_chunk_ids: list,
                             entity_chunks: dict) -> dict:
    top_set = set(top_chunk_ids)
    freq: dict = {}
    for entity, chunk_set in entity_chunks.items():
        overlap = len(chunk_set & top_set)
        if overlap > 0:
            freq[entity] = overlap
    return freq


def select_seeds(
    query: str,
    query_triplets: list,
    top_chunk_ids: list,
    entity_chunks: dict,
    embedder: SentenceTransformer,
    top_n: int = TOP_N_SEEDS,
) -> list:
    query_entities = entities_from_query_triplets(query_triplets)
    freq_map       = frequency_in_top_chunks(top_chunk_ids, entity_chunks)

    candidates = set(freq_map.keys()) | query_entities

    if not candidates:
        log.warning("No candidate entities found. Returning empty seed list.")
        return []

    max_freq = max(freq_map.values()) if freq_map else 1

    q_emb     = embedder.encode(query, normalize_embeddings=True)
    cand_list = list(candidates)
    c_embs    = embedder.encode(
        cand_list, normalize_embeddings=True, show_progress_bar=False
    )

    scored = []
    for i, entity in enumerate(cand_list):
        sim    = float(np.dot(q_emb, c_embs[i]))
        norm_f = freq_map.get(entity, 0) / max_freq
        bonus  = 1.0 if entity in query_entities else 0.0

        hybrid = ALPHA * sim + BETA * norm_f + GAMMA * bonus
        scored.append((entity, hybrid))

    scored.sort(key=lambda x: x[1], reverse=True)

    top_seeds = [e for e, _ in scored[:top_n]]
    log.info(f"  Top seeds selected: {top_seeds}")
    return top_seeds


def khop_traverse(seeds: list, graph: dict,
                  max_hops: int = MAX_HOPS) -> list:
    visited: set     = set(s.lower() for s in seeds)
    current: set     = set(s.lower() for s in seeds)
    seen_texts: set  = set()
    collected: list  = []

    for hop in range(max_hops):
        next_hop: set = set()

        for entity in current:
            for edge in graph.get(entity, []):
                text = edge["triplet_text"]
                if text not in seen_texts:
                    collected.append(text)
                    seen_texts.add(text)

                nbr = edge["neighbor"]
                if nbr not in visited:
                    next_hop.add(nbr)

        visited.update(next_hop)
        current = next_hop

        log.info(
            f"    Hop {hop + 1}: "
            f"{len(current)} new entities frontier | "
            f"{len(collected)} total triplets collected"
        )

        if not current:
            break

    return collected


def rank_triplets(
    query: str,
    triplet_texts: list,
    embedder: SentenceTransformer,
    top_n: int = MAX_TRIPLETS_CONTEXT,
) -> list:
    if not triplet_texts:
        return []

    seen, unique = set(), []
    for t in triplet_texts:
        if t not in seen:
            unique.append(t)
            seen.add(t)
    triplet_texts = unique

    pre_n = min(top_n * 2, len(triplet_texts))

    try:
        vec    = TfidfVectorizer()
        tfidf  = vec.fit_transform(triplet_texts + [query])
        scores = (tfidf[:-1] @ tfidf[-1].T).toarray().flatten()
        order  = np.argsort(-scores)
        cands  = [triplet_texts[i] for i in order[:pre_n]]
    except Exception:
        cands  = triplet_texts[:pre_n]

    q_emb    = embedder.encode(query,  normalize_embeddings=True)
    c_embs   = embedder.encode(cands,  normalize_embeddings=True,
                               show_progress_bar=False)
    bi_scores = np.dot(c_embs, q_emb)
    bi_order  = np.argsort(-bi_scores)

    return [cands[i] for i in bi_order[:top_n]]


ANSWER_PROMPT = """You are a precise analyst. Answer the question using ONLY the knowledge graph facts below.

Each fact is formatted as:  Subject | Relation | Object

Rules:
- Base your answer strictly on the given facts.
- Be concise and specific. Quote exact values when present.
- If the facts do not contain enough information, say "Insufficient information."
- Do NOT invent or assume anything beyond the facts.

Knowledge Graph Facts:
{context}

Question: {question}

Answer:"""


def generate_answer(question: str, triplet_texts: list) -> str:
    context = "\n".join(f"  {t}" for t in triplet_texts)
    if not context.strip():
        return "Insufficient information."
    prompt = ANSWER_PROMPT.format(context=context, question=question)
    try:
        response = ollama.chat(
            model   = LLM_MODEL,
            messages= [{"role": "user", "content": prompt}],
            options = {"temperature": 0.0, "num_ctx": 8192}
        )
        return response["message"]["content"].strip()
    except Exception as exc:
        log.error(f"LLM call failed: {exc}")
        return "Error generating answer."


def run():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    with open(RETRIEVAL_FILE, encoding="utf-8") as f:
        retrieval_results = json.load(f)

    graph, entity_chunks = build_full_graph(TRIPLETS_FILE)

    log.info("Loading embedding model ...")
    embedder = SentenceTransformer(EMBED_MODEL)

    with open(ANSWERS_FILE, "w", encoding="utf-8") as out:
        for item in tqdm(retrieval_results, desc="Generating answers"):
            query          = item["question"]
            query_triplets = item.get("query_triplets", [])
            top_chunk_ids  = [c["chunk_id"] for c in item["top_chunks"]]

            log.info(f"\nQ{item['qid']}: {query[:80]} ...")

            seeds = select_seeds(
                query          = query,
                query_triplets = query_triplets,
                top_chunk_ids  = top_chunk_ids,
                entity_chunks  = entity_chunks,
                embedder       = embedder,
                top_n          = TOP_N_SEEDS,
            )

            if not seeds:
                log.warning(f"  No seeds for Q{item['qid']} — skipping traversal.")
                answer      = "Insufficient information."
                final_texts = []

            else:
                collected = khop_traverse(seeds, graph, max_hops=MAX_HOPS)
                log.info(
                    f"  {MAX_HOPS}-hop BFS collected "
                    f"{len(collected)} triplets from full graph."
                )

                final_texts = rank_triplets(query, collected, embedder)
                log.info(
                    f"  After ranking: "
                    f"{len(final_texts)} triplets used as context."
                )

                answer = generate_answer(query, final_texts)

            result = {
                "qid":                item["qid"],
                "question":           query,
                "gold_answer":        item["answer"],
                "predicted_answer":   answer,
                "type":               item.get("type"),
                "gold_chunk_id":      item.get("gold_chunk_id"),
                "seeds_used":         seeds,
                "triplets_collected": len(final_texts),
                "chunks_retrieved":   len(top_chunk_ids),
            }
            out.write(json.dumps(result, ensure_ascii=False) + "\n")

    log.info(f"Answers saved → {ANSWERS_FILE}")


if __name__ == "__main__":
    run()