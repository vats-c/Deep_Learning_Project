import json
import os
import logging
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import (
    ASPECTS_FILE, RETRIEVAL_FILE, OUTPUTS_DIR,
    EMBED_MODEL, LLM_MODEL,
    TOP_K_CHUNKS, MATCH_THRESHOLD, FACTUAL_TYPE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
log = logging.getLogger(__name__)


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


def _setup_chains(model_name: str):
    llm    = ChatOllama(model=model_name, temperature=0.0, num_predict=1024)
    parser = StrOutputParser()
    return (
        _coref_prompt | llm | parser,
        _pass1_prompt | llm | parser,
        _pass2_prompt | llm | parser,
    )


def extract_query_triplets(query: str, chains) -> list:
    coref_c, pass1_c, pass2_c = chains
    try:
        resolved = coref_c.invoke({"text": query})
        facts    = pass1_c.invoke({"chunk": resolved})
        raw      = pass2_c.invoke({"facts": facts})
    except Exception as exc:
        log.warning(f"Query decomposition failed: {exc}")
        return [query]

    texts = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 3 and parts[0]:
            texts.append(f"{parts[0]} {parts[1]} {parts[2]}")

    return texts if texts else [query]


def score_chunk(query_vecs: np.ndarray, aspects: list) -> float:
    if not aspects:
        return 0.0

    centroids  = np.array([a["centroid"] for a in aspects])
    sim_matrix = cosine_similarity(query_vecs, centroids)
    best_per_query = sim_matrix.max(axis=1)
    return float(best_per_query[best_per_query >= MATCH_THRESHOLD].sum())


def run(qa_file: str):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    with open(ASPECTS_FILE, encoding="utf-8") as f:
        aspects_store = json.load(f)

    with open(qa_file, encoding="utf-8") as f:
        qa_pairs = json.load(f)

    factual_pairs = [q for q in qa_pairs if q.get("type") == FACTUAL_TYPE]
    log.info(
        f"QA file: {len(qa_pairs)} total pairs | "
        f"{len(factual_pairs)} Factual pairs selected."
    )

    embedder   = SentenceTransformer(EMBED_MODEL)
    chains     = _setup_chains(LLM_MODEL)
    chunk_ids  = list(aspects_store.keys())

    all_results = []

    for item in tqdm(factual_pairs, desc="Scoring chunks"):
        query = item["question"]

        query_triplet_texts = extract_query_triplets(query, chains)
        log.info(f"  Q{item['qid']}: {len(query_triplet_texts)} query triplet(s)")

        query_vecs = embedder.encode(
            query_triplet_texts, normalize_embeddings=True
        )

        scores = {
            cid: score_chunk(query_vecs, aspects_store[cid].get("aspects", []))
            for cid in chunk_ids
        }

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_k  = ranked[:TOP_K_CHUNKS]

        all_results.append({
            "qid":              item["qid"],
            "question":         query,
            "answer":           item["answer"],
            "type":             item["type"],
            "doc_id":           item.get("doc_id"),
            "gold_chunk_id":    item.get("chunk_id"),
            "query_triplets":   query_triplet_texts,
            "top_chunks": [
                {"chunk_id": cid, "score": score}
                for cid, score in top_k
            ],
        })

    with open(RETRIEVAL_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    log.info(f"Retrieval results saved → {RETRIEVAL_FILE}")

    hit_ks = [1, 5, 10, 20, 50]
    print("\n" + "=" * 50)
    print("RETRIEVAL HIT@K REPORT (Factual questions)")
    print("=" * 50)
    n = len(all_results)
    for k in hit_ks:
        hits = sum(
            1 for r in all_results
            if r["gold_chunk_id"] in
               [c["chunk_id"] for c in r["top_chunks"][:k]]
        )
        print(f"  Hit@{k:3d}: {hits:4d} / {n}  ({hits / max(n,1):.1%})")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 3: Aspect-based chunk retrieval"
    )
    parser.add_argument(
        "--qa", required=True,
        help="Path to the QA JSON file (e.g. qa_pairs.json)"
    )
    args = parser.parse_args()
    run(args.qa)