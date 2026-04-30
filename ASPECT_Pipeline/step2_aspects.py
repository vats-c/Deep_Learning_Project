import json
import os
import logging
import numpy as np
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import (
    CHUNKS_FILE, TRIPLETS_FILE, ASPECTS_FILE, OUTPUTS_DIR,
    EMBED_MODEL, LLM_MODEL,
    EPS_START, EPS_END, EPS_STEP, BURST_DROP,
    MIN_TRIPLETS_FOR_CLUSTER,
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
     "Do NOT rewrite, summarise, or add anything."),
    ("human", "Original text:\n{text}\n\nOutput ONLY the resolved text."),
])

_pass1_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a factual information extractor. "
     "Extract every atomic fact from the text without loss. "
     "Process sentence by sentence."),
    ("human", "Text:\n{chunk}\n\nFacts (one per line):"),
])

_pass2_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Convert each fact into a KG triplet using the format: "
     "Subject | Relation | Object\n"
     "One triplet per line. No explanation."),
    ("human", "Facts:\n{facts}\n\nTriplets:"),
])


def _setup_chains(model_name: str):
    llm    = ChatOllama(model=model_name, temperature=0.0, num_predict=4096)
    parser = StrOutputParser()
    return (
        _coref_prompt | llm | parser,
        _pass1_prompt | llm | parser,
        _pass2_prompt | llm | parser,
    )


def extract_triplets(text: str, chains, chunk_id: str) -> list:
    coref_c, pass1_c, pass2_c = chains
    try:
        resolved    = coref_c.invoke({"text": text})
        facts_raw   = pass1_c.invoke({"chunk": resolved})
        triplets_raw = pass2_c.invoke({"facts": facts_raw})
    except Exception as exc:
        log.warning(f"Extraction failed for {chunk_id}: {exc}")
        return []

    triplets = []
    for line in triplets_raw.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3 or not parts[0]:
            continue
        triplets.append({
            "chunk_id": chunk_id,
            "subject":  parts[0],
            "relation": parts[1],
            "object":   parts[2],
            "text":     f"{parts[0]} {parts[1]} {parts[2]}",
        })
    return triplets


def adaptive_cluster(embeddings: np.ndarray) -> np.ndarray:
    n = len(embeddings)
    if n < MIN_TRIPLETS_FOR_CLUSTER:
        return np.arange(n)

    prev_count  = n
    best_labels = np.arange(n)

    for raw_eps in np.arange(EPS_START, EPS_END + EPS_STEP, EPS_STEP):
        eps    = round(float(raw_eps), 4)
        labels = DBSCAN(eps=eps, min_samples=1,
                        metric="cosine").fit_predict(embeddings)

        unique     = set(labels)
        n_clusters = max(len(unique) - (1 if -1 in unique else 0), 1)
        drop       = (prev_count - n_clusters) / prev_count if prev_count > 0 else 0

        if drop > BURST_DROP:
            break

        best_labels = labels
        prev_count  = n_clusters

    return best_labels


def build_aspects(triplet_texts: list, embedder: SentenceTransformer) -> list:
    if not triplet_texts:
        return []

    embeddings = embedder.encode(triplet_texts, normalize_embeddings=True)
    labels     = adaptive_cluster(np.array(embeddings))

    aspects = []
    for label in sorted(set(labels)):
        idxs     = [i for i, l in enumerate(labels) if l == label]
        vecs     = np.array([embeddings[i] for i in idxs])
        centroid = vecs.mean(axis=0)
        norm     = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        aspects.append({
            "centroid": centroid.tolist(),
            "size":     len(idxs),
            "triplets": [triplet_texts[i] for i in idxs],
        })
    return aspects


def run():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)

    log.info(f"Loaded {len(chunks)} chunks. Initialising models ...")
    embedder = SentenceTransformer(EMBED_MODEL)
    chains   = _setup_chains(LLM_MODEL)

    aspects_store: dict = {}

    triplet_out = open(TRIPLETS_FILE, "w", encoding="utf-8")

    for chunk in tqdm(chunks, desc="Extracting triplets & building aspects"):
        chunk_id = chunk["chunk_id"]
        text     = chunk.get("text", "").strip()

        if len(text.split()) < 10:
            continue

        triplets = extract_triplets(text, chains, chunk_id)
        for t in triplets:
            triplet_out.write(json.dumps(t) + "\n")

        triplet_texts = [t["text"] for t in triplets]
        aspects       = build_aspects(triplet_texts, embedder)

        aspects_store[chunk_id] = {"aspects": aspects}
        log.info(f"  {chunk_id}: {len(triplets)} triplets → {len(aspects)} aspects")

    triplet_out.close()

    with open(ASPECTS_FILE, "w", encoding="utf-8") as f:
        json.dump(aspects_store, f, indent=2, ensure_ascii=False)

    log.info(f"Aspects saved   → {ASPECTS_FILE}")
    log.info(f"Triplets saved  → {TRIPLETS_FILE}")


if __name__ == "__main__":
    run()