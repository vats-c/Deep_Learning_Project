import os

MD_ROOT      = "."
OUTPUTS_DIR  = "outputs"

CHUNKS_FILE      = os.path.join(OUTPUTS_DIR, "chunks.json")
TRIPLETS_FILE    = os.path.join(OUTPUTS_DIR, "triplets.jsonl")
ASPECTS_FILE     = os.path.join(OUTPUTS_DIR, "aspects.json")
RETRIEVAL_FILE   = os.path.join(OUTPUTS_DIR, "retrieval_results.json")
ANSWERS_FILE     = os.path.join(OUTPUTS_DIR, "answers.jsonl")
EVAL_FILE        = os.path.join(OUTPUTS_DIR, "evaluation.jsonl")
EVAL_REPORT      = os.path.join(OUTPUTS_DIR, "eval_report.txt")

EMBED_MODEL  = "all-MiniLM-L6-v2"
LLM_MODEL    = "qwen2.5:14b"
JUDGE_MODEL  = "qwen2.5:14b"

MAX_CHUNK_SIZE = 4096
CHUNK_OVERLAP  = 512

EPS_START   = 0.10
EPS_END     = 0.90
EPS_STEP    = 0.05
BURST_DROP  = 0.40
MIN_TRIPLETS_FOR_CLUSTER = 3

TOP_K_CHUNKS     = 50
MATCH_THRESHOLD  = 0.40

MAX_HOPS              = 3
MAX_TRIPLETS_CONTEXT  = 120

FACTUAL_TYPE = "Factual"