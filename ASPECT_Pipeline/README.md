## What This Pipeline Does

1. Reads your `.md` documents and splits them into chunks
2. Extracts knowledge triplets from each chunk using an LLM (Subject → Relation → Object)
3. Clusters those triplets into semantic groups called **aspects** and stores their centroids
4. When a question arrives, it decomposes the question into triplets and matches them against stored aspects to find the most relevant chunks
5. Traverses a knowledge graph built from the retrieved chunks to collect supporting facts
6. Feeds those facts to the LLM to generate a final answer
7. Evaluates the answers on factual questions using an LLM-as-Judge

---

## Requirements

### 1. Python

You need Python 3.9 or newer. Check your version:

```bash
python --version
```

If you do not have Python, download it from https://www.python.org/downloads/

---

### 2. Ollama (runs the language model locally)

Ollama lets you run large language models on your own machine.

**Install Ollama:**

- On Linux / Mac, open a terminal and run:
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```
- On Windows, download the installer from https://ollama.com/download

**Pull the model the pipeline uses:**

```bash
ollama pull qwen2.5:14b
```

This downloads the model (around 9 GB). You only need to do this once.

**Make sure Ollama is running** before you run the pipeline:

```bash
ollama serve
```

Leave that terminal open and open a new one for the pipeline commands.

---

### 3. Python Libraries

Install everything the pipeline needs in one command:

```bash
pip install sentence-transformers scikit-learn langchain-ollama tqdm numpy
```

---

## Folder Setup

Your project folder should look like this before you start:

```
your_project_folder/
│
├── config.py              ← pipeline settings
├── run.py                 ← master runner (this is what you execute)
├── step1_ingest.py
├── step2_aspects.py
├── step3_retrieve.py
├── step4_answer.py
├── step5_evaluate.py
│
├── qa_pairs.json          ← your QA test file
│
├── document_one.md        ← your markdown documents
├── document_two.md
└── ...
```

All your `.md` files and the `qa_pairs.json` file should be in the same folder as the Python scripts.

---

## Running the Pipeline

### Full Run (first time)

Open a terminal, navigate to your project folder, and run:

```bash
python run.py --qa qa_pairs.json
```

Replace `qa_pairs.json` with the actual name of your QA file.

This will run all five steps one after another and save all outputs to a folder called `outputs/`.

---

### What Happens During Each Step

| Step | Script | What it does | Output file |
|------|--------|--------------|-------------|
| 1 | step1_ingest.py | Reads `.md` files, creates chunks | `outputs/chunks.json` |
| 2 | step2_aspects.py | Extracts triplets, builds aspects | `outputs/triplets.jsonl`, `outputs/aspects.json` |
| 3 | step3_retrieve.py | Matches question to relevant chunks | `outputs/retrieval_results.json` |
| 4 | step4_answer.py | Generates answers from the KG | `outputs/answers.jsonl` |
| 5 | step5_evaluate.py | Scores answers (factual questions only) | `outputs/evaluation.jsonl`, `outputs/eval_report.txt` |

---

### Skipping Steps You Already Did

Steps 1 and 2 are slow because they process every document. Once you have done them once, you can skip them on future runs:

```bash
# Chunks already exist, skip step 1
python run.py --qa qa_pairs.json --skip-ingest

# Chunks AND aspects already exist, skip steps 1 and 2
python run.py --qa qa_pairs.json --skip-ingest --skip-aspects
```

---

### Running a Single Step

If you want to run just one step on its own:

```bash
python step1_ingest.py
python step2_aspects.py
python step3_retrieve.py --qa qa_pairs.json
python step4_answer.py
python step5_evaluate.py
```

Note that each step depends on the output of the previous one, so they must be run in order at least once.

---

## Reading the Results

### Evaluation Report

After the pipeline finishes, open `outputs/eval_report.txt`. It looks like this:

```
====================================================
EVALUATION REPORT — FACTUAL QUESTIONS ONLY
====================================================
Total Factual Questions   : 30
Average Judge Score       : 7.43 / 10
----------------------------------------------------
Correct   (score >= 8)    : 19  (63.3%)
Partial   (5 <= score < 8):  8  (26.7%)
Failures  (score <  5)    :  3  (10.0%)
====================================================
```

Only **Factual** type questions from your QA file are evaluated. Other question types (procedural, legal, etc.) are ignored.

### Detailed Scores

Open `outputs/evaluation.jsonl` to see the score for each individual question. Each line is one question with its predicted answer, gold answer, and judge score.

### Generated Answers

Open `outputs/answers.jsonl` to see what the pipeline actually answered for each question.

---

## Configuring the Pipeline

All settings are in `config.py`. You do not need to touch any other file to change behaviour.

### The most useful settings to change:

```python
# Which embedding model to use (must be a sentence-transformers model)
EMBED_MODEL = "all-MiniLM-L6-v2"

# Which Ollama model to use for triplet extraction and answering
LLM_MODEL = "qwen2.5:14b"

# How many candidate chunks to retrieve per question
TOP_K_CHUNKS = 50

# How deep the knowledge graph traversal goes
MAX_HOPS = 3

# Maximum triplets fed to the LLM for answer generation
MAX_TRIPLETS_CONTEXT = 120
```

### Clustering settings (controls aspect quality):

```python
# These control the adaptive DBSCAN clustering
EPS_START  = 0.10   # start with tight communities
EPS_END    = 0.90   # maximum looseness to try
EPS_STEP   = 0.05   # step size per sweep
BURST_DROP = 0.40   # stop when communities merge too aggressively
```

If you feel aspects are too fine-grained (too many small clusters), increase `EPS_START`. If they are too coarse (one big cluster), decrease `BURST_DROP`.

---

## QA File Format

Your QA file must be a JSON array. Each object must have at least these fields:

```json
[
  {
    "qid": 0,
    "type": "Factual",
    "question": "What is the blade length of a 6 MW offshore wind turbine?",
    "answer": "73 m",
    "chunk_id": "Report_chunk_42",
    "doc_id": "Report_2016"
  }
]
```

- `type` must be exactly `"Factual"` (capital F) for questions you want evaluated
- `chunk_id` is used to compute Hit@K (how often the right chunk was retrieved)
- Other fields (`doc_id`, `context`, `section_header`, etc.) are optional

---

## Common Problems

**"No .md files found"**
Make sure your markdown files are in the same folder as the Python scripts, not inside a subfolder.

**"Connection refused" or Ollama errors**
Ollama is not running. Open a separate terminal and run `ollama serve`, then try again.

**"Model not found"**
You have not downloaded the model yet. Run `ollama pull qwen2.5:14b`.

**Step 2 is very slow**
This is expected. Qwen processes every chunk to extract triplets. For a large document set this can take hours. Run it once and then use `--skip-aspects` on future runs.

**Out of memory during step 2**
Reduce the number of documents being processed, or switch to a smaller model in `config.py` by changing `LLM_MODEL = "qwen2.5:7b"`.

**Low Hit@K score printed by step 3**
The aspect-based retrieval did not find the right chunks. Try lowering `MATCH_THRESHOLD` in `config.py` from `0.40` to `0.30`, or increasing `TOP_K_CHUNKS` from `50` to `100`.

---

## Output Folder Structure

After a full run, your `outputs/` folder will contain:

```
outputs/
├── chunks.json              ← all document chunks (from step 1)
├── triplets.jsonl           ← all extracted triplets (from step 2)
├── aspects.json             ← aspect centroids per chunk (from step 2)
├── retrieval_results.json   ← top-K chunks per question (from step 3)
├── answers.jsonl            ← generated answers (from step 4)
├── evaluation.jsonl         ← per-question judge scores (from step 5)
└── eval_report.txt          ← summary statistics (from step 5)
```

---

## Dependencies Reference

| Library | Purpose |
|---------|---------|
| `sentence-transformers` | Embedding model for triplets and queries |
| `scikit-learn` | DBSCAN clustering for aspect building |
| `langchain-ollama` | Connecting to Ollama for LLM calls |
| `tqdm` | Progress bars |
| `numpy` | Vector maths |
| `ollama` | Direct Ollama API calls for answer generation |
