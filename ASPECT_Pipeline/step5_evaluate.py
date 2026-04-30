import json
import re
import os
import logging
from tqdm import tqdm
import ollama

from config import (
    ANSWERS_FILE, EVAL_FILE, EVAL_REPORT, OUTPUTS_DIR,
    JUDGE_MODEL, FACTUAL_TYPE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
log = logging.getLogger(__name__)


JUDGE_PROMPT = """You are a strict technical evaluator for a Knowledge-Graph QA system.
Compare the PREDICTED ANSWER against the GROUND TRUTH for the given question.

SCORING CRITERIA
  10 : Perfect — facts match exactly, same meaning.
   8-9 : Mostly correct — right facts, minor phrasing difference.
   5-7 : Partially correct — core idea right but missed a key detail or number.
   2-4 : Wrong — right entity, but value / relation is incorrect.
     1 : Complete failure, hallucination, or "insufficient information" when the truth exists.

[QUESTION]:         {question}
[GROUND TRUTH]:     {truth}
[PREDICTED ANSWER]: {prediction}

Respond with ONLY a single integer between 1 and 10.
No explanation. No words. Just the number.

SCORE:"""


def extract_score(text: str) -> int:
    match = re.search(r"\b(10|[1-9])\b", text.strip())
    return int(match.group(1)) if match else 0


def run():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    items = []
    with open(ANSWERS_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    factual = [i for i in items if i.get("type") == FACTUAL_TYPE]
    log.info(
        f"Answers file: {len(items)} total | "
        f"{len(factual)} Factual selected for evaluation."
    )

    if not factual:
        log.warning("No Factual questions found — nothing to evaluate.")
        return

    total_score  = 0
    correct      = 0
    partial      = 0
    failures     = 0
    scored: list = []

    with open(EVAL_FILE, "w", encoding="utf-8") as out:
        for item in tqdm(factual, desc="Judging answers"):
            prompt = JUDGE_PROMPT.format(
                question   = item["question"],
                truth      = item["gold_answer"],
                prediction = item["predicted_answer"],
            )
            try:
                response = ollama.chat(
                    model   = JUDGE_MODEL,
                    messages= [{"role": "user", "content": prompt}],
                    options = {"temperature": 0.0}
                )
                score = extract_score(response["message"]["content"])
            except Exception as exc:
                log.error(f"Judge failed for Q{item['qid']}: {exc}")
                score = 0

            item["eval_score"] = score
            total_score += score

            if score >= 8:
                correct += 1
            elif score >= 5:
                partial += 1
            else:
                failures += 1

            scored.append(item)
            out.write(json.dumps(item, ensure_ascii=False) + "\n")

    n   = len(scored)
    avg = total_score / n if n > 0 else 0

    report = f"""
====================================================
EVALUATION REPORT — FACTUAL QUESTIONS ONLY
====================================================
Total Factual Questions   : {n}
Average Judge Score       : {avg:.2f} / 10
----------------------------------------------------
Correct   (score >= 8)    : {correct:4d}  ({correct / max(n,1):.1%})
Partial   (5 <= score < 8): {partial:4d}  ({partial / max(n,1):.1%})
Failures  (score <  5)    : {failures:4d}  ({failures / max(n,1):.1%})
====================================================
Evaluation details  → {EVAL_FILE}
====================================================
"""

    print(report)
    with open(EVAL_REPORT, "w", encoding="utf-8") as f:
        f.write(report)

    dist = {}
    for item in scored:
        s = item["eval_score"]
        dist[s] = dist.get(s, 0) + 1

    print("Score distribution:")
    for score in sorted(dist):
        bar = "█" * dist[score]
        print(f"  {score:2d} : {bar}  ({dist[score]})")

    log.info(f"Evaluation saved → {EVAL_FILE}")
    log.info(f"Report saved     → {EVAL_REPORT}")


if __name__ == "__main__":
    run()