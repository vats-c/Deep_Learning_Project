import argparse
import json
import logging
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import ollama
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_INPUT_FILE = "BMP_Eval/Hotpot/output.md"
DEFAULT_OUTPUT_FILE = "BMP_Eval/Hotpot/evaluation_scores_hotpot.json"
DEFAULT_JUDGE_MODEL = "qwen2.5:14b"


PAIR_BLOCK_RE = re.compile(
    r"###\s*Pair\s*(\d+)\s*.*?"
    r"\*\*Question:\*\*\s*(.*?)\n"
    r"\*\*Ground Truth:\*\*\s*(.*?)\n"
    r"\*\*Retrieved Answer:\*\*\s*(.*?)(?=\n---|\n###\s*Pair|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def parse_kgi_markdown(md_text: str) -> List[Dict[str, Any]]:
    items = []
    for m in PAIR_BLOCK_RE.finditer(md_text):
        pair_id = int(m.group(1))
        question = m.group(2).strip()
        ground_truth = m.group(3).strip()
        predicted = m.group(4).strip()

        items.append({
            "qid": pair_id,
            "question": question,
            "ground_truth": ground_truth,
            "predicted_answer": predicted,
        })
    return items


def load_kgi_output(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    items = parse_kgi_markdown(text)
    if not items:
        raise ValueError(f"No QA pairs found in markdown file: {path}")
    return items


NEGATION_MARKERS = [
    " not ", " no ", " never ", " except ", " instead ", " rather than ",
    "without", "false", "incorrect", "wrong"
]


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[\u2018\u2019\u201c\u201d]", "'", text)
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_yes_no(text: str) -> Optional[str]:
    m = re.match(r"^\s*(yes|no)\b", text.strip(), re.IGNORECASE)
    return m.group(1).lower() if m else None


def fast_semantic_match(predicted: str, ground_truth: str) -> bool:
    p = normalize_text(predicted)
    g = normalize_text(ground_truth)

    if not p or not g:
        return False

    if p == g:
        return True

    pred_yn = normalize_yes_no(predicted)
    gt_yn = normalize_yes_no(ground_truth)
    if pred_yn and gt_yn:
        return pred_yn == gt_yn

    if g in p and not any(marker in f" {p} " for marker in NEGATION_MARKERS):
        return True

    return False


JUDGE_PROMPT = """You are a careful semantic answer judge.

Decide whether the PREDICTED ANSWER is correct with respect to the GROUND TRUTH.

Rules:
- Judge meaning, not exact wording.
- Paraphrases are correct.
- If the predicted answer contains the ground truth answer in a longer phrase, mark it correct unless the extra text changes or negates the meaning.
- Ignore case, punctuation, articles, and minor formatting differences.
- For names, places, dates, numbers, and short facts, accept equivalent surface forms.
- Mark incorrect only when the answer is clearly wrong, contradictory, or missing the key answer.

Return ONLY valid JSON with these keys:
- verdict: "correct" or "incorrect"
- score: 1 or 0
- reason: short phrase, max 12 words

Question:
{question}

Ground Truth:
{ground_truth}

Predicted Answer:
{predicted}
"""


def parse_judge_response(raw: str) -> Optional[Dict[str, Any]]:
    raw = raw.strip()

    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            verdict = str(obj.get("verdict", "")).strip().lower()
            score = obj.get("score", None)
            reason = str(obj.get("reason", "")).strip()

            if verdict in {"correct", "incorrect"}:
                if isinstance(score, str) and score.isdigit():
                    score = int(score)
                elif isinstance(score, (int, float)):
                    score = int(score)
                else:
                    score = 1 if verdict == "correct" else 0

                if score in {0, 1}:
                    return {
                        "verdict": verdict,
                        "score": score,
                        "reason": reason,
                        "raw": raw,
                        "method": "llm_json",
                    }
        except Exception:
            pass

    if re.search(r"\bcorrect\b", raw, re.IGNORECASE):
        return {
            "verdict": "correct",
            "score": 1,
            "reason": "",
            "raw": raw,
            "method": "llm_text",
        }
    if re.search(r"\bincorrect\b", raw, re.IGNORECASE):
        return {
            "verdict": "incorrect",
            "score": 0,
            "reason": "",
            "raw": raw,
            "method": "llm_text",
        }

    m = re.search(r"\b([01])\b", raw)
    if m:
        score = int(m.group(1))
        return {
            "verdict": "correct" if score == 1 else "incorrect",
            "score": score,
            "reason": "",
            "raw": raw,
            "method": "llm_text",
        }

    return None


def judge_with_qwen(question: str, ground_truth: str, predicted: str, model: str) -> Dict[str, Any]:
    prompt = JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        predicted=predicted
    )

    last_raw = ""
    for attempt in range(4):
        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict but fair semantic equivalence judge."
                    },
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": 0.0, "num_predict": 128},
            )
            raw = response["message"]["content"].strip()
            last_raw = raw

            parsed = parse_judge_response(raw)
            if parsed is not None:
                return parsed

            raise ValueError(f"Could not parse judge output: {raw[:200]!r}")

        except Exception as e:
            logger.warning(f"Judge attempt {attempt + 1}/4 failed: {e}")

    return {
        "verdict": "incorrect",
        "score": 0,
        "reason": "judge_failed",
        "raw": last_raw,
        "method": "judge_failed",
    }


def evaluate_item(item: Dict[str, Any], judge_model: str) -> Dict[str, Any]:
    qid = item["qid"]
    question = item["question"]
    gold = item["ground_truth"]
    pred = item["predicted_answer"]

    if fast_semantic_match(pred, gold):
        return {
            "qid": qid,
            "question": question,
            "ground_truth": gold,
            "predicted_answer": pred,
            "verdict": "correct",
            "score": 1,
            "method": "fast_semantic_match",
            "reason": "containment_or_exact_match",
        }

    judged = judge_with_qwen(question, gold, pred, judge_model)
    return {
        "qid": qid,
        "question": question,
        "ground_truth": gold,
        "predicted_answer": pred,
        "verdict": judged["verdict"],
        "score": judged["score"],
        "method": judged["method"],
        "reason": judged.get("reason", ""),
        "raw": judged.get("raw", ""),
    }


def compute_summary(results: List[Dict[str, Any]], judge_model: str) -> Dict[str, Any]:
    total = len(results)
    correct = sum(1 for r in results if r["score"] == 1)
    fast = sum(1 for r in results if r["method"] == "fast_semantic_match")
    llm = sum(1 for r in results if r["method"] in {"llm_json", "llm_text"})

    return {
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": round((correct / total) * 100, 2) if total else 0.0,
        "fast_semantic_match": fast,
        "llm_judged": llm,
        "judge_model": judge_model,
    }


def print_summary(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print(f"JUDGE MODEL : {summary['judge_model']}")
    print(f"TOTAL       : {summary['total']}")
    print(f"CORRECT     : {summary['correct']}")
    print(f"INCORRECT   : {summary['incorrect']}")
    print(f"ACCURACY    : {summary['accuracy']}%")
    print(f"FAST MATCH   : {summary['fast_semantic_match']}")
    print(f"LLM JUDGED   : {summary['llm_judged']}")
    print("=" * 72 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Semantic evaluation for KGI Hotpot output")
    parser.add_argument("--input-file", default=DEFAULT_INPUT_FILE, help="KGI output markdown file")
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE, help="JSON file for evaluation results")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="Ollama judge model")
    args = parser.parse_args()

    logger.info(f"Loading input file: {args.input_file}")
    items = load_kgi_output(args.input_file)
    logger.info(f"Loaded {len(items)} QA pairs")

    results = []
    for item in tqdm(items, desc="Evaluating"):
        result = evaluate_item(item, args.judge_model)
        results.append(result)
        logger.info(f"Q{result['qid']} -> {result['score']}/1 [{result['method']}]")

    summary = compute_summary(results, args.judge_model)
    print_summary(summary)

    payload = {
        "summary": summary,
        "results": results,
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved evaluation to: {args.output_file}")


if __name__ == "__main__":
    main()