import argparse
import json
import os
import time
from datetime import datetime
from typing import List, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required.")
client = OpenAI(api_key=api_key)

SCORING_RUBRICS = {
    "completeness": {
        "description": "whether the answer includes ALL important facts and distinct points from the ground truth, allowing consistent factual additions",
        "rubric": (
            "Scoring Guide (0-10):\n"
            "- 10: Fully captures all Ground Truth facts with possibly helpful relevant detail.\n"
            "- 8-9: Covers most facts clearly with minor omissions or some additional context that does not contradict.\n"
            "- 6-7: Captures some key facts but misses several points or adds moderately extraneous/non-contradictory info.\n"
            "- 4-5: Partial coverage with many omissions or questionable additional info.\n"
            "- 1-3: Contains little of the Ground Truth facts.\n"
            "- 0: No relevant facts are present or answer is misleading."
        )
    },
    "accuracy": {
        "description": "whether the answer is factually correct compared to ground truth, tolerating consistent elaborations",
        "rubric": (
            "Scoring Guide (0-10):\n"
            "- 10: Fully accurate; no factual errors.\n"
            "- 8-9: Mostly accurate with minor trivial errors or consistent additions.\n"
            "- 6-7: Some factual inaccuracies or minor misinterpretations.\n"
            "- 4-5: Several incorrect points.\n"
            "- 1-3: Largely incorrect.\n"
            "- 0: Completely false or unrelated."
        )
    },
    "knowledgeability": {
        "description": "whether the answer shows accurate domain knowledge consistent with the ground truth, allowing relevant expansions",
        "rubric": (
            "Scoring Guide (0-10):\n"
            "- 10: Fully matches domain knowledge with clarity.\n"
            "- 8-9: Mostly aligns with minor gaps or some relevant added detail.\n"
            "- 6-7: Exhibits some understanding but also gaps.\n"
            "- 4-5: Limited knowledge shown.\n"
            "- 1-3: Minimal or incorrect domain knowledge.\n"
            "- 0: No relevant domain knowledge."
        )
    },
    "relevance": {
        "description": "whether the answer stays on-topic using only ground truth facts or consistent relevant information",
        "rubric": (
            "Scoring Guide (0-10):\n"
            "- 10: Entirely relevant and on-topic.\n"
            "- 8-9: Mostly relevant; minimal off-topic content.\n"
            "- 6-7: Some minor digressions.\n"
            "- 4-5: Noticeable off-topic content.\n"
            "- 1-3: Barely related.\n"
            "- 0: Completely irrelevant."
        )
    },
    "logical_coherence": {
        "description": "whether the answer presents the ground truth facts clearly and logically, with possible well-integrated expansions",
        "rubric": (
            "Scoring Guide (0-10):\n"
            "- 10: Clear, well-structured, logically coherent.\n"
            "- 8-9: Mostly clear with minor flow issues.\n"
            "- 6-7: Some structure but less clear.\n"
            "- 4-5: Poorly organized.\n"
            "- 1-3: Very hard to follow.\n"
            "- 0: Completely incoherent."
        )
    }
}


def parse_qa_pairs_from_md(filepath: str) -> List[Tuple[str, str, str]]:
    pairs = []
    current = {"q": None, "gt": None, "pred": None}
    collecting_pred = False

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            stripped = line.strip()

            if stripped.startswith("**Question:**"):
                if current["q"] and current["gt"] and current["pred"]:
                    pairs.append((current["q"], current["gt"], current["pred"].strip()))
                current = {
                    "q": stripped.replace("**Question:**", "", 1).strip(),
                    "gt": None, "pred": None
                }
                collecting_pred = False

            elif stripped.startswith("**Ground Truth:**"):
                current["gt"] = stripped.replace("**Ground Truth:**", "", 1).strip()
                collecting_pred = False

            elif stripped.startswith("**Retrieved Answer:**"):
                current["pred"] = stripped.replace("**Retrieved Answer:**", "", 1).strip()
                collecting_pred = True

            elif stripped == "---":
                if current["q"] and current["gt"] and current["pred"]:
                    pairs.append((current["q"], current["gt"], current["pred"].strip()))
                current = {"q": None, "gt": None, "pred": None}
                collecting_pred = False

            elif collecting_pred and stripped != "":
                current["pred"] = current["pred"] + " " + stripped

    if current["q"] and current["gt"] and current["pred"]:
        pairs.append((current["q"], current["gt"], current["pred"].strip()))

    return pairs


def build_scoring_prompt(question: str, ground_truth: str, retrieved_answer: str,
                         criterion: str, description: str, rubric: str) -> str:
    return f"""You are an impartial evaluation judge.

You are given:

Question:
\"\"\"{question}\"\"\"

Ground Truth Answer:
\"\"\"{ground_truth}\"\"\"

Retrieved Answer:
\"\"\"{retrieved_answer}\"\"\"

Your task:
Evaluate how well the retrieved answer captures ALL relevant factual information in the Ground Truth Answer, considering the context of the Question.

- The retrieved answer should fully include every important fact from the Ground Truth Answer.
- Relevant facts present in the Question but not explicitly in the Ground Truth Answer may be included without penalty.
- The retrieved answer should not omit key facts from the Ground Truth Answer.
- The retrieved answer should not contain incorrect facts or contradictions relative to both the Ground Truth and the Question.

Your evaluation must be based on the criterion: {criterion} — {description}

Scoring Rubric:
{rubric}

Provide output ONLY in this exact JSON format (no extra text, no markdown):
{{
  "retrieved": {{"score": <integer 0 to 10>}}
}}
""".strip()


def run_sequential_evaluation(pairs: List[Tuple[str, str, str]]):
    organized = {}

    total_requests = len(pairs) * len(SCORING_RUBRICS)

    with tqdm(total=total_requests, desc="Evaluating") as pbar:
        for pair_idx, (question, gt, retrieved) in enumerate(pairs):
            for criterion, details in SCORING_RUBRICS.items():
                prompt = build_scoring_prompt(
                    question, gt, retrieved,
                    criterion, details["description"], details["rubric"]
                )

                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are an impartial evaluation judge. Always respond with valid JSON only."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0,
                        max_tokens=80,
                        response_format={"type": "json_object"}
                    )

                    content = response.choices[0].message.content.strip()

                    if content.startswith("```json"):
                        content = content.split("```json", 1)[1].rsplit("```", 1)[0].strip()
                    elif content.startswith("```"):
                        content = content.strip("```").strip()

                    parsed = json.loads(content)
                    score = int(float(parsed["retrieved"]["score"]))

                    if pair_idx not in organized:
                        organized[pair_idx] = {}
                    organized[pair_idx][criterion] = score

                except Exception as e:
                    print(f"Error at pair {pair_idx}, {criterion}: {e}")

                pbar.update(1)

    eval_results = []
    for idx in sorted(organized.keys()):
        if idx < len(pairs):
            q, gt, pred = pairs[idx]
            scores_dict = {}
            for crit in SCORING_RUBRICS:
                scores_dict[crit] = organized[idx].get(crit, None)

            eval_results.append({
                "pair_index": idx + 1,
                "question": q,
                "ground_truth": gt,
                "retrieved_answer": pred,
                "scores": scores_dict
            })

    return eval_results


def calculate_averages(eval_results):
    avgs = {}
    for crit in SCORING_RUBRICS:
        values = [r["scores"][crit] for r in eval_results if r["scores"][crit] is not None]
        avgs[crit] = sum(values) / len(values) if values else None
    return avgs


def save_markdown_report(eval_results, averages, output_file: str):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Retrieval Evaluation – OpenAI Batch\n\n")
        f.write("## Average Scores\n\n")
        for crit, val in averages.items():
            v = f"{val:.2f}" if val is not None else "N/A"
            f.write(f"- **{crit.capitalize()}**: {v}\n")
        f.write("\n---\n\n## Individual Evaluations\n\n")

        for r in eval_results:
            f.write(f"### Pair {r['pair_index']}\n\n")
            f.write(f"**Question:**\n{r['question']}\n\n")
            f.write(f"**Ground Truth:**\n{r['ground_truth']}\n\n")
            f.write(f"**Retrieved Answer:**\n{r['retrieved_answer']}\n\n")
            f.write("**Scores:**\n")
            for crit, score in r["scores"].items():
                s = f"{score:.1f}" if score is not None else "N/A"
                f.write(f"- {crit.capitalize():14} {s}\n")
            f.write("\n---\n")

    print(f"Report saved → {output_file}")


def main():
    parser = argparse.ArgumentParser(description="OpenAI Sequential Evaluation – strict markdown format")
    parser.add_argument("--input", "-i", default="output.md", help="input markdown file")
    parser.add_argument("--output", "-o", default="openai_evaluation.md", help="output report")
    args = parser.parse_args()

    print("Loading evaluation data...")
    pairs = parse_qa_pairs_from_md(args.input)
    if not pairs:
        print("No valid QA pairs found.")
        return

    print(f"→ Found {len(pairs)} QA pairs")

    print("\nRunning sequential evaluation...")
    eval_results = run_sequential_evaluation(pairs)

    if eval_results:
        averages = calculate_averages(eval_results)
        save_markdown_report(eval_results, averages, args.output)

        print("\nFinal averages:")
        for k, v in averages.items():
            print(f"  {k:16} : {v:.2f}" if v is not None else f"  {k:16} : N/A")


if __name__ == "__main__":
    main()