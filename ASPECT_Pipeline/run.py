import argparse
import logging
import subprocess
import sys
import os
from config import OUTPUTS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
log = logging.getLogger(__name__)


def run_step(label: str, cmd: list):
    log.info("")
    log.info("=" * 60)
    log.info(f"  STEP: {label}")
    log.info("=" * 60)

    result = subprocess.run(cmd)
    if result.returncode != 0:
        log.error(f"'{label}' failed with exit code {result.returncode}. Aborting.")
        sys.exit(result.returncode)

    log.info(f"  ✓  {label} complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Aspect-based KG-RAG Pipeline — full run"
    )
    parser.add_argument(
        "--qa", required=True,
        help="Path to the QA JSON file"
    )
    parser.add_argument(
        "--skip-ingest", action="store_true",
        help="Skip Step 1 (chunks.json already exists)"
    )
    parser.add_argument(
        "--skip-aspects", action="store_true",
        help="Skip Step 2 (aspects.json + triplets.jsonl already exist)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.qa):
        log.error(f"QA file not found: {args.qa}")
        sys.exit(1)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    py = sys.executable

    if not args.skip_ingest:
        run_step(
            "Step 1 — Ingest markdown files → chunks",
            [py, "step1_ingest.py"]
        )
    else:
        log.info("Skipping Step 1 (--skip-ingest)")

    if not args.skip_aspects:
        run_step(
            "Step 2 — Extract triplets & build aspect centroids",
            [py, "step2_aspects.py"]
        )
    else:
        log.info("Skipping Step 2 (--skip-aspects)")

    run_step(
        "Step 3 — Retrieve top-K chunks per query",
        [py, "step3_retrieve.py", "--qa", args.qa]
    )

    run_step(
        "Step 4 — K-hop graph traversal + answer generation",
        [py, "step4_answer.py"]
    )

    run_step(
        "Step 5 — LLM-as-Judge evaluation (Factual questions only)",
        [py, "step5_evaluate.py"]
    )

    log.info("")
    log.info("=" * 60)
    log.info("  PIPELINE COMPLETE")
    log.info(f"  All outputs saved to: {OUTPUTS_DIR}/")
    log.info("=" * 60)


if __name__ == "__main__":
    main()