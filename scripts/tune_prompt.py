import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import TOP_K, FAISS_WEIGHT, BM25_WEIGHT, VERIFIER_SUPPORTED_MIN_OVERLAP, VERIFIER_UNCERTAIN_MIN_OVERLAP
from app.persist import load_store
from app.retrieval import HybridRetriever
from app.verify import verify_with_claims
from app.llm import generate_answer_with_evidence
from app.prompts import PROMPT_VARIANTS

DATASET_PATH = Path("data/eval_cases.jsonl")
OUT_JSON = Path("results/prompt_tuning_summary.json")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_txt in fh:
            line_txt = line_txt.strip()
            if not line_txt:
                continue
            rows.append(json.loads(line_txt))
    return rows


def eval_prompt(
    cases: List[Dict[str, Any]],
    retriever: HybridRetriever,
    prompt_text: str,
    max_cases: int,
) -> Dict[str, float]:
    total = 0
    expected_unsupported = 0
    false_supported_on_unsupported = 0
    expected_supported = 0
    correct_supported = 0
    covered = 0

    for case in cases:
        if max_cases and total >= max_cases:
            break

        query = case.get("query", "")
        expected = case.get("expected", "supported")
        if not query:
            continue

        results = retriever.search(
            query,
            top_k=TOP_K,
            w_faiss=FAISS_WEIGHT,
            w_bm25=BM25_WEIGHT,
        )
        draft_answer = generate_answer_with_evidence(
            query=query,
            evidence=results,
            prompt=prompt_text,
        )
        verdict, _, _ = verify_with_claims(
            query=query,
            draft_answer=draft_answer,
            evidence=results,
            supported_min_overlap=VERIFIER_SUPPORTED_MIN_OVERLAP,
            uncertain_min_overlap=VERIFIER_UNCERTAIN_MIN_OVERLAP,
        )

        total += 1

        if verdict in ("supported", "uncertain"):
            covered += 1

        if expected == "unsupported":
            expected_unsupported += 1
            if verdict == "supported":
                false_supported_on_unsupported += 1

        if expected == "supported":
            expected_supported += 1
            if verdict == "supported":
                correct_supported += 1

    hallucination_rate = (false_supported_on_unsupported / expected_unsupported) if expected_unsupported else 0.0
    supported_accuracy = (correct_supported / expected_supported) if expected_supported else 0.0
    coverage_rate = covered / total if total else 0.0

    return {
        "total": total,
        "coverage_rate": coverage_rate,
        "supported_accuracy": supported_accuracy,
        "hallucination_rate": hallucination_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cases", type=int, default=0, help="Limit number of cases to reduce cost.")
    args = parser.parse_args()

    if not DATASET_PATH.exists():
        raise SystemExit(f"Missing {DATASET_PATH}. Run scripts/generate_eval_cases.py first.")

    cases = load_jsonl(DATASET_PATH)
    texts, metas = load_store()
    retriever = HybridRetriever()
    retriever.build(texts, metas)

    results = []
    start_ts = time.time()
    for variant in PROMPT_VARIANTS:
        metrics = eval_prompt(
            cases=cases,
            retriever=retriever,
            prompt_text=variant["text"],
            max_cases=args.max_cases,
        )
        results.append(
            {
                "name": variant["name"],
                "metrics": metrics,
            }
        )
        print("Prompt:", variant["name"], "hallucination rate:", round(metrics["hallucination_rate"], 3))

    best = min(results, key=lambda r: r["metrics"]["hallucination_rate"])
    summary = {
        "best_prompt": best,
        "all_prompts": results,
        "elapsed_seconds": round(time.time() - start_ts, 2),
        "max_cases": args.max_cases or len(cases),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Best prompt:", best["name"])
    print("Summary saved:", OUT_JSON)


if __name__ == "__main__":
    main()
