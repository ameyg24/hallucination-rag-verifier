import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import (
    TOP_K,
    FAISS_WEIGHT,
    BM25_WEIGHT,
    VERIFIER_SUPPORTED_MIN_OVERLAP,
    VERIFIER_UNCERTAIN_MIN_OVERLAP,
)
from app.persist import load_store
from app.retrieval import HybridRetriever
from app.verify import verify_with_claims

DATASET_PATH = Path("data/eval_cases.jsonl")
OUT_JSON = Path("results/tuning_summary.json")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_txt in fh:
            line_txt = line_txt.strip()
            if not line_txt:
                continue
            rows.append(json.loads(line_txt))
    return rows


def evaluate(
    cases: List[Dict[str, Any]],
    retriever: HybridRetriever,
    top_k: int,
    w_faiss: float,
    w_bm25: float,
    supported_min_overlap: int,
    uncertain_min_overlap: int,
) -> Dict[str, float]:
    total = 0
    expected_unsupported = 0
    false_supported_on_unsupported = 0
    expected_supported = 0
    correct_supported = 0
    covered = 0

    for case in cases:
        query = case.get("query", "")
        expected = case.get("expected", "supported")

        if not query:
            continue

        results = retriever.search(
            query,
            top_k=top_k,
            w_faiss=w_faiss,
            w_bm25=w_bm25,
        )
        verdict, _, _ = verify_with_claims(
            query=query,
            draft_answer=query,
            evidence=results,
            supported_min_overlap=supported_min_overlap,
            uncertain_min_overlap=uncertain_min_overlap,
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
    if not DATASET_PATH.exists():
        raise SystemExit(f"Missing {DATASET_PATH}. Run scripts/generate_eval_cases.py first.")

    cases = load_jsonl(DATASET_PATH)
    texts, metas = load_store()
    retriever = HybridRetriever()
    retriever.build(texts, metas)

    baseline = evaluate(
        cases=cases,
        retriever=retriever,
        top_k=TOP_K,
        w_faiss=FAISS_WEIGHT,
        w_bm25=BM25_WEIGHT,
        supported_min_overlap=VERIFIER_SUPPORTED_MIN_OVERLAP,
        uncertain_min_overlap=VERIFIER_UNCERTAIN_MIN_OVERLAP,
    )

    grid_top_k = [3, 5, 6, 8, 10]
    grid_faiss = [0.2, 0.4, 0.6, 0.8]
    grid_supported = [4, 5, 6]
    grid_uncertain = [2, 3]

    best = None
    for top_k in grid_top_k:
        for w_faiss in grid_faiss:
            w_bm25 = round(1.0 - w_faiss, 2)
            for supported_min_overlap in grid_supported:
                for uncertain_min_overlap in grid_uncertain:
                    metrics = evaluate(
                        cases=cases,
                        retriever=retriever,
                        top_k=top_k,
                        w_faiss=w_faiss,
                        w_bm25=w_bm25,
                        supported_min_overlap=supported_min_overlap,
                        uncertain_min_overlap=uncertain_min_overlap,
                    )
                    candidate = {
                        "top_k": top_k,
                        "w_faiss": w_faiss,
                        "w_bm25": w_bm25,
                        "supported_min_overlap": supported_min_overlap,
                        "uncertain_min_overlap": uncertain_min_overlap,
                        "metrics": metrics,
                    }
                    if best is None:
                        best = candidate
                        continue
                    if metrics["hallucination_rate"] < best["metrics"]["hallucination_rate"]:
                        best = candidate
                        continue
                    if metrics["hallucination_rate"] == best["metrics"]["hallucination_rate"]:
                        if metrics["supported_accuracy"] > best["metrics"]["supported_accuracy"]:
                            best = candidate

    reduction = None
    if baseline["hallucination_rate"] > 0:
        reduction = 100.0 * (baseline["hallucination_rate"] - best["metrics"]["hallucination_rate"]) / baseline["hallucination_rate"]

    summary = {
        "baseline": {
            "top_k": TOP_K,
            "w_faiss": FAISS_WEIGHT,
            "w_bm25": BM25_WEIGHT,
            "supported_min_overlap": VERIFIER_SUPPORTED_MIN_OVERLAP,
            "uncertain_min_overlap": VERIFIER_UNCERTAIN_MIN_OVERLAP,
            "metrics": baseline,
        },
        "best": best,
        "hallucination_reduction_pct": round(reduction or 0.0, 2),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Baseline hallucination rate:", round(baseline["hallucination_rate"], 3))
    print("Best hallucination rate:", round(best["metrics"]["hallucination_rate"], 3))
    print("Hallucination reduction %:", round(reduction or 0.0, 2))
    print("Best params:", {k: best[k] for k in ("top_k", "w_faiss", "w_bm25")})
    print("Summary saved:", OUT_JSON)


if __name__ == "__main__":
    main()
