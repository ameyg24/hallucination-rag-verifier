import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import requests

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
from app.llm import generate_answer_with_evidence
from app.prompts import PROMPT_VARIANTS

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")
DEFAULT_DATASET_PATH = Path("data/eval_cases.jsonl")
DEFAULT_OUT_CSV = Path("results/eval_results.csv")
VALID_EXPECTED_LABELS = {"supported", "unsupported"}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for idx, line_txt in enumerate(fh, start=1):
            line_txt = line_txt.strip()
            if not line_txt:
                continue
            try:
                rows.append(json.loads(line_txt))
            except json.JSONDecodeError as err:
                raise RuntimeError(f"Invalid JSON on line {idx} in {path}: {err}")
    return rows


def evaluate_local(
    cases: List[Dict[str, Any]],
    top_k: int,
    w_faiss: float,
    w_bm25: float,
    supported_min_overlap: int,
    uncertain_min_overlap: int,
    use_llm: bool,
    prompt_name: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    texts, metas = load_store()
    if not texts:
        raise RuntimeError("indexed_items is 0. Load or ingest corpus before evaluation.")

    retriever = HybridRetriever()
    try:
        retriever.build(texts, metas)
    except ValueError as err:
        raise RuntimeError(f"Invalid local index: {err}")

    total = 0
    expected_unsupported = 0
    false_supported_on_unsupported = 0
    expected_supported = 0
    correct_supported = 0
    covered = 0

    results_rows: List[Dict[str, Any]] = []

    prompt_text = None
    if use_llm:
        for variant in PROMPT_VARIANTS:
            if variant["name"] == prompt_name:
                prompt_text = variant["text"]
                break
        if not prompt_text:
            raise RuntimeError(f"Unknown prompt name: {prompt_name}")

    for case in cases:
        cid = case.get("id", "")
        query = case.get("query", "")
        expected = case.get("expected", "supported")
        if expected not in VALID_EXPECTED_LABELS:
            raise RuntimeError(
                f"Invalid expected label for case id='{cid}': '{expected}'. "
                f"Allowed labels: {sorted(VALID_EXPECTED_LABELS)}"
            )

        if not query:
            continue

        results = retriever.search(
            query,
            top_k=top_k,
            w_faiss=w_faiss,
            w_bm25=w_bm25,
        )
        draft_answer = query
        if use_llm:
            draft_answer = generate_answer_with_evidence(
                query=query,
                evidence=results,
                prompt=prompt_text,
            )

        verdict, _, _ = verify_with_claims(
            query=query,
            draft_answer=draft_answer,
            evidence=results,
            supported_min_overlap=supported_min_overlap,
            uncertain_min_overlap=uncertain_min_overlap,
        )

        evidence_count = len(results)
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

        results_rows.append(
            {
                "id": cid,
                "query": query,
                "expected": expected,
                "verdict": verdict,
                "evidence_count": evidence_count,
            }
        )

    hallucination_rate = (false_supported_on_unsupported / expected_unsupported) if expected_unsupported else 0.0
    supported_accuracy = (correct_supported / expected_supported) if expected_supported else 0.0
    coverage_rate = covered / total if total else 0.0

    metrics = {
        "total": total,
        "coverage_rate": coverage_rate,
        "supported_accuracy": supported_accuracy,
        "hallucination_rate": hallucination_rate,
    }
    return results_rows, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Run evaluation locally without the API server.")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--faiss-weight", type=float, default=FAISS_WEIGHT)
    parser.add_argument("--bm25-weight", type=float, default=BM25_WEIGHT)
    parser.add_argument("--supported-min-overlap", type=int, default=VERIFIER_SUPPORTED_MIN_OVERLAP)
    parser.add_argument("--uncertain-min-overlap", type=int, default=VERIFIER_UNCERTAIN_MIN_OVERLAP)
    parser.add_argument("--use-llm", action="store_true", help="Generate draft answers with OpenAI.")
    parser.add_argument("--prompt-name", type=str, default="strict_grounding_v1")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    args = parser.parse_args()

    print("run_eval starting...")
    print("cwd:", Path(".").resolve())
    print("dataset:", args.dataset_path.resolve())

    if not args.dataset_path.exists():
        print("ERROR: dataset file not found:", args.dataset_path)
        sys.exit(1)

    try:
        cases = load_jsonl(args.dataset_path)
    except Exception as err:
        print("ERROR: failed to read dataset:", repr(err))
        sys.exit(1)
    print("cases loaded:", len(cases))

    start_ts = time.time()

    if args.local:
        try:
            results_rows, metrics = evaluate_local(
                cases=cases,
                top_k=args.top_k,
                w_faiss=args.faiss_weight,
                w_bm25=args.bm25_weight,
                supported_min_overlap=args.supported_min_overlap,
                uncertain_min_overlap=args.uncertain_min_overlap,
                use_llm=args.use_llm,
                prompt_name=args.prompt_name,
            )
        except Exception as err:
            print("ERROR during local evaluation:", repr(err))
            sys.exit(1)

        total = metrics["total"]
        coverage_rate = metrics["coverage_rate"]
        supported_accuracy = metrics["supported_accuracy"]
        hallucination_rate = metrics["hallucination_rate"]
    else:
        # Check API health
        try:
            resp = requests.get(f"{BASE_URL}/health", timeout=5)
            print("health status:", resp.status_code, resp.text)

            health = resp.json()
            if health.get("indexed_items", 0) == 0:
                print("ERROR: indexed_items is 0. Load or ingest corpus before evaluation.")
                sys.exit(1)

            resp.raise_for_status()
        except Exception as err:
            print("ERROR: API not reachable at", BASE_URL)
            print("Exception:", repr(err))
            sys.exit(1)

        total = 0
        expected_unsupported = 0
        false_supported_on_unsupported = 0
        expected_supported = 0
        correct_supported = 0
        covered = 0
        results_rows = []

        for case in cases:
            cid = case.get("id", "")
            query = case.get("query", "")
            expected = case.get("expected", "supported")
            if expected not in VALID_EXPECTED_LABELS:
                print(
                    f"ERROR: Invalid expected label for case id='{cid}': '{expected}'. "
                    f"Allowed labels: {sorted(VALID_EXPECTED_LABELS)}"
                )
                sys.exit(1)

            if not query:
                continue

            try:
                resp = requests.post(f"{BASE_URL}/query", json={"query": query}, timeout=30)
                status = resp.status_code
                body_txt = resp.text
                resp.raise_for_status()
                out = resp.json()
            except Exception as err:
                print("ERROR during request")
                print("case id:", cid)
                print("query:", query)
                print("http status:", locals().get("status", None))
                print("body:", locals().get("body_txt", "")[:300])
                print("exception:", repr(err))
                sys.exit(1)

            verdict = out.get("verdict", "uncertain")
            evidence_count = len(out.get("evidence", []))

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

            results_rows.append(
                {
                    "id": cid,
                    "query": query,
                    "expected": expected,
                    "verdict": verdict,
                    "evidence_count": evidence_count,
                }
            )

        if total == 0:
            print("No valid cases found (check that each line has a non-empty 'query').")
            sys.exit(1)

        hallucination_rate = (false_supported_on_unsupported / expected_unsupported) if expected_unsupported else 0.0
        supported_accuracy = (correct_supported / expected_supported) if expected_supported else 0.0
        coverage_rate = covered / total if total else 0.0

    elapsed = time.time() - start_ts

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "query", "expected", "verdict", "evidence_count"])
        writer.writeheader()
        writer.writerows(results_rows)

    print("Eval complete")
    print("Total cases:", total)
    print("Coverage rate:", round(coverage_rate, 3))
    print("Supported accuracy:", round(supported_accuracy, 3))
    print("Hallucination rate:", round(hallucination_rate, 3))
    print("Results saved:", args.out_csv.resolve())
    print("Elapsed seconds:", round(elapsed, 2))


if __name__ == "__main__":
    main()
