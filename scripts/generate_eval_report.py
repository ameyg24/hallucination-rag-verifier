import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime, timezone

DEFAULT_IN_CSV = Path("results/eval_results.csv")
DEFAULT_OUT_JSON = Path("results/eval_summary.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-csv", type=Path, default=DEFAULT_IN_CSV)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    args = parser.parse_args()

    if not args.in_csv.exists():
        raise SystemExit(f"Missing {args.in_csv}. Run scripts/run_eval.py first.")

    total = 0
    expected_unsupported = 0
    false_supported_on_unsupported = 0

    expected_supported = 0
    correct_supported = 0

    covered = 0

    with args.in_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            total += 1
            expected = row.get("expected", "supported")
            verdict = row.get("verdict", "uncertain")

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
    coverage_rate = (covered / total) if total else 0.0

    report: Dict[str, Any] = {
        "total_cases": total,
        "coverage_rate": round(coverage_rate, 3),
        "supported_accuracy": round(supported_accuracy, 3),
        "hallucination_rate": round(hallucination_rate, 3),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Wrote", args.out_json)


if __name__ == "__main__":
    main()
