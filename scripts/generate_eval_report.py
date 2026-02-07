import csv
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime, timezone

IN_CSV = Path("results/eval_results.csv")
OUT_JSON = Path("results/eval_summary.json")


def main() -> None:
    if not IN_CSV.exists():
        raise SystemExit(f"Missing {IN_CSV}. Run scripts/run_eval.py first.")

    total = 0
    expected_unsupported = 0
    false_supported_on_unsupported = 0

    expected_supported = 0
    correct_supported = 0

    covered = 0

    with IN_CSV.open("r", encoding="utf-8", newline="") as fh:
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

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Wrote", OUT_JSON)


if __name__ == "__main__":
    main()
