import json
from pathlib import Path
from typing import Dict, Any, Iterable

IN_JSONL = Path("data/eval_cases.jsonl")
OUT_JSONL = Path("data/eval_cases_large.jsonl")


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line_txt in fh:
            line_txt = line_txt.strip()
            if not line_txt:
                continue
            yield json.loads(line_txt)


def main() -> None:
    if not IN_JSONL.exists():
        raise SystemExit(f"Missing {IN_JSONL}.")

    # This script duplicates the small sample set to create a larger test set.
    # Replace this with a real dataset generator for 500+ cases.
    rows = list(read_jsonl(IN_JSONL))
    if not rows:
        raise SystemExit("No rows found in eval_cases.jsonl")

    target = 500
    out = []
    idx = 0
    while len(out) < target:
        row = dict(rows[idx % len(rows)])
        row["id"] = f"{row.get('id', 'case')}-{len(out)+1}"
        out.append(row)
        idx += 1

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w", encoding="utf-8") as fh:
        for row in out:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Wrote", OUT_JSONL, "rows:", len(out))


if __name__ == "__main__":
    main()
