import json
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    ds = load_dataset("pminervini/HaluEval", "qa_samples", split="data")

    eval_cases = []
    corpus_docs = []
    selected_counts = {"yes": 0, "no": 0}

    for row in ds:
        label = row["hallucination"]
        if label not in selected_counts or selected_counts[label] >= 500:
            continue

        case_id = f"{label}_{selected_counts[label]}"
        expected = "unsupported" if label == "yes" else "supported"
        corpus_docs.append(
            {
                "doc_id": f"halueval_{case_id}",
                "text": f"{row['question']} {row['knowledge']}",
            }
        )
        eval_cases.append(
            {
                "id": case_id,
                "query": row["question"],
                "claim": row["answer"],
                "expected": expected,
            }
        )
        selected_counts[label] += 1

        if selected_counts["yes"] == 500 and selected_counts["no"] == 500:
            break

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    (data_dir / "halueval_eval_cases.jsonl").write_text(
        "\n".join(json.dumps(case) for case in eval_cases) + "\n",
        encoding="utf-8",
    )
    (data_dir / "halueval_corpus.jsonl").write_text(
        "\n".join(json.dumps(doc) for doc in corpus_docs) + "\n",
        encoding="utf-8",
    )

    print(f"Eval cases: {len(eval_cases)}")
    print(f"Corpus docs: {len(corpus_docs)}")


if __name__ == "__main__":
    main()
