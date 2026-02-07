import json
import random
from pathlib import Path
from typing import List, Dict

OUT_JSONL = Path("data/eval_cases.jsonl")


SUPPORTED_TEMPLATES = [
    "FAISS supports similarity search over dense vectors.",
    "FAISS is commonly used for nearest neighbor retrieval.",
    "BM25 is a sparse retrieval method for keyword-based ranking.",
    "Hybrid retrieval combines dense retrieval with sparse retrieval.",
    "RAG retrieves relevant context first and then generates an answer.",
    "FAISS is not a keyword ranking algorithm.",
    "BM25 is the classic keyword ranking method.",
    "BM25 does not store dense embeddings.",
    "BM25 ranks documents using lexical term statistics.",
    "A verifier labels claims as supported when evidence directly backs them.",
    "Dense retrieval uses vectors and similarity.",
    "Sparse retrieval uses keywords and BM25 scoring.",
    "Hybrid retrieval uses FAISS for dense retrieval and BM25 for sparse retrieval.",
    "RAG answers are grounded in retrieved context, often with citations.",
    "The diagram shows query -> retrieve (BM25 + FAISS) -> generate -> verify with citations.",
]

UNSUPPORTED_TEMPLATES = [
    "FAISS is a keyword ranking algorithm.",
    "BM25 stores dense embeddings in a vector database.",
    "RAG generates answers before retrieving context.",
    "Hybrid retrieval combines only keyword ranking methods.",
    "BM25 is a dense retrieval method using vectors.",
    "FAISS is a sparse keyword scorer.",
    "Dense retrieval relies on keywords and BM25.",
    "Sparse retrieval uses dense vector similarity.",
    "The diagram shows query -> generate -> retrieve -> verify.",
    "A verifier labels claims as supported when evidence is missing.",
    "BM25 is deprecated.",
    "FAISS stores lexical term statistics for documents.",
    "RAG does not use citations.",
    "Hybrid retrieval ignores BM25.",
    "BM25 is used for nearest neighbor search.",
]


def build_cases(target: int, seed: int = 42) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    cases: List[Dict[str, str]] = []

    supported_target = target // 2
    unsupported_target = target - supported_target

    for idx in range(supported_target):
        claim = rng.choice(SUPPORTED_TEMPLATES)
        cases.append(
            {
                "id": f"supported-{idx+1}",
                "query": claim,
                "expected": "supported",
            }
        )

    for idx in range(unsupported_target):
        claim = rng.choice(UNSUPPORTED_TEMPLATES)
        cases.append(
            {
                "id": f"unsupported-{idx+1}",
                "query": claim,
                "expected": "unsupported",
            }
        )

    rng.shuffle(cases)
    return cases


def main() -> None:
    target = 600
    cases = build_cases(target=target, seed=42)
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(json.dumps(case, ensure_ascii=False) + "\n")
    print("Wrote", OUT_JSONL, "rows:", len(cases))


if __name__ == "__main__":
    main()
