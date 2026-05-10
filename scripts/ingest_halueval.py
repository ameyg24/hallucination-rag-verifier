import json
from pathlib import Path

import requests


BASE_URL = "http://127.0.0.1:8000"
CORPUS_PATH = Path("data/halueval_corpus.jsonl")


def main() -> None:
    docs = [
        json.loads(line)
        for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    for i, doc in enumerate(docs, start=1):
        response = requests.post(
            f"{BASE_URL}/ingest/text",
            json={
                "doc_id": doc["doc_id"],
                "text": doc["text"],
                "metadata": {"source": "halueval"},
            },
            timeout=30,
        )
        response.raise_for_status()
        if i % 50 == 0 or i == len(docs):
            total_indexed = response.json().get("total_indexed")
            print(f"{i}/{len(docs)} indexed={total_indexed}")

    print("Done ingesting")


if __name__ == "__main__":
    main()
