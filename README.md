# AI Hallucination Detection & Multimodal RAG Verification Tool

FastAPI service that verifies claims using hybrid retrieval (FAISS + BM25) over text and image evidence. It returns a verdict (`supported` / `unsupported` / `uncertain`) with retrieved evidence snippets.

Includes local scripts for HaluEval-backed evaluation, synthetic smoke tests, and retrieval/prompt tuning.

## Features
- Hybrid retrieval: dense (FAISS) + sparse (BM25)
- Multimodal evidence: text chunks and image captions
- Simple claim verification with contradiction heuristics
- Evaluation pipeline with CSV + JSON summary

## MVP Scope (Current)
This repo ships a demo index, a synthetic smoke-test set, and a HaluEval QA conversion for public real-data evaluation. It is **not** a production-grade verifier or a large-scale benchmark.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m uvicorn app.main:app --reload --port 8000
```

Set your OpenAI API key in the environment:

```bash
export OPENAI_API_KEY="your_key_here"
```

Load a saved index (or ingest new docs), then run evaluation:

```bash
python3 scripts/run_eval.py --local
python3 scripts/generate_eval_report.py
```

Results are written to:
- `results/eval_results.csv`
- `results/eval_summary.json`

## API Endpoints
- `GET /health`
- `GET /stats`
- `POST /reset`
- `POST /save_index`
- `POST /load_index`
- `POST /ingest/text`
- `POST /ingest/image`
- `POST /query`

Example query:

```bash
curl -s -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"FAISS supports similarity search over dense vectors."}' | jq .
```

## Evaluation
Edit `data/eval_cases.jsonl` or swap in a larger dataset. Each line should look like:

```json
{"id":"1","query":"...", "expected":"supported"}
```

You can point to another server via:

```bash
BASE_URL=http://127.0.0.1:8000 python3 scripts/run_eval.py
```

To generate a synthetic 600‑case dataset (supported + unsupported):

```bash
python3 scripts/generate_eval_cases.py
```

To run eval locally without the API server:

```bash
python3 scripts/run_eval.py --local
```

To control input/output paths without editing code:

```bash
python3 scripts/run_eval.py --local \
  --dataset-path data/eval_cases.jsonl \
  --out-csv results/eval_results.csv
```

To generate and evaluate the HaluEval QA dataset:

```bash
python3 scripts/load_halueval.py
python3 scripts/ingest_halueval.py
curl -sS -X POST http://127.0.0.1:8000/save_index
python3 scripts/run_eval.py --local \
  --dataset-path data/halueval_eval_cases.jsonl \
  --out-csv results/halueval_eval_results.csv
python3 scripts/generate_eval_report.py \
  --in-csv results/halueval_eval_results.csv \
  --out-json results/eval_summary.json
```

To run eval locally with OpenAI-generated draft answers:

```bash
python3 scripts/run_eval.py --local --use-llm --prompt-name strict_grounding_v1
```

To tune retrieval weights and top‑k on the eval set:

```bash
python3 scripts/tune_retrieval.py
```

To tune prompts using OpenAI (prompt iteration):

```bash
python3 scripts/tune_prompt.py --max-cases 200
```

## Latest Verified Local Results (HaluEval QA)
- Evaluation scale: 1000 real HaluEval QA cases in `data/halueval_eval_cases.jsonl` (500 supported + 500 unsupported)
- Corpus scale: 1000 HaluEval evidence documents in `data/halueval_corpus.jsonl`, saved as 1456 indexed chunks
- Current HaluEval local run: coverage `0.939`, supported accuracy `0.926`, hallucination rate `0.004`
- Verifier tuning reduced HaluEval false-support hallucination rate from `0.530` to `0.004` (99.2% relative reduction) by requiring exact evidence containment for supported verdicts
- Synthetic eval remains available only as a smoke test in `data/eval_cases.jsonl`
- Current tuned defaults in `app/config.py`: `TOP_K=3`, `FAISS_WEIGHT=0.2`, `BM25_WEIGHT=0.8`, `VERIFIER_SUPPORTED_MIN_OVERLAP=6`, `VERIFIER_UNCERTAIN_MIN_OVERLAP=2`

Generate detailed summaries locally with:

```bash
python3 scripts/tune_retrieval.py
python3 scripts/generate_eval_report.py \
  --in-csv results/halueval_eval_results.csv \
  --out-json results/eval_summary.json
```

Then inspect `results/tuning_summary.json` and `results/eval_summary.json`.

## File Notes
- `scripts/build_eval_set.py` is an optional helper that duplicates cases into a larger JSONL file; it is not used by the default eval pipeline.

## Reproducibility Notes
- The default dataset in `data/eval_cases.jsonl` is synthetic for smoke testing and tuning.
- The HaluEval QA dataset in `data/halueval_eval_cases.jsonl` is the public real-data baseline used for the latest reported metrics.
- For credible metrics, keep the dataset and summary JSON in the repo (or publish them) so the numbers can be verified.

## Roadmap
- Expand exact-containment support logic with semantic entailment for paraphrased evidence.
- Add more public real-world datasets or internal benchmarks.
- Add stronger verification logic (claim parsing, contradiction detection, and citation scoring).

## License
MIT
