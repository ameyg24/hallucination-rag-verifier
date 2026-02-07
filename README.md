# AI Hallucination Detection & Multimodal RAG Verification Tool

FastAPI service that verifies claims using hybrid retrieval (FAISS + BM25) over text and image evidence. It returns a verdict (`supported` / `unsupported` / `uncertain`) with retrieved evidence snippets.

Includes local scripts for synthetic evaluation and retrieval/prompt tuning.

## Features
- Hybrid retrieval: dense (FAISS) + sparse (BM25)
- Multimodal evidence: text chunks and image captions
- Simple claim verification with contradiction heuristics
- Evaluation pipeline with CSV + JSON summary

## MVP Scope (Current)
This repo ships a demo index and a synthetic evaluation set for a working MVP. It is **not** a production‑grade verifier or a large‑scale benchmark. For credible metrics, plug in a real corpus aligned to a real evaluation dataset and re‑run the eval.

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

## Latest Verified Local Results (Synthetic Eval)
- Evaluation scale: 500+ synthetic cases (current run: 600 in `data/eval_cases.jsonl`)
- Hallucination reduction target met: 30%+ after retrieval tuning and prompt iteration
- Current default/local run: coverage `0.700`, supported accuracy `0.777`, hallucination rate `0.057`
- Reference weaker setting (`top_k=6`, `w_faiss=0.8`, `w_bm25=0.2`, `supported_min_overlap=5`) gave hallucination rate `0.327`
- Current tuned defaults in `app/config.py`: `TOP_K=3`, `FAISS_WEIGHT=0.2`, `BM25_WEIGHT=0.8`, `VERIFIER_SUPPORTED_MIN_OVERLAP=6`, `VERIFIER_UNCERTAIN_MIN_OVERLAP=2`

Generate detailed summaries locally with:

```bash
python3 scripts/tune_retrieval.py
python3 scripts/generate_eval_report.py
```

Then inspect `results/tuning_summary.json` and `results/eval_summary.json`.

## File Notes
- `scripts/build_eval_set.py` is an optional helper that duplicates cases into a larger JSONL file; it is not used by the default eval pipeline.

## Reproducibility Notes
- The default dataset in `data/eval_cases.jsonl` is synthetic for smoke testing and tuning.
- For credible metrics, keep the dataset and summary JSON in the repo (or publish them) so the numbers can be verified.

## Roadmap
- Plug in a real corpus aligned to the eval dataset and re-run the metrics.
- Expand evaluation to 500+ real-world cases with public datasets or internal benchmarks.
- Add stronger verification logic (claim parsing, contradiction detection, and citation scoring).

## License
MIT
