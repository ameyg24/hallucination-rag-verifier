.PHONY: install serve eval eval-local eval-api report

install:
	python3 -m pip install -r requirements.txt

serve:
	python3 -m uvicorn app.main:app --reload --port 8000

eval-local:
	python3 scripts/run_eval.py --local

eval: eval-local

eval-api:
	python3 scripts/run_eval.py

report:
	python3 scripts/generate_eval_report.py
