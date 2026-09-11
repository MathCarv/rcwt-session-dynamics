.PHONY: analyze-window check-python fit-curves rebuild-intervals rescore-intact session-generate session-run session-score session-plot test verify verify-session

PYTHON ?= python3
SESSION_RESULTS ?= results/session_v1
export PYTHONUTF8 := 1
export PYTHONIOENCODING := utf-8

check-python:
	$(PYTHON) -m compileall -q src

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -p 'test_*.py'

rescore-intact:
	PYTHONPATH=src $(PYTHON) src/rescore_intact_ablation.py \
		--responses results/intact_ablation/rcwt_intact_ablation_responses.jsonl \
		--output-dir results/intact_ablation

rebuild-intervals:
	PYTHONPATH=src $(PYTHON) src/rebuild_call_level_intervals.py

fit-curves:
	PYTHONPATH=src $(PYTHON) src/rcwt_curve_fitting.py

analyze-window:
	PYTHONPATH=src $(PYTHON) src/analyze_new_results.py

session-generate:
	PYTHONPATH=src $(PYTHON) src/rcwt_session_generate.py \
		--seed 20260911 \
		--cases 64 \
		--turns 64 \
		--output-dir $(SESSION_RESULTS)

session-run:
	PYTHONPATH=src $(PYTHON) src/rcwt_session_run.py \
		--cases $(SESSION_RESULTS)/public_cases.jsonl \
		--manifest $(SESSION_RESULTS)/manifest.json \
		--treatments tail,state_latest,state_first \
		--checkpoints 8,16,32,64 \
		--budgets 256,512,1024 \
		--output $(SESSION_RESULTS)/contexts.jsonl

session-score:
	PYTHONPATH=src $(PYTHON) src/rcwt_session_score.py \
		--contexts $(SESSION_RESULTS)/contexts.jsonl \
		--cases $(SESSION_RESULTS)/public_cases.jsonl \
		--oracle $(SESSION_RESULTS)/oracle_cases.jsonl \
		--manifest $(SESSION_RESULTS)/manifest.json \
		--output $(SESSION_RESULTS)/aggregates.json

session-plot:
	PYTHONPATH=src $(PYTHON) src/plot_session_results.py \
		--aggregates $(SESSION_RESULTS)/aggregates.json \
		--output $(SESSION_RESULTS)/decision_readiness.svg

verify-session: session-generate session-run session-score session-plot

verify: check-python verify-session test rescore-intact rebuild-intervals fit-curves analyze-window
