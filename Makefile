.PHONY: analyze-window check-python fit-curves rebuild-intervals rescore-intact test verify

PYTHON ?= python3
PY_FILES := $(shell find src -name '*.py' | sort)

check-python:
	$(PYTHON) -m py_compile $(PY_FILES)

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

verify: check-python test rescore-intact rebuild-intervals fit-curves analyze-window
