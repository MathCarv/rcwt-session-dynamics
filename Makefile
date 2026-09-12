.PHONY: agent-demo agent-evaluate agent-report agent-verify online-demo online-verify online-v4-demo online-v4-verify online-v4-verify-development online-v4-verify-confirmation online-v4-verify-interrupted replication-verify replication-verify-if-present replication-demo analyze-window bootstrap-test-tokenizer check-python fit-curves rebuild-intervals rescore-intact session-generate session-run session-score session-plot test verify verify-session

PYTHON ?= python3
SESSION_RESULTS ?= results/session_v1
AGENT_RESULTS ?= results/agent_v2
AGENT_POLICY ?= summary
AGENT_EPISODE ?= 0
ONLINE_RESULTS ?= results/agent_v3_development/attempt_05
ONLINE_DEVELOPMENT ?= results/agent_v3_development
ONLINE_V4_RESULTS ?= results/agent_v4
ONLINE_V4_DEVELOPMENT ?= results/agent_v4_development
ONLINE_V4_INTERRUPTED ?= results/agent_v4_interrupted
REPLICATION_RESULTS ?= results/agent_v4_replication
export PYTHONUTF8 := 1
export PYTHONIOENCODING := utf-8

check-python:
	$(PYTHON) -m compileall -q src tools

bootstrap-test-tokenizer:
	$(PYTHON) -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

# Dependency/tokenizer setup is explicit; the suite itself blocks all sockets.
# The Python wrapper also creates the ignored .runs scratch directory needed
# by trusted local-archive fixtures on a fresh Linux or Windows checkout.
test:
	$(PYTHON) -B tools/run_tests_offline.py

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

# Local inference only. Use a NEW directory for each experiment; existing runs
# are never silently overwritten. Requires the pinned llama.cpp server.
agent-evaluate:
	PYTHONPATH=src $(PYTHON) src/rcwt_agent_run.py --stage all \
		--runtime-receipt docs/rcwt_agent_runtime.json --output-dir $(AGENT_RESULTS)

# Offline action replay and evidence verification; does not call a model.
agent-verify:
	PYTHONPATH=src $(PYTHON) src/rcwt_agent_run.py --stage verify --output-dir $(AGENT_RESULTS)

# Rebuild the aggregate and narrative from verified saved calls, with no model.
agent-report:
	PYTHONPATH=src $(PYTHON) src/rcwt_agent_run.py --stage report --output-dir $(AGENT_RESULTS)

# Recorded demonstration, explicitly not a live inference run. The first
# manifest episode is the default; no best-case search is performed.
agent-demo:
	PYTHONPATH=src $(PYTHON) tools/replay_agent_episode.py --run-dir $(AGENT_RESULTS) \
		--episode-index $(AGENT_EPISODE) --policy $(AGENT_POLICY)

# Integrity PASS is separate from the failed development quality screen.
# All commands below are offline; they never restart or query a model.
online-verify:
	$(PYTHON) src/rcwt_online_v3.py --stage verify --output-dir $(ONLINE_RESULTS)
	$(PYTHON) tools/report_online_development.py --runs \
		$(ONLINE_DEVELOPMENT)/attempt_01 $(ONLINE_DEVELOPMENT)/attempt_02 \
		$(ONLINE_DEVELOPMENT)/attempt_03 $(ONLINE_DEVELOPMENT)/attempt_04 \
		$(ONLINE_DEVELOPMENT)/attempt_05 --output-dir $(ONLINE_DEVELOPMENT) --verify
	$(PYTHON) tools/inspect_online_v3.py --run-dir $(ONLINE_RESULTS) \
		--output-dir $(ONLINE_RESULTS)/diagnosis --verify

online-demo:
	$(PYTHON) tools/replay_online_v3.py --run-dir $(ONLINE_RESULTS) \
		--episode-index $(AGENT_EPISODE) --show-memory

# Released v4 development is always checked, including unsuccessful revisions.
# Each diagnosis uses its own pinned tools, so later source revisions do not
# silently change its interpretation. The index sums costs, not quality.
online-v4-verify-development:
	$(PYTHON) tools/report_online_v4_development.py --runs \
		"$(ONLINE_V4_DEVELOPMENT)/attempt_01" "$(ONLINE_V4_DEVELOPMENT)/attempt_02" \
		--output-dir "$(ONLINE_V4_DEVELOPMENT)" --verify
	$(PYTHON) tools/verify_online_v4_diagnosis_archive.py \
		--run-dir "$(ONLINE_V4_DEVELOPMENT)/attempt_01" \
		--diagnosis-dir "$(ONLINE_V4_DEVELOPMENT)/attempt_01/diagnosis_verified"
	$(PYTHON) tools/verify_online_v4_diagnosis_archive.py \
		--run-dir "$(ONLINE_V4_DEVELOPMENT)/attempt_02" \
		--diagnosis-dir "$(ONLINE_V4_DEVELOPMENT)/attempt_02/diagnosis"

# Strict target: an absent, incomplete or corrupt confirmation is an error.
# Report-integrity PASS is not an improvement or production-safety verdict.
online-v4-verify-confirmation:
	$(PYTHON) tools/verify_online_v4_report.py --run-dir "$(ONLINE_V4_RESULTS)"
	$(PYTHON) tools/inspect_online_v4.py --run-dir "$(ONLINE_V4_RESULTS)" \
		--output-dir "$(ONLINE_V4_RESULTS)/diagnosis" --verify

# This verifies only the recorded prefix, never a complete confirmation or gain.
online-v4-verify-interrupted:
	$(PYTHON) tools/verify_online_v4_partial.py --run-dir "$(ONLINE_V4_INTERRUPTED)"

# Publication may precede confirmation. Skip ONLY a wholly absent path;
# existing files/directories and even dangling links must reach strict checks.
# This does not audit unpublished .runs/ inventory or start any inference.
online-v4-verify: online-v4-verify-development
	@if [ -e "$(ONLINE_V4_INTERRUPTED)" ] || [ -L "$(ONLINE_V4_INTERRUPTED)" ]; then \
		$(MAKE) online-v4-verify-interrupted; \
	fi
	@if [ -e "$(ONLINE_V4_RESULTS)" ] || [ -L "$(ONLINE_V4_RESULTS)" ]; then \
		$(MAKE) online-v4-verify-confirmation; \
	else \
		printf '%s\n' 'No published v4 confirmation: development checked; confirmation not evaluated.'; \
	fi

online-v4-demo:
	$(PYTHON) tools/replay_online_v4.py --run-dir "$(ONLINE_V4_RESULTS)" \
		--episode-index $(AGENT_EPISODE)

# R1 is a separate authorized experiment, not a replacement for interrupted v4.
# These commands verify/replay saved evidence only. No server or inference.
replication-verify:
	$(PYTHON) tools/verify_v4_replication_report.py --run-dir "$(REPLICATION_RESULTS)"

replication-verify-if-present:
	@if [ -e "$(REPLICATION_RESULTS)" ] || [ -L "$(REPLICATION_RESULTS)" ]; then \
		$(MAKE) replication-verify; \
	else \
		printf '%s\n' 'No complete R1 archive available: R1 confirmation not evaluated.'; \
	fi

replication-demo:
	$(PYTHON) tools/replay_v4_replication.py --run-dir "$(REPLICATION_RESULTS)"
