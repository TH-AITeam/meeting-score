# meeting-score: 開発・評価用 Makefile
#
# eval ハーネス (Issue #5) の実行と日常開発のショートカットを提供する。

PY            ?= python
PYTEST        ?= ../.venv/bin/pytest
VENV_PY       ?= ../.venv/bin/python
DATASET       ?= ../data/annotations/gold/v1
MEETINGS_DIR  ?=
MODEL         ?=
OUT           ?= ../reports/eval/$(shell date +%Y%m%d_%H%M%S).json
STABILITY_OUT ?= ../reports/eval/stability_$(shell date +%Y%m%d_%H%M%S).json
N             ?= 5
SAMPLE        ?= ../data/sample_meetings/sample_meeting_01.json

.PHONY: help eval eval-stability test test-eval lint

help:
	@echo "Targets:"
	@echo "  make eval [DATASET=...] [MEETINGS_DIR=...] [MODEL=...] [OUT=...]"
	@echo "      アノテーション済みデータセットでベースライン評価"
	@echo "  make eval-stability [SAMPLE=...] [N=5] [OUT=...]"
	@echo "      同一会議を N 回採点して分散を測る"
	@echo "  make test           backend のテストを全部走らせる"
	@echo "  make test-eval      eval ハーネスのテストだけ走らせる"

# Issue #5 §15 ベースライン評価
eval:
	cd backend && $(VENV_PY) -m evals.cli \
		$(if $(MODEL),--model $(MODEL),) \
		run \
		--dataset $(DATASET) \
		$(if $(MEETINGS_DIR),--meetings-dir $(MEETINGS_DIR),) \
		--out $(OUT)

# Issue #5 §15 安定性評価 (N=5)
eval-stability:
	cd backend && $(VENV_PY) -m evals.cli \
		$(if $(MODEL),--model $(MODEL),) \
		stability \
		--meeting $(SAMPLE) \
		--n $(N) \
		--out $(STABILITY_OUT)

test:
	cd backend && $(PYTEST) -q

test-eval:
	cd backend && $(PYTEST) tests/test_eval_metrics.py tests/test_eval_stability.py tests/test_eval_runner.py -v
