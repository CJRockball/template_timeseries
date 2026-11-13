# Makefile - project pipeline, one command end to end.
PKG    ?= ml_template
DS     ?= ds_v1
CONFIG ?= configs/bronze_silver_contract.yaml
EXP    ?= configs/experiment/xgb_baseline.yaml
SWEEP  ?= configs/sweep/xgb_grid.yaml
RUN    := uv run python
CALIBRATION_CONFIG ?= configs/calibration/test_958bec.yaml
EVALUATION_CONFIG ?= configs/evaluation/test_958bec.yaml
RELEASE_CONFIG ?= configs/release/example_958bec.yaml
RELEASE_DIR ?=
SCORE_INPUT ?=
SCORE_ARGS ?=

.PHONY: ingest eda silver split features train sweep evaluate pipeline clean rebuild check dashboard calibrate calibrate-check calibration-tests evaluate-check evaluation-tests release-check release score-check score scoring-tests
 
ingest:
	$(RUN) -m $(PKG).data.ingest --contract_file $(CONFIG) --dataset_version_id $(DS)
eda:
	$(RUN) -m $(PKG).scripts.eda_script --dataset-version-id $(DS)
silver:
	$(RUN) -m $(PKG).data.silver --dataset-version-id $(DS)
split:
	$(RUN) -m $(PKG).training.split --config $(EXP)
features:
	$(RUN) -m $(PKG).features.build_features --config $(EXP) \
	--blocks $(BLOCKS) $(if $(PARENT),--parent $(PARENT),) $(if $(COLUMNS),--columns $(COLUMNS),)  \
	$(if $(BLOCK_PARAMS),--block-params $(BLOCK_PARAMS),)
train:
	$(RUN) -m $(PKG).training.train_model --config $(EXP)
sweep:
	$(RUN) -m $(PKG).training.train_model --sweep $(SWEEP)
dashboard:
	$(RUN) -m $(PKG).dashboard.app
calibrate:
	$(RUN) -m $(PKG).training.calibrate --config $(CALIBRATION_CONFIG)
calibrate-check:
	$(RUN) -m $(PKG).training.calibrate --config $(CALIBRATION_CONFIG) --check
calibration-tests:
	$(RUN) -m pytest -q tests/test_probability_calibrator.py tests/test_calibration_*.py
evaluate-check:
	$(RUN) -m $(PKG).evaluation.evaluate_run --config $(EVALUATION_CONFIG) --check
evaluate:
	$(RUN) -m $(PKG).evaluation.evaluate_run --config $(EVALUATION_CONFIG) --execute
evaluation-tests:
	$(RUN) -m pytest -q tests/evaluation/test_final_evaluation_*.py tests/dashboard/test_final_evaluation_*.py

pipeline: ingest eda silver split train
rebuild: clean pipeline

clean:
	# Preserve datasets, project.db, model provenance, releases and scored batches.
	rm -rf .pytest_cache .ruff_cache .mypy_cache

check:
	uv run ruff check && uv run pytest -q

release-check:
	$(RUN) -m $(PKG).scoring.release_builder --config "$(RELEASE_CONFIG)" --check
release:
	$(RUN) -m $(PKG).scoring.release_builder --config "$(RELEASE_CONFIG)" --execute
score-check:
	@test -n "$(RELEASE_DIR)" -a -n "$(SCORE_INPUT)" || (echo 'Set RELEASE_DIR and SCORE_INPUT'; exit 2)
	$(RUN) -m $(PKG).scoring.batch --release "$(RELEASE_DIR)" --input "$(SCORE_INPUT)" --check $(SCORE_ARGS)
score:
	@test -n "$(RELEASE_DIR)" -a -n "$(SCORE_INPUT)" || (echo 'Set RELEASE_DIR and SCORE_INPUT'; exit 2)
	$(RUN) -m $(PKG).scoring.batch --release "$(RELEASE_DIR)" --input "$(SCORE_INPUT)" --execute $(SCORE_ARGS)
scoring-tests:
	$(RUN) -m pytest -q tests/scoring

# BEGIN explainability batch 2 (backend only)
EXPLANATION_CONFIG ?= configs/explainability/example.yaml
.PHONY: explain-check explain explanation-tests
explain-check:
	$(RUN) -m $(PKG).explainability.explain_batch --config "$(EXPLANATION_CONFIG)" --check
explain:
	$(RUN) -m $(PKG).explainability.explain_batch --config "$(EXPLANATION_CONFIG)" --execute
explanation-tests:
	$(RUN) -m pytest -q tests/explainability
# END explainability batch 2

# BEGIN runtime and separate usage presentation
RUNTIME_RELEASE ?=
RUNTIME_CHECKOUT ?=
RUNTIME_COMMIT ?=
USAGE_DATABASE ?= data/project.db
USAGE_PORT ?= 8051
EXPLANATION_ID ?=
REPORT_ROOT ?= artifacts/explanation_reports
.PHONY: runtime-check runtime-create explanation-report usage-dashboard presentation-tests
runtime-check:
	@test -n "$(RUNTIME_RELEASE)" || (echo 'Set RUNTIME_RELEASE'; exit 2)
	$(RUN) -m $(PKG).scripts.prepare_runtime --release "$(RUNTIME_RELEASE)" $(if $(RUNTIME_CHECKOUT),--checkout "$(RUNTIME_CHECKOUT)",) $(if $(RUNTIME_COMMIT),--commit "$(RUNTIME_COMMIT)",) --check
runtime-create:
	@test -n "$(RUNTIME_RELEASE)" || (echo 'Set RUNTIME_RELEASE'; exit 2)
	$(RUN) -m $(PKG).scripts.prepare_runtime --release "$(RUNTIME_RELEASE)" $(if $(RUNTIME_CHECKOUT),--checkout "$(RUNTIME_CHECKOUT)",) $(if $(RUNTIME_COMMIT),--commit "$(RUNTIME_COMMIT)",) --create
explanation-report:
	@test -n "$(EXPLANATION_ID)" || (echo 'Set EXPLANATION_ID'; exit 2)
	$(RUN) -m $(PKG).explainability.export_report --database "$(USAGE_DATABASE)" --explanation-id "$(EXPLANATION_ID)" --output-root "$(REPORT_ROOT)"
usage-dashboard:
	$(RUN) -m $(PKG).usage_dashboard.app --database "$(USAGE_DATABASE)" --port "$(USAGE_PORT)"
presentation-tests:
	$(RUN) -m pytest -q tests/runtime tests/presentation tests/usage_dashboard
# END runtime and separate usage presentation

# Operational monitoring artifact consumers
include make/monitoring.mk

# Operational monitoring part two
include make/monitoring_part2.mk
