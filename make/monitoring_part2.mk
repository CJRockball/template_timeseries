.PHONY: monitoring-part2-tests score-pinned-check score-pinned
PINNED_PYTHON ?=
PINNED_SCORE_ARGS ?=
monitoring-part2-tests:
	$(RUN) -m pytest -q tests/monitoring/test_monitoring_part2_bundle.py tests/usage_dashboard/test_usage_app.py
score-pinned-check:
	@test -n "$(PINNED_PYTHON)" -a -n "$(RELEASE_DIR)" -a -n "$(SCORE_INPUT)" || (echo 'Set PINNED_PYTHON, RELEASE_DIR and SCORE_INPUT'; exit 2)
	$(RUN) -m $(PKG).monitoring.score_pinned --python "$(PINNED_PYTHON)" --database "$(MONITOR_DATABASE)" --release "$(RELEASE_DIR)" --input "$(SCORE_INPUT)" --check $(PINNED_SCORE_ARGS)
score-pinned:
	@test -n "$(PINNED_PYTHON)" -a -n "$(RELEASE_DIR)" -a -n "$(SCORE_INPUT)" || (echo 'Set PINNED_PYTHON, RELEASE_DIR and SCORE_INPUT'; exit 2)
	$(RUN) -m $(PKG).monitoring.score_pinned --python "$(PINNED_PYTHON)" --database "$(MONITOR_DATABASE)" --release "$(RELEASE_DIR)" --input "$(SCORE_INPUT)" --execute $(PINNED_SCORE_ARGS)
