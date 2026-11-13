MONITOR_DATABASE ?= data/project.db
MONITOR_OUTPUT ?= artifacts/monitoring
MONITOR_CONFIG ?= configs/monitoring/monitor.example.yaml
REFERENCE_CONFIG ?= configs/monitoring/reference.example.yaml
LABEL_CONFIG ?= configs/monitoring/labels.example.yaml

.PHONY: monitor-check monitor reference-check reference labels-check labels monitoring-tests

monitor-check:
	$(RUN) -m $(PKG).monitoring.monitor_batch --operation monitor --config "$(MONITOR_CONFIG)" --database "$(MONITOR_DATABASE)" --output-root "$(MONITOR_OUTPUT)" --check

monitor:
	$(RUN) -m $(PKG).monitoring.monitor_batch --operation monitor --config "$(MONITOR_CONFIG)" --database "$(MONITOR_DATABASE)" --output-root "$(MONITOR_OUTPUT)" --execute

reference-check:
	$(RUN) -m $(PKG).monitoring.monitor_batch --operation reference --config "$(REFERENCE_CONFIG)" --database "$(MONITOR_DATABASE)" --output-root "$(MONITOR_OUTPUT)" --check

reference:
	$(RUN) -m $(PKG).monitoring.monitor_batch --operation reference --config "$(REFERENCE_CONFIG)" --database "$(MONITOR_DATABASE)" --output-root "$(MONITOR_OUTPUT)" --execute

labels-check:
	$(RUN) -m $(PKG).monitoring.monitor_batch --operation labels --config "$(LABEL_CONFIG)" --database "$(MONITOR_DATABASE)" --output-root "$(MONITOR_OUTPUT)" --check

labels:
	$(RUN) -m $(PKG).monitoring.monitor_batch --operation labels --config "$(LABEL_CONFIG)" --database "$(MONITOR_DATABASE)" --output-root "$(MONITOR_OUTPUT)" --execute

monitoring-tests:
	$(RUN) -m pytest -q tests/monitoring
