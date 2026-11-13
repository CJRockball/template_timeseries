"""Pytest evidence plugin; loaded only by the acceptance runner."""

import json
import os
from pathlib import Path

STATE = {
    "collected": {},
    "passed": [],
    "skipped": [],
    "failed": [],
    "xfail": [],
    "deselected": [],
    "collection_errors": [],
}


def pytest_collection_finish(session):
    for item in session.items:
        name = item.nodeid.split("::", 1)[0].replace("\\", "/")
        STATE["collected"][name] = STATE["collected"].get(name, 0) + 1


def pytest_deselected(items):
    STATE["deselected"].extend(item.nodeid for item in items)


def pytest_collectreport(report):
    if report.failed:
        STATE["collection_errors"].append({"nodeid": report.nodeid, "reason": str(report.longrepr)})
    elif report.skipped:
        STATE["skipped"].append(
            {"nodeid": report.nodeid, "when": "collection", "reason": str(report.longrepr)}
        )


def pytest_runtest_logreport(report):
    if getattr(report, "wasxfail", None):
        STATE["xfail"].append(
            {"nodeid": report.nodeid, "when": report.when, "reason": str(report.wasxfail)}
        )
    if report.failed:
        STATE["failed"].append(
            {"nodeid": report.nodeid, "when": report.when, "reason": str(report.longrepr)}
        )
    elif report.skipped:
        STATE["skipped"].append(
            {"nodeid": report.nodeid, "when": report.when, "reason": str(report.longrepr)}
        )
    elif report.when == "call" and report.passed:
        STATE["passed"].append(report.nodeid)


def pytest_sessionfinish(session, exitstatus):
    output = os.environ.get("ML_TEMPLATE_NN_ACCEPTANCE_EVENTS")
    if not output:
        raise RuntimeError("Acceptance event destination missing")
    value = {**STATE, "exitstatus": int(exitstatus)}
    destination = Path(output)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(destination)
