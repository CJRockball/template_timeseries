"""Non-authoritative live progress; never substitutes for Package 07 events."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

STATE = {
    "stage": "starting",
    "nodeid": None,
    "collected": 0,
    "completed": 0,
    "failed": 0,
    "skipped": 0,
}


def save():
    name = os.environ.get("ML_TEMPLATE_NN_PROGRESS")
    if not name:
        return
    path = Path(name)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps({**STATE, "updated_at": datetime.now(UTC).isoformat()}))
    temp.replace(path)


def pytest_sessionstart(session):
    STATE["stage"] = "collecting"
    save()


def pytest_collection_finish(session):
    STATE.update(stage="collected", collected=len(session.items))
    save()


def pytest_runtest_logstart(nodeid, location):
    STATE.update(stage="setup", nodeid=nodeid)
    save()


def pytest_runtest_logreport(report):
    STATE.update(stage=report.when, nodeid=report.nodeid)
    if report.when == "call" or (report.when == "setup" and (report.failed or report.skipped)):
        STATE["completed"] += 1
    if report.failed:
        STATE["failed"] += 1
    if report.skipped:
        STATE["skipped"] += 1
    save()


def pytest_sessionfinish(session, exitstatus):
    STATE.update(stage="finished", exitstatus=int(exitstatus))
    save()
