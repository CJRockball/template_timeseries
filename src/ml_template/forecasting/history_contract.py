"Strict request alignment shared by generation and separate outcome scoring."

import hashlib
import json
import platform
import sqlite3
from pathlib import Path

from .history import positive, sample_issuance
from .policy import canonical, day, digest

REQUEST_KEYS = {
    "request_id",
    "plan_id",
    "purpose",
    "series",
    "origin_date",
    "issued_at",
    "target_date",
    "horizon",
    "observation_id",
}


def inputs(plan, configuration):
    return {
        "plan_id": plan["plan_id"],
        "plan_manifest_sha256": plan["manifest_sha256"],
        "silver": plan["identity"]["silver"],
        "recipe": configuration,
        "producer": producer(),
    }


class RequestIndex:
    def __init__(self, path, plan, requests):
        self.plan = plan
        self.temporal = plan["identity"]["silver"]["temporal_schema"]
        self.policy = plan["identity"]["policy"]
        self.holdout = day(plan["identity"]["holdout"]["start"]).isoformat()
        self.conn = sqlite3.connect(path)
        try:
            self.conn.execute("PRAGMA temp_store=FILE")
            self.conn.execute("PRAGMA cache_size=-32768")
            self.conn.execute(
                "CREATE TABLE requests (request_id TEXT PRIMARY KEY,series TEXT,date TEXT,origin TEXT,observation_id TEXT UNIQUE,row TEXT,UNIQUE(series,date))"
            )
            windows = {w["origin_date"]: w for w in plan["identity"]["windows"]}
            count = 0
            with self.conn:
                for request in requests:
                    self.check(request, windows)
                    self.conn.execute(
                        "INSERT INTO requests VALUES (?,?,?,?,?,?)",
                        (
                            request["request_id"],
                            canonical(request["series"]),
                            request["target_date"],
                            request["origin_date"],
                            canonical(request["observation_id"]),
                            canonical(request),
                        ),
                    )
                    count += 1
            if count != positive(plan["validation"]["requests"]):
                raise ValueError("Plan request count mismatch")
            self.count = count
        except BaseException:
            self.conn.close()
            raise

    def check(self, request, windows):
        if set(request) != REQUEST_KEYS:
            raise ValueError("Unexpected request fields")
        if (
            request["plan_id"] != self.plan["plan_id"]
            or request["purpose"] != "development"
            or type(request["horizon"]) is not int
            or (request["horizon"] != 1)
        ):
            raise ValueError("Only bound one-step development requests are allowed")
        date = day(request["target_date"]).isoformat()
        if date >= self.holdout:
            raise ValueError("Holdout request prohibited")
        origin, issued = sample_issuance(date, self.policy, self.temporal)
        window = windows.get(origin)
        if (
            request["origin_date"] != origin
            or request["issued_at"] != issued
            or (not window)
            or (window["issued_at"] != issued)
            or (window["target_date"] != date)
        ):
            raise ValueError("Request origin/target mismatch")
        if (
            not isinstance(request["series"], list)
            or len(request["series"]) != len(self.temporal["series"])
            or any(type(v) not in (str, int) for v in request["series"])
        ):
            raise ValueError("Invalid series identity")
        if type(request["observation_id"]) not in (str, int):
            raise ValueError("Invalid observation identity")
        value = {k: v for k, v in request.items() if k != "request_id"}
        if request["request_id"] != "request_" + digest(value):
            raise ValueError("Request ID integrity failure")

    def rows(self):
        for (value,) in self.conn.execute(
            "SELECT row FROM requests ORDER BY origin,series,request_id"
        ):
            yield json.loads(value)

    def close(self):
        self.conn.close()


def producer():
    directory = Path(__file__).parent
    names = [
        "history.py",
        "history_artifacts.py",
        "history_contract.py",
        "history_engine.py",
        "history_metrics.py",
        "history_io.py",
        "history_cli.py",
        "policy.py",
    ]
    return {
        "python": platform.python_version(),
        "sqlite": sqlite3.sqlite_version,
        "source_sha256": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in names
        },
    }
