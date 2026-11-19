"Disk-backed as-of history; calendar lags never mean row offsets."

import math
import sqlite3
from datetime import timedelta

from .policy import boundary, canonical, day, effective_availability, utc

METHODS = ("last_available", "seasonal_naive_7")


def finite(value):
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("Expected numeric continuous value")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Nonfinite numeric value")
    return value


def positive(value):
    if type(value) is not int or value < 1:
        raise ValueError("Positive integer required")
    return value


def recipe(value=None):
    value = (
        {
            "lags": [1, 7, 14, 28],
            "rolling": [{"periods": 7, "min_periods": 7}, {"periods": 28, "min_periods": 28}],
            "methods": list(METHODS),
        }
        if value is None
        else value
    )
    if not isinstance(value, dict) or set(value) != {"lags", "rolling", "methods"}:
        raise ValueError("Recipe requires exactly lags, rolling and methods")
    lags = value["lags"]
    methods = value["methods"]
    if not isinstance(lags, list) or len(lags) != len(set(lags)):
        raise ValueError("Unique lag list required")
    for lag in lags:
        positive(lag)
    if (
        not isinstance(methods, list)
        or not methods
        or len(set(methods)) != len(methods)
        or (not set(methods) <= set(METHODS))
    ):
        raise ValueError("Unknown/duplicate baseline method")
    if "seasonal_naive_7" in methods and 7 not in lags:
        raise ValueError("Weekly baseline requires explicit calendar lag 7")
    rolling = value["rolling"]
    if not isinstance(rolling, list):
        raise ValueError("Rolling list required")
    periods = set()
    for window in rolling:
        if not isinstance(window, dict) or set(window) != {"periods", "min_periods"}:
            raise ValueError("Explicit rolling periods/min_periods required")
        positive(window["periods"])
        positive(window["min_periods"])
        if window["min_periods"] > window["periods"] or window["periods"] in periods:
            raise ValueError("Invalid/duplicate rolling window")
        periods.add(window["periods"])
    return {
        "lags": sorted(lags),
        "rolling": sorted(rolling, key=lambda x: x["periods"]),
        "methods": sorted(methods),
    }


def sample_issuance(target_date, policy, temporal):
    target = day(target_date)
    end = temporal["origin_policy"] == "end_of_business_day"
    if temporal["origin_policy"] not in {"end_of_business_day", "start_of_business_day"}:
        raise ValueError("Invalid origin convention")
    origin = target - timedelta(days=1 if end else 0)
    return (origin.isoformat(), boundary(origin, policy["business_timezone"], end))


class HistoryIndex:
    def __init__(self, path, policy, temporal, holdout_start, history_end):
        self.policy, self.temporal = (policy, temporal)
        if (
            type(temporal["target_delay_periods"]) is not int
            or temporal["target_delay_periods"] < 0
        ):
            raise ValueError("Nonnegative target delay required")
        self.holdout_start = day(holdout_start).isoformat()
        self.history_end = day(history_end).isoformat()
        if self.history_end >= self.holdout_start:
            raise ValueError("History overlaps holdout")
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA temp_store=FILE")
        self.conn.execute("PRAGMA cache_size=-32768")
        self.conn.execute(
            "CREATE TABLE history(series TEXT,day TEXT,id TEXT UNIQUE,y REAL NOT NULL,available TEXT NOT NULL,PRIMARY KEY(series,day)) WITHOUT ROWID"
        )

    def close(self):
        self.conn.close()

    def add_rows(self, records, batch_size=5000):
        positive(batch_size)
        t = self.temporal
        columns = {t["observation_id"], *t["series"], t["time"], t["target"]}
        if t.get("available_at"):
            columns.add(t["available_at"])
        batch = []
        count = 0
        with self.conn:
            for record in records:
                if set(record) != columns:
                    raise ValueError("Unexpected history projection")
                date = day(record[t["time"]]).isoformat()
                if date > self.history_end or date >= self.holdout_start:
                    raise ValueError("History source returned out-of-bound date")
                series = [record[c] for c in t["series"]]
                identifier = record[t["observation_id"]]
                if identifier is None or any(v is None for v in series):
                    raise ValueError("Null history identity")
                actual = record.get(t.get("available_at"))
                if t.get("available_at") and actual is None:
                    raise ValueError("Missing declared availability")
                available = effective_availability(date, actual, self.policy, t)
                batch.append(
                    (
                        canonical(series),
                        date,
                        canonical(identifier),
                        finite(record[t["target"]]),
                        available,
                    )
                )
                count += 1
                if len(batch) == batch_size:
                    self.conn.executemany("INSERT INTO history VALUES (?,?,?,?,?)", batch)
                    batch.clear()
            if batch:
                self.conn.executemany("INSERT INTO history VALUES (?,?,?,?,?)", batch)
        return count

    def for_sample(self, series, target_date, configuration=None):
        origin, issued = sample_issuance(target_date, self.policy, self.temporal)
        return self.features(series, target_date, origin, issued, configuration)

    def features(self, series, target_date, origin_date, issued_at, configuration=None):
        r = recipe(configuration)
        target = day(target_date)
        if target.isoformat() >= self.holdout_start:
            raise ValueError("Holdout history request prohibited")
        expected_origin, expected_issued = sample_issuance(target, self.policy, self.temporal)
        if origin_date != expected_origin or utc(issued_at) != expected_issued:
            raise ValueError("Feature issuance does not match sample's own origin")
        issued = expected_issued
        key = canonical(series)
        last = self.conn.execute(
            "SELECT day,y FROM history WHERE series=? AND day<? AND available<=? ORDER BY day DESC LIMIT 1",
            (key, target.isoformat(), issued),
        ).fetchone()
        count = self.conn.execute(
            "SELECT COUNT(*) FROM history WHERE series=? AND day<? AND available<=?",
            (key, target.isoformat(), issued),
        ).fetchone()[0]
        values = {"last_available": last[1] if last else None}
        reasons = {"last_available": None if last else "no_eligible_history"}
        for lag in r["lags"]:
            date = (target - timedelta(days=lag)).isoformat()
            row = self.conn.execute(
                "SELECT y,available FROM history WHERE series=? AND day=?", (key, date)
            ).fetchone()
            floor = effective_availability(date, None, self.policy, self.temporal)
            reason = (
                "unavailable_by_policy"
                if floor > issued
                else "missing_calendar_date"
                if row is None
                else "late_availability"
                if row[1] > issued
                else None
            )
            values[f"lag_{lag}"] = None if reason else row[0]
            reasons[f"lag_{lag}"] = reason
        end = day(origin_date) - timedelta(
            days=self.temporal["target_delay_periods"]
            + (0 if self.temporal["origin_policy"] == "end_of_business_day" else 1)
        )
        diagnostics = {
            "history_count": count,
            "last_available_date": last[0] if last else None,
            "last_available_age_days": (target - day(last[0])).days if last else None,
            "rolling_end": end.isoformat(),
            "rolling_counts": {},
        }
        for window in r["rolling"]:
            width = window["periods"]
            start = (end - timedelta(days=width - 1)).isoformat()
            args = (key, start, end.isoformat(), issued)
            n = self.conn.execute(
                "SELECT COUNT(*) FROM history WHERE series=? AND day BETWEEN ? AND ? AND available<=?",
                args,
            ).fetchone()[0]
            diagnostics["rolling_counts"][str(width)] = n
            name = f"mean_{width}"
            values[name] = None
            reasons[name] = "insufficient_calendar_window"
            if n >= window["min_periods"]:
                rows = self.conn.execute(
                    "SELECT y FROM history WHERE series=? AND day BETWEEN ? AND ? AND available<=? ORDER BY day",
                    args,
                )
                values[name] = finite(math.fsum(row[0] for row in rows) / n)
                reasons[name] = None
        return {
            "features": values,
            "unsupported_features": {k: v for k, v in reasons.items() if v},
            "diagnostics": diagnostics,
        }


def predictions(feature_row, configuration=None):
    r = recipe(configuration)
    for method in r["methods"]:
        name = "last_available" if method == "last_available" else "lag_7"
        value = feature_row["features"][name]
        yield {
            "method": method,
            "status": "unsupported" if value is None else "ok",
            "y_pred": value,
            "reason": feature_row["unsupported_features"].get(name),
        }
