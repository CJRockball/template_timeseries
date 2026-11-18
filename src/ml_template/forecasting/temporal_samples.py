"Materialise plan predicates and replay each historical sample's own origin."

import json
import math
import sys
from collections import Counter, OrderedDict
from contextlib import suppress

from .learned_artifacts import Publication, write_line


def number(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or (not math.isfinite(float(value)))
    ):
        raise ValueError("Finite numeric value required")
    return float(value)


def options(value):
    required = {"features", "max_rows_per_partition"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Sample options require features and max_rows_per_partition")
    features = value["features"]
    if (
        not isinstance(features, list)
        or not features
        or any(not isinstance(x, str) for x in features)
        or (len(set(features)) != len(features))
    ):
        raise ValueError("Explicit ordered unique feature names required")
    cap = value["max_rows_per_partition"]
    if type(cap) is not int or cap < 1:
        raise ValueError("Positive partition cap required")
    return value


def vector(row, features):
    if not set(features) <= set(row["features"]):
        raise ValueError("Unknown selected history feature")
    missing = [f for f in features if row["features"][f] is None]
    return (None, missing) if missing else ([number(row["features"][f]) for f in features], [])


class _FeatureVectors:
    """Private LRU for one materialization with fixed history/recipe/features.

    Stores compact immutable vectors, never full history diagnostic dictionaries.
    Budget accounts for retained Python objects, not allocator RSS or total process
    memory. History must not change during this materialization.
    """

    def __init__(self, history, recipe, features, budget_bytes=16 * 1024 * 1024):
        if type(budget_bytes) is not int or budget_bytes < 0:
            raise ValueError("Nonnegative integer cache budget required")
        self.history = history
        self.recipe = recipe
        self.features = tuple(features)
        self.budget = budget_bytes
        self.entries = OrderedDict()
        self.payload_bytes = 0
        self.requests = self.hits = self.misses = self.evictions = self.skipped = 0
        self.peak_entries = self.peak_accounted_bytes = 0

    @staticmethod
    def _size(value):
        size = sys.getsizeof(value)
        if isinstance(value, tuple):
            size += sum(_FeatureVectors._size(item) for item in value)
        return size

    def get(self, serialized_series, target_date):
        self.requests += 1
        key = (serialized_series, target_date)
        if key in self.entries:
            self.hits += 1
            frozen, cost = self.entries[key]
            self.entries.move_to_end(key)
        else:
            self.misses += 1
            x, missing = vector(
                self.history.for_sample(json.loads(serialized_series), target_date, self.recipe),
                self.features,
            )
            frozen = (None if x is None else tuple(x), tuple(missing))
            cost = (
                self._size(key) + self._size(frozen) + sys.getsizeof((frozen, 0)) + sys.getsizeof(0)
            )
            if self.budget and cost + sys.getsizeof(OrderedDict()) + 256 <= self.budget:
                # Evict before insert; 256 reserves node/table growth overhead.
                while self.entries and (
                    self.payload_bytes + cost + sys.getsizeof(self.entries) + 256 > self.budget
                ):
                    self._evict()
                self.entries[key] = (frozen, cost)
                self.payload_bytes += cost
                while self.entries and self.accounted_bytes > self.budget:
                    self._evict()
                self.peak_entries = max(self.peak_entries, len(self.entries))
                self.peak_accounted_bytes = max(self.peak_accounted_bytes, self.accounted_bytes)
            else:
                self.skipped += 1
        # Do not expose mutable cached state to output writers or other callers.
        return (None if frozen[0] is None else list(frozen[0]), list(frozen[1]))

    @property
    def accounted_bytes(self):
        return self.payload_bytes + sys.getsizeof(self.entries)

    def _evict(self):
        _, (_, cost) = self.entries.popitem(last=False)
        self.payload_bytes -= cost
        self.evictions += 1
        if not self.entries:
            self.entries.clear()

    def summary(self):
        return {
            "event": "temporal_feature_cache",
            "requests": self.requests,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "oversize_or_disabled": self.skipped,
            "budget_bytes": self.budget,
            "peak_entries": self.peak_entries,
            "peak_accounted_bytes": self.peak_accounted_bytes,
            "scope": "run-local retained-object accounting; not an RSS cap",
        }

    def close(self):
        summary = self.summary()
        self.entries.clear()
        self.payload_bytes = 0
        with suppress(OSError):
            print(json.dumps(summary, sort_keys=True), file=sys.stderr, flush=True)


def materialize(plan, requests, history, recipe, configuration, producer, output_root, database):
    """requests is a validated RequestIndex; history is the shared HistoryIndex."""
    configuration = options(configuration)
    features = configuration["features"]
    cap = configuration["max_rows_per_partition"]
    publication = Publication(output_root)
    counts = []
    cache = _FeatureVectors(history, recipe, features)
    try:
        conn = history.conn
        conn.execute("CREATE TEMP TABLE IF NOT EXISTS active_series(series TEXT PRIMARY KEY)")
        with (
            (publication.stage / "samples.jsonl").open("w", encoding="utf-8") as samples,
            (publication.stage / "requests.jsonl").open("w", encoding="utf-8") as forecasts,
            (publication.stage / "eligibility.jsonl").open("w", encoding="utf-8") as audit,
        ):
            for window in plan["identity"]["windows"]:
                origin = window["origin_date"]
                if (
                    window["refit_end"] >= plan["identity"]["holdout"]["start"]
                    or window["target_date"] >= plan["identity"]["holdout"]["start"]
                ):
                    raise ValueError("Holdout materialisation prohibited")
                conn.execute("DELETE FROM active_series")
                for (key,) in requests.conn.execute(
                    "SELECT DISTINCT series FROM requests WHERE origin=? ORDER BY series", (origin,)
                ):
                    conn.execute("INSERT INTO active_series VALUES (?)", (key,))
                phases = [
                    ("fit", window["fit_start"], window["fit_end"], window["fit_available_by"]),
                    (
                        "stopping",
                        window["stopping_start"],
                        window["stopping_end"],
                        window["issued_at"],
                    ),
                    ("refit", window["refit_start"], window["refit_end"], window["issued_at"]),
                ]
                for phase, start, end, available in phases:
                    base = " FROM history h JOIN active_series a USING(series) WHERE h.day BETWEEN ? AND ?"
                    n = conn.execute("SELECT COUNT(*)" + base, (start, end)).fetchone()[0]
                    if n > cap:
                        raise ValueError(
                            f"{origin}/{phase}: {n} candidates exceed explicit cap {cap}; no sampling performed"
                        )
                    eligible = Counter()
                    complete = Counter()
                    for key, date, identifier, target in conn.execute(
                        "SELECT h.series,h.day,h.id,h.y"
                        + base
                        + " AND h.available<=? ORDER BY h.series,h.day",
                        (start, end, available),
                    ):
                        eligible[key] += 1
                        x, missing = cache.get(key, date)
                        write_line(
                            audit,
                            {
                                "origin": origin,
                                "partition": phase,
                                "series": json.loads(key),
                                "target_date": date,
                                "complete": not missing,
                                "missing_features": missing,
                            },
                        )
                        if missing:
                            continue
                        complete[key] += 1
                        write_line(
                            samples,
                            {
                                "origin": origin,
                                "partition": phase,
                                "series": json.loads(key),
                                "target_date": date,
                                "observation_id": json.loads(identifier),
                                "x": x,
                                "y": number(target),
                            },
                        )
                    for (key,) in conn.execute("SELECT series FROM active_series ORDER BY series"):
                        minimum = (
                            plan["identity"]["policy"]["stopping_periods"]
                            if phase == "stopping"
                            else plan["identity"]["policy"]["min_train_observations"]
                        )
                        if complete[key] < minimum:
                            raise ValueError(
                                f"{origin}/{phase}/{key}: {complete[key]} feature-complete rows, minimum {minimum}; labels eligible {eligible[key]}"
                            )
                    counts.append(
                        {
                            "origin": origin,
                            "partition": phase,
                            "date_candidates": n,
                            "label_eligible": sum(eligible.values()),
                            "feature_complete": sum(complete.values()),
                        }
                    )
                supported = 0
                for emitted, (raw,) in enumerate(
                    requests.conn.execute(
                        "SELECT row FROM requests WHERE origin=? ORDER BY series,request_id",
                        (origin,),
                    ),
                    start=1,
                ):
                    request = json.loads(raw)
                    key = json.dumps(
                        request["series"],
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    x, missing = cache.get(key, request["target_date"])
                    write_line(
                        forecasts,
                        {
                            "request": request,
                            "x": x,
                            "missing_features": missing,
                        },
                    )
                    supported += x is not None

                    if emitted > cap:
                        raise ValueError("Forecast request count exceeds explicit cap")

                if not supported:
                    raise ValueError("No feature-complete requests at origin " + origin)
        cache.close()
        cache = None
        (publication.stage / "report.json").write_text(
            json.dumps(
                {
                    "counts": counts,
                    "requests": requests.count,
                    "missing_policy": "complete_selected_features_only",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        inputs = {
            "plan_id": plan["plan_id"],
            "plan_manifest_sha256": plan["manifest_sha256"],
            "silver": plan["identity"]["silver"],
            "recipe": recipe,
            "sample_options": configuration,
            "windows": plan["identity"]["windows"],
            "holdout": plan["identity"]["holdout"],
            "producer": producer,
        }
        return publication.publish("samples", inputs, database)
    finally:
        if cache is not None:
            cache.close()
        publication.close()
