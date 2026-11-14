"Daily one-step development policy and explicit availability boundaries."

import hashlib
import json
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

FORMAT = "temporal-origin-plan-v1"


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def day(value):
    if isinstance(value, datetime):
        raise ValueError("Expected business date, not timestamp")
    if isinstance(value, date):
        return value
    result = date.fromisoformat(value)
    if result.isoformat() != value:
        raise ValueError("Expected canonical YYYY-MM-DD")
    return result


def utc(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Availability must be an aware UTC timestamp")
    if value.utcoffset() != timedelta(0):
        raise ValueError("Availability must be UTC")
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def boundary(value, zone, end):
    value = day(value)
    local = datetime.combine(value + timedelta(days=1) if end else value, time.min, ZoneInfo(zone))
    alternate = local.replace(fold=1)
    if local.utcoffset() != alternate.utcoffset() or local.astimezone(UTC).astimezone(
        ZoneInfo(zone)
    ).replace(tzinfo=None) != local.replace(tzinfo=None):
        raise ValueError("Ambiguous/nonexistent business-day boundary")
    instant = local.astimezone(UTC)
    if end:
        instant -= timedelta(microseconds=1)
    return utc(instant)


def validate_policy(value):
    value = json.loads(canonical(value))
    required = {
        "development_origins",
        "train_start",
        "stopping_periods",
        "min_train_observations",
        "holdout_periods",
        "business_timezone",
    }
    optional = {"horizon", "window", "series_selection"}
    if set(value) - (required | optional) or not required <= set(value):
        raise ValueError("Unknown/missing origin policy field")
    value.setdefault("horizon", 1)
    value.setdefault("window", "expanding")
    value.setdefault("series_selection", None)
    if type(value["horizon"]) is not int or value["horizon"] != 1 or value["window"] != "expanding":
        raise ValueError("v1 supports one-step development and expanding history only")
    for key in ["stopping_periods", "min_train_observations", "holdout_periods"]:
        if type(value[key]) is not int or value[key] < 1:
            raise ValueError(f"{key} must be positive integer")
    origins = value["development_origins"]
    if not isinstance(origins, list) or not origins or any(not isinstance(x, str) for x in origins):
        raise ValueError("Explicit nonempty origin-date list required")
    dates = [day(x).isoformat() for x in origins]
    if dates != sorted(set(dates)):
        raise ValueError("Origins must be unique and chronological")
    value["train_start"] = day(value["train_start"]).isoformat()
    ZoneInfo(value["business_timezone"])
    selected = value["series_selection"]
    if selected is not None:
        if (
            not isinstance(selected, list)
            or not selected
            or any(
                not isinstance(x, list)
                or not x
                or any(v is None or isinstance(v, (list, dict, bool)) for v in x)
                for x in selected
            )
        ):
            raise ValueError("series_selection must be a nonempty list of composite series lists")
        if len({canonical(x) for x in selected}) != len(selected):
            raise ValueError("Duplicate series selection")
        value["series_selection"] = sorted(selected, key=canonical)
    return value


def windows(origin, policy, temporal):
    issued = day(origin)
    end = temporal["origin_policy"] == "end_of_business_day"
    if temporal["origin_policy"] not in {"end_of_business_day", "start_of_business_day"}:
        raise ValueError("Unsupported Silver origin policy")
    delay = temporal["target_delay_periods"]
    if type(delay) is not int or delay < 0:
        raise ValueError("Invalid Silver target delay")
    cutoff = issued - timedelta(days=delay + (0 if end else 1))
    stopping_start = cutoff - timedelta(days=policy["stopping_periods"] - 1)
    fit_end = stopping_start - timedelta(days=delay + 1)
    fit_origin = stopping_start - timedelta(days=1 if end else 0)
    if fit_end < day(policy["train_start"]):
        raise ValueError("Insufficient calendar history before stopping window")
    return {
        "origin_date": issued.isoformat(),
        "issued_at": boundary(issued, policy["business_timezone"], end),
        "fit_start": policy["train_start"],
        "fit_end": fit_end.isoformat(),
        "fit_available_by": boundary(fit_origin, policy["business_timezone"], end),
        "stopping_start": stopping_start.isoformat(),
        "stopping_end": cutoff.isoformat(),
        "refit_start": policy["train_start"],
        "refit_end": cutoff.isoformat(),
        "target_date": (issued + timedelta(days=1 if end else 0)).isoformat(),
        "horizon": 1,
        "purpose": "development",
    }


def effective_availability(observation_date, actual, policy, temporal):
    available_day = day(observation_date) + timedelta(days=temporal["target_delay_periods"])
    floor = boundary(available_day, policy["business_timezone"], True)
    return max(floor, utc(actual)) if actual is not None else floor
