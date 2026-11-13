"""Attempt tracking around the existing scoring path. Preflight remains write-free."""

import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, model_validator

from ml_template.db import scoring_attempt_registry as registry

logger = logging.getLogger(__name__)


class BatchMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal[1] = 1
    source_system: str | None = None
    business_batch_id: str | None = None
    observation_start: datetime | None = None
    observation_end: datetime | None = None

    @model_validator(mode="after")
    def validate_dates(self):
        for value in (self.observation_start, self.observation_end):
            if value is not None and value.tzinfo is None:
                raise ValueError("observation times require an explicit timezone")
        if (
            self.observation_start
            and self.observation_end
            and self.observation_start > self.observation_end
        ):
            raise ValueError("observation_start exceeds observation_end")
        return self


def parse_metadata(value):
    if isinstance(value, (str, Path)):
        from ml_template.monitoring.artifacts import read_json

        value = read_json(value)
    return BatchMetadata.model_validate({} if value is None else value).model_dump(mode="json")


def tracked_score(
    function, engine, release_dir, input_path, *, batch_metadata=None, check_only=False, **kwargs
):
    if check_only:
        return function(
            engine,
            release_dir,
            input_path,
            check_only=True,
            batch_metadata=parse_metadata(batch_metadata),
            **kwargs,
        )
    identity = f"attempt_{datetime.now(UTC):%Y%m%d_%H%M%S}_{uuid4().hex[:12]}"
    started = time.perf_counter()
    registry.start(engine, identity, release_dir, Path(input_path).name)
    last_time, last_stage = time.perf_counter(), "metadata_validation"
    timings = {}

    def transition(stage, batch_id=None):
        nonlocal last_time, last_stage
        current = time.perf_counter()
        timings[last_stage] = timings.get(last_stage, 0.0) + current - last_time
        registry.stage(engine, identity, stage, batch_id)
        last_stage, last_time = stage, current

    def elapsed():
        current = time.perf_counter()
        values = dict(timings)
        values[last_stage] = values.get(last_stage, 0.0) + current - last_time
        return current - started, values

    try:
        parsed = parse_metadata(batch_metadata)
        registry.metadata(engine, identity, parsed)
        result = function(
            engine,
            release_dir,
            input_path,
            batch_metadata=parsed,
            stage_hook=transition,
            check_only=False,
            **kwargs,
        )
    except Exception as exc:
        duration, values = elapsed()
        try:
            registry.finish(engine, identity, duration, values, error=exc)
        except Exception:
            logger.exception(
                "Could not finish attempt %s; preserving original scoring exception", identity
            )
        raise
    duration, values = elapsed()
    registry.finish(engine, identity, duration, values, batch_id=result["scoring_batch_id"])
    return {**result, "attempt_id": identity}
