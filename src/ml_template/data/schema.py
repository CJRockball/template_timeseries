"""Schema facts — structural constants the pipeline assumes."""

import enum


class Role(enum.StrEnum):
    IDENTIFIER = "identifier"
    MEASUREMENT = "measurement"
    OUTCOME = "outcome"
    UNKNOWN = "unknown"


class Status(enum.StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
