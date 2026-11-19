"Streaming error accumulators. Coverage is separate from the common comparison."

import math

from .history import finite


class Errors:
    def __init__(self):
        self.n = 0
        self.absolute = 0.0
        self.squared = 0.0

    def add(self, actual, predicted):
        error = finite(finite(predicted) - finite(actual))
        self.absolute = finite(math.fsum((self.absolute, abs(error))))
        self.squared = finite(math.fsum((self.squared, finite(error * error))))
        self.n += 1
        return error

    def result(self):
        return {
            "n": self.n,
            "mae": self.absolute / self.n if self.n else None,
            "rmse": math.sqrt(self.squared / self.n) if self.n else None,
        }
