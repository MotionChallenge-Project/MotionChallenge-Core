"""Streaming metric means without retaining one dictionary per recorded frame."""
import math


class MetricAccumulator:
    def __init__(self):
        self.count = 0
        self.partials = {}
        self.last = None

    def add(self, values):
        if self.count and values.keys() != self.partials.keys():
            raise ValueError('metric keys changed during a clip')
        for key, value in values.items():
            if not math.isfinite(value):
                raise ValueError('non-finite metric')
            partials = self.partials.setdefault(key, [])
            # Error-free two-sum expansion; fsum performs the final rounding.
            # Storage depends on floating-point exponent range, not frame count.
            x, i = float(value), 0
            for y in partials:
                if abs(x) < abs(y):
                    x, y = y, x
                high = x + y
                low = y - (high - x)
                if low:
                    partials[i] = low
                    i += 1
                x = high
            partials[i:] = [x]
        self.last = values.copy()
        self.count += 1

    def means(self):
        return {key: math.fsum(values)/self.count for key, values in self.partials.items()}
