"""Streaming metric means without retaining one dictionary per recorded frame."""
import math
import numpy as np


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


class BatchAccumulator:
    """Vectorized two-sum expansions; only completed/prefix results become dicts."""
    def __init__(self, size):
        self.size = size
        self.names = None
        self.count = np.zeros(size,dtype=np.int64)

    def reset(self, ids):
        self.count[ids] = 0
        if self.names is not None:
            self.lengths[ids] = 0
            self.partials[ids] = 0

    def add(self, ids, values):
        if self.names is None:
            self.names = tuple(values)
            self.lengths = np.zeros((self.size,len(self.names)),dtype=np.int64)
            self.partials = np.zeros((self.size,len(self.names),16))
            self.last = np.zeros((self.size,len(self.names)))
        if tuple(values)!=self.names:
            raise ValueError('metric keys changed during a clip')
        x = np.column_stack(tuple(values.values()))
        if not np.isfinite(x).all():
            raise ValueError('non-finite metric')
        self.last[ids] = x
        lengths = self.lengths[ids]
        positions = np.zeros_like(lengths)
        row,col = np.indices(x.shape)
        for index in range(int(lengths.max(initial=0))):
            active = index<lengths
            y = self.partials[ids,:,index].copy()
            swap = np.abs(x)<np.abs(y)
            a,b = np.where(swap,y,x),np.where(swap,x,y)
            high = a+b; low = b-(high-a)
            keep = active&(low!=0)
            self.partials[np.asarray(ids)[row[keep]],col[keep],positions[keep]] = low[keep]
            positions += keep
            x = np.where(active,high,x)
        if positions.max(initial=0)>=self.partials.shape[-1]:
            self.partials = np.pad(self.partials,((0,0),(0,0),(0,16)))
        self.partials[np.asarray(ids)[:,None],col,positions] = x
        self.lengths[ids] = positions+1
        self.count[ids] += 1

    def summary(self, slot):
        count = int(self.count[slot])
        means = {name:math.fsum(self.partials[slot,i,:self.lengths[slot,i]])/count
                 for i,name in enumerate(self.names or ())} if count else {}
        last = {name:float(self.last[slot,i]) for i,name in enumerate(self.names or ())} if count else None
        return count,means,last
