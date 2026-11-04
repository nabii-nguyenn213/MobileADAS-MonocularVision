"""Per-track post-estimation filtering; no detector, calibration, or range formula."""
from collections import deque
import math
import statistics


class DistanceSmoother:
    def __init__(self, window=3, alpha=.35, max_age_seconds=.5):
        if (not isinstance(window, int) or window < 1 or not 0 < alpha <= 1
                or not math.isfinite(max_age_seconds) or max_age_seconds <= 0):
            raise ValueError('Invalid distance smoothing configuration')
        self.history = deque(maxlen=window)
        self.alpha = alpha
        self.max_age_seconds = max_age_seconds
        self.reset()

    def reset(self):
        self.history.clear()
        self.smoothed = None
        self.last_valid_time = None
        self.last_raw = None
        self.pending_outlier = None
        self.rejection = None

    def update(self, value, timestamp):
        if not math.isfinite(timestamp):
            raise ValueError('Distance timestamp must be finite')
        if self.last_valid_time is not None and (timestamp < self.last_valid_time or
                timestamp-self.last_valid_time > self.max_age_seconds):
            self.reset()
        self.rejection = None
        if value is None:
            self.rejection = 'missing'
            return None
        try:
            value = float(value)
        except (ValueError, TypeError, OverflowError):
            self.rejection = 'invalid'
            return None
        if not math.isfinite(value) or value <= 0:
            self.rejection = 'invalid'
            return None
        # Reject order-of-magnitude numerical spikes, not ordinary rapid closure.
        # Repeated consistent discontinuities reacquire instead of locking forever.
        dt = max(0.,timestamp-self.last_valid_time) if self.last_valid_time is not None else 0.
        extreme = (self.last_raw is not None and max(value/self.last_raw,self.last_raw/value) > 4
                   and abs(value-self.last_raw) > max(3.,120*dt))
        if extreme:
            if self.pending_outlier is None or max(value/self.pending_outlier,self.pending_outlier/value) > 1.25:
                self.pending_outlier = value
                self.rejection = 'extreme_ratio'
                return None
            self.reset()
        self.pending_outlier = None
        self.history.append(value)
        median = statistics.median(self.history)
        self.smoothed = median if self.smoothed is None else self.alpha*median+(1-self.alpha)*self.smoothed
        self.last_raw = value
        self.last_valid_time = timestamp
        return float(self.smoothed)
