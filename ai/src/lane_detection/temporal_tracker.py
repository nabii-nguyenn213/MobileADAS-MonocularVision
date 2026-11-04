"""Bounded per-side temporal tracking. Confidence is geometric quality, not probability."""
from dataclasses import dataclass
import math
import cv2
import numpy as np


@dataclass
class Track:
    parameters: np.ndarray
    target: np.ndarray
    last_detected: np.ndarray
    observation_valid: bool
    last_seen: float
    hits: int
    confidence: float


class TemporalLaneTracker:
    SIDES = ('left_ego', 'right_ego')

    def __init__(self, alpha=.25, max_hold_seconds=.5, max_jump=.045,
                 acquisition_hits=2, classification_max_age=0., classification_confidence=.65,
                 samples=30, camera_center=.5):
        if not (0 < alpha <= 1 and 0 <= max_hold_seconds <= 5 and max_jump > 0 and
                acquisition_hits >= 1 and 0 <= classification_max_age <= max_hold_seconds and
                0 <= classification_confidence <= 1 and samples >= 6 and 0 < camera_center < 1):
            raise ValueError('Invalid lane tracker configuration')
        self.camera_center = camera_center
        self.alpha = alpha
        self.max_hold_seconds = max_hold_seconds
        self.max_jump = max_jump
        self.acquisition_hits = acquisition_hits
        self.classification_max_age = classification_max_age
        self.classification_confidence = classification_confidence
        self.samples = samples
        self.reset()

    def reset(self):
        self.tracks = {}
        self.last_timestamp = None
        self.resolution = None
        self.thumbnail = None
        self.timeout_events = 0
        self.scene_resets = 0

    @staticmethod
    def fit(points, resolution):
        try:
            p = np.asarray(points, dtype=float).reshape(-1, 2) / resolution
        except (TypeError, ValueError):
            return None
        if len(p) < 6 or not np.isfinite(p).all() or np.any(p < 0) or np.any(p >= 1):
            return None
        low, high = p[:, 1].min(), p[:, 1].max()
        if high-low < .06:
            return None
        a,b = np.polyfit(p[:,1],p[:,0],1)
        residual = np.abs(p[:,0] - (a*p[:,1]+b))
        if abs(a) > 2.5 or np.quantile(residual,.9) > .012:
            return None
        return np.array([a,b,low,high])

    @staticmethod
    def distance(a,b):
        return max(abs((a[0]-b[0])*y + a[1]-b[1]) for y in (.65,.9))

    def update(self, observations, resolution, timestamp, frame=None, detection_ran=True):
        if not math.isfinite(timestamp):
            raise ValueError('Lane timestamp must be finite seconds')
        reset_reason = None
        if self.resolution is not None and self.resolution != tuple(resolution):
            reset_reason = 'resolution'
        if self.last_timestamp is not None and (timestamp < self.last_timestamp or
                timestamp-self.last_timestamp > max(self.max_hold_seconds, .5)):
            reset_reason = 'time_discontinuity'
        if frame is not None:
            thumb = cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(64,36)).astype(np.float32)/255
            if self.thumbnail is not None and float(np.mean(abs(thumb-self.thumbnail))) > .32:
                reset_reason = 'scene_change'
            self.thumbnail = thumb
        if reset_reason:
            self.tracks.clear()
            self.scene_resets += 1
        self.resolution = tuple(resolution)
        self.last_timestamp = timestamp
        meta = {}
        accepted = {}
        expired = []
        for side in self.SIDES:
            track = self.tracks.get(side)
            if track is not None and timestamp-track.last_seen > self.max_hold_seconds+1e-9:
                self.tracks.pop(side)
                self.timeout_events += 1
                expired.append(side)
                track = None
            candidate = self.fit(observations.get(side,[]),resolution) if detection_ran else None
            reason = 'missing' if detection_ran else 'inference_skipped'
            if candidate is not None:
                projected = candidate[0]*.98+candidate[1]
                if (side == 'left_ego' and projected >= self.camera_center or
                        side == 'right_ego' and projected <= self.camera_center):
                    candidate = None
                    reason = 'identity_rejected'
            if candidate is not None:
                # Identity gate uses the camera's current ego pair, then temporal association.
                if track is not None and self.distance(candidate,track.parameters) > self.max_jump:
                    candidate = None
                    reason = 'jump_rejected'
                else:
                    reason = 'accepted'
            accepted[side] = candidate
            meta[side] = dict(rejection=reason, timed_out=side in expired)
        # Never blend candidates into an invalid pair. Preserve independent sides
        # when only one candidate is present, and validate rendered pairs below.
        l,r = accepted.values()
        if l is not None and r is not None:
            low,high = max(l[2],r[2]),min(l[3],r[3])
            if low >= high or any((r[0]-l[0])*y+r[1]-l[1] <= .015 for y in (low,high)):
                for side in self.SIDES:
                    accepted[side] = None
                    meta[side]['rejection'] = 'invalid_pair'
        rendered, reliable = {}, {}
        for side in self.SIDES:
            candidate = accepted[side]
            track = self.tracks.get(side)
            fresh = candidate is not None
            if fresh:
                if track is None:
                    track = Track(candidate.copy(),candidate.copy(),candidate.copy(),True,timestamp,1,
                                  .75 if self.acquisition_hits == 1 else .6)
                else:
                    track.target = candidate.copy()
                    track.last_detected = candidate.copy()
                    track.observation_valid = True
                    # Support must not expand beyond current observations for classification.
                    track.last_seen = timestamp
                    track.hits += 1
                    track.confidence = min(.95,track.confidence+.15)
                self.tracks[side] = track
            if track is not None:
                if detection_ran and not fresh:
                    track.observation_valid = False
                # Advance toward a stable fitted target on every input frame.
                track.parameters = self.alpha*track.target+(1-self.alpha)*track.parameters
            rendered[side] = []
            reliable[side] = []
            if track is None:
                meta[side].update(state='missing',status='lost',age_seconds=None,confidence=0.,classification_reliable=False)
                continue
            age = max(0.,timestamp-track.last_seen)
            confidence = track.confidence * max(.2,1-(.2 if track.observation_valid else .8)*age/max(self.max_hold_seconds,1e-9))
            confirmed = track.hits >= self.acquisition_hits
            a,b,low,high = track.parameters
            points = [(round((a*y+b)*resolution[0]), min(resolution[1]-1,round(y*resolution[1])))
                      for y in np.linspace(low,high,self.samples) if 0 <= a*y+b < 1]
            if confirmed:
                rendered[side] = points
            can_classify = confirmed and track.observation_valid and age <= self.classification_max_age+1e-9 and confidence >= self.classification_confidence
            if can_classify:
                # EMA support is for display; current support is the safe evaluation domain.
                if track.last_detected is not None:
                    low,high = track.last_detected[2:]
                    reliable[side] = [(round((a*y+b)*resolution[0]),min(resolution[1]-1,round(y*resolution[1])))
                                      for y in np.linspace(low,high,self.samples) if 0 <= a*y+b < 1]
                else:
                    reliable[side] = points
            meta[side].update(state=('tentative' if not confirmed else 'observed' if fresh else 'held'),
                              status='detected' if fresh else 'tracked',
                              age_seconds=age,confidence=confidence,classification_reliable=bool(reliable[side]))
        # Mixed held/fresh estimates may cross even if raw current candidates did not.
        if all(rendered.values()):
            l,r = [self.tracks[s].parameters for s in self.SIDES]
            lo,hi = max(l[2],r[2]),min(l[3],r[3])
            if lo >= hi or any((r[0]-l[0])*y+r[1]-l[1] <= .015 for y in (lo,hi)):
                rendered = {s:[] for s in self.SIDES}
                reliable = {s:[] for s in self.SIDES}
                for info in meta.values():
                    info.update(classification_reliable=False,rejection='tracked_pair_invalid')
        return rendered,reliable,dict(sides=meta,reset_reason=reset_reason,
                                     timeout_events=self.timeout_events,scene_resets=self.scene_resets,
                                     detection_ran=detection_ran)
