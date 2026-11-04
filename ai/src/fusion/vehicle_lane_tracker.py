"""Lightweight motion/IoU association and per-vehicle lane/distance state.

No detector calls. Ambiguous associations start new IDs rather than transferring
confirmed membership or range history. IDs are never reused within an instance.
"""
from dataclasses import dataclass, field
import math
import numpy as np
from .distance_smoother import DistanceSmoother
from .ego_lane_classifier import EgoLaneClassifier


@dataclass
class VehicleTrack:
    track_id: int
    bbox: np.ndarray
    class_name: str
    last_seen: float
    distance: DistanceSmoother
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4))
    lane_status: str = 'unknown'
    inside_count: int = 0
    outside_count: int = 0
    evidence_start: float = 0.
    previous_observation: str = 'unknown'
    last_supported: float = None
    last_frame: int = 0

    def clear_evidence(self):
        self.lane_status = 'unknown'
        self.inside_count = self.outside_count = 0
        self.last_supported = None
        self.previous_observation = 'unknown'


class VehicleLaneTracker:
    def __init__(self, inside_confirm_frames=3, outside_confirm_frames=6,
                 max_uncertain_hold_seconds=.3, track_timeout_seconds=.5,
                 distance_median_window=3, distance_ema_alpha=.35,
                 distance_max_age_seconds=.5, fps=30., min_iou=.25):
        if (not isinstance(inside_confirm_frames,int) or inside_confirm_frames < 1 or
                not isinstance(outside_confirm_frames,int) or outside_confirm_frames < 1 or
                not math.isfinite(max_uncertain_hold_seconds) or max_uncertain_hold_seconds < 0 or
                not math.isfinite(track_timeout_seconds) or track_timeout_seconds <= 0 or
                not math.isfinite(fps) or fps <= 0 or not 0 < min_iou < 1):
            raise ValueError('Invalid vehicle tracker configuration')
        self.inside_frames = inside_confirm_frames
        self.outside_frames = outside_confirm_frames
        self.hold_seconds = max_uncertain_hold_seconds
        self.timeout = track_timeout_seconds
        self.fps = fps
        self.min_iou = min_iou
        self.distance_options = (distance_median_window,distance_ema_alpha,distance_max_age_seconds)
        DistanceSmoother(*self.distance_options)  # Validate before model loading.
        self.next_id = 1
        self.reset()

    def reset(self):
        self.tracks = {}
        self.timestamp = None
        self.frame_index = 0
        self.expired_tracks = 0
        self.ambiguous_matches = 0

    def associate(self, detections, timestamp):
        if not math.isfinite(timestamp):
            raise ValueError('Vehicle timestamp must be finite')
        if self.timestamp is not None and (timestamp < self.timestamp or timestamp-self.timestamp > self.timeout):
            self.reset()
        self.timestamp = timestamp
        self.frame_index += 1
        for key,track in list(self.tracks.items()):
            if timestamp-track.last_seen > self.timeout+1e-9:
                del self.tracks[key]
                self.expired_tracks += 1
        boxes = []
        for d in detections:
            try:
                b = np.asarray(d.get('bbox'),dtype=float)
                valid = b.shape == (4,) and np.isfinite(b).all() and b[2]>b[0] and b[3]>b[1]
            except (TypeError,ValueError):
                valid = False
            boxes.append(b if valid else None)
        pairs = []
        for i,(d,box) in enumerate(zip(detections,boxes)):
            if box is None:
                continue
            for key,track in self.tracks.items():
                if d['class_name'] != track.class_name:
                    continue
                dt = timestamp-track.last_seen
                predicted = track.bbox+track.velocity*min(dt,.2)
                old_size = track.bbox[2:]-track.bbox[:2]
                ratio = (box[2:]-box[:2])/old_size
                if np.any(ratio < .5) or np.any(ratio > 2):
                    continue
                score = EgoLaneClassifier.iou(box,predicted)
                if score >= self.min_iou:
                    pairs.append((score,i,key))
        # Active tracks get first refusal; inactive hypotheses must not compete
        # with a continuously observed vehicle and create repeated ID churn.
        active_candidates = {i for _,i,key in pairs
                             if self.tracks[key].last_frame == self.frame_index-1}
        pairs = [(score,i,key) for score,i,key in pairs
                 if self.tracks[key].last_frame == self.frame_index-1 or i not in active_candidates]
        # Reject a close tie for either detection or track (e.g. overlapping cars).
        ambiguous_d,ambiguous_t = set(),set()
        for axis,blocked in ((1,ambiguous_d),(2,ambiguous_t)):
            groups = {}
            for pair in pairs:
                groups.setdefault(pair[axis],[]).append(pair[0])
            for key,scores in groups.items():
                scores.sort(reverse=True)
                if len(scores)>1 and scores[0]-scores[1] < .02:
                    blocked.add(key)
        ambiguous_d.update(i for _,i,key in pairs if key in ambiguous_t)
        self.ambiguous_matches += len(ambiguous_d)
        assignments,used = {},set()
        for score,i,key in sorted(pairs,reverse=True):
            if i in assignments or key in used or i in ambiguous_d or key in ambiguous_t:
                continue
            assignments[i] = key
            used.add(key)
        # Retire ambiguous hypotheses rather than allowing their old state to
        # compete again with replacement IDs on the next frame.
        retired = {key for _,i,key in pairs if i in ambiguous_d or key in ambiguous_t}-used
        for key in retired:
            del self.tracks[key]
        results = []
        for i,(d,box) in enumerate(zip(detections,boxes)):
            item = dict(d,track_id=None,tracking_reason='invalid_box')
            if box is None:
                results.append(item)
                continue
            if i in assignments:
                track = self.tracks[assignments[i]]
                dt = timestamp-track.last_seen
                lost = track.last_frame != self.frame_index-1
                if lost:
                    track.clear_evidence()
                    track.distance.reset()
                    track.velocity[:] = 0
                elif dt>0:
                    track.velocity = .5*track.velocity+.5*(box-track.bbox)/dt
                track.bbox = box.copy()
                track.last_seen = timestamp
                track.last_frame = self.frame_index
                reason = 'reacquired_reset' if lost else 'matched'
            else:
                track = VehicleTrack(self.next_id,box.copy(),d['class_name'],timestamp,
                                     DistanceSmoother(*self.distance_options),last_frame=self.frame_index)
                self.next_id += 1
                self.tracks[track.track_id] = track
                reason = 'ambiguous_new_id' if i in ambiguous_d else 'new_id'
            item.update(track_id=track.track_id,tracking_reason=reason)
            results.append(item)
        return results

    def classify(self, item, observation, timestamp):
        track = self.tracks.get(item.get('track_id'))
        result = dict(item,**observation,distance=None,raw_distance=None,
                      distance_fresh=False,distance_for_alerts=False)
        if track is None:
            result.update(lane_status='unknown',distance_eligible=False)
            return result
        obs = observation['lane_observation']
        hard_invalid = bool(item.get('crop_clipped') or observation.get('hard_invalid'))
        if hard_invalid:
            track.clear_evidence()
            track.distance.reset()
        else:
            if observation.get('implausible_association') and track.lane_status == 'inside':
                track.clear_evidence()
                track.distance.reset()
            if obs != track.previous_observation:
                track.inside_count = track.outside_count = 0
                track.evidence_start = timestamp
            track.inside_count = track.inside_count+1 if obs == 'inside' else 0
            track.outside_count = track.outside_count+1 if obs == 'outside' else 0
            required = self.inside_frames if obs == 'inside' else self.outside_frames
            count = track.inside_count if obs == 'inside' else track.outside_count
            # At >=30 FPS, maintain at least the confirmation duration at 30 FPS.
            # Lower FPS still requires the requested number of observations.
            confirmed = (obs in ('inside','outside') and count >= required and
                         timestamp-track.evidence_start+1e-9 >= (required-1)/30.)
            if confirmed and track.lane_status != obs:
                track.lane_status = obs
                track.last_supported = timestamp
                track.inside_count = track.outside_count = 0
            if obs == track.lane_status and obs in ('inside','outside'):
                track.last_supported = timestamp
            if track.last_supported is None or timestamp-track.last_supported > self.hold_seconds+1e-9:
                track.lane_status = 'unknown'
            track.previous_observation = obs
        # A green box may briefly retain membership, but only current inside
        # evidence permits a new measurement. No held range is displayed.
        eligible = track.lane_status == 'inside' and obs == 'inside' and not hard_invalid
        result.update(lane_status=track.lane_status,distance_eligible=eligible,
                      inside_count=track.inside_count,outside_count=track.outside_count,
                      classification_held=track.lane_status != 'unknown' and obs != track.lane_status)
        return result

    def smooth_distance(self, item, raw, timestamp):
        track = self.tracks.get(item.get('track_id'))
        if track is None:
            return None
        return track.distance.update(raw,timestamp)
