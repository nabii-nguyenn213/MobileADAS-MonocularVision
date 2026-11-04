"""Conservative straight-lane membership using tightly bounded lane support."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Boundary:
    slope: float
    intercept: float
    low: float
    high: float

    def x(self, y):
        return self.slope*y + self.intercept


class EgoLaneClassifier:
    def __init__(self, margin=8., smooth=False, confirmation_frames=2, max_extrapolation=.015):
        if not np.isfinite(margin) or margin < 0 or confirmation_frames < 1 or not 0 <= max_extrapolation <= .03:
            raise ValueError('Invalid margin or confirmation count')
        self.max_extrapolation = max_extrapolation
        self.margin = margin
        self.smooth = smooth
        self.confirmation_frames = confirmation_frames
        self.reset()

    def reset(self):
        self.boundaries = None
        self.diagnostics = {'geometry_reason': 'not_prepared'}
        self.previous = []

    @staticmethod
    def fit(points, width, height):
        try:
            pts = np.asarray(points, dtype=float).reshape(-1,2)
        except (ValueError, TypeError):
            return None
        if len(pts) < 6 or not np.isfinite(pts).all():
            return None
        if not ((pts[:,0] >= 0).all() and (pts[:,0] < width).all() and
                (pts[:,1] >= 0).all() and (pts[:,1] < height).all()):
            return None
        # Normalize both coordinates for resolution-independent RANSAC residuals.
        p = pts / [width,height]
        rng = np.random.default_rng(17)
        best = None
        for _ in range(48):
            a,b = p[rng.choice(len(p),2,replace=False)]
            if abs(a[1]-b[1]) < .05:
                continue
            slope = (a[0]-b[0])/(a[1]-b[1])
            if abs(slope) > 2.5:
                continue
            residual = abs(p[:,0] - (slope*p[:,1] + a[0]-slope*a[1]))
            inside = residual < .008
            if best is None or inside.sum() > best.sum():
                best = inside
        if best is None or best.sum() < max(6, int(.7*len(p))):
            return None
        support = pts[best]
        if np.ptp(support[:,1]) < .10*height:
            return None
        slope, intercept = np.polyfit(support[:,1], support[:,0], 1)
        return Boundary(slope,intercept,float(support[:,1].min()),float(support[:,1].max()))

    def prepare(self, lanes, resolution):
        width,height = resolution
        left = self.fit(lanes.get('left_ego', []), width,height)
        right = self.fit(lanes.get('right_ego', []), width,height)
        self.boundaries = None
        self.diagnostics = dict(geometry_reason='missing_or_invalid_boundary')
        if left is None or right is None:
            return
        low,high = max(left.low,right.low), min(left.high,right.high)
        if high-low < .10*height:
            self.diagnostics['geometry_reason'] = 'insufficient_common_support'
            return
        low = max(0.,low-self.max_extrapolation*height)
        high = min(height-1.,high+self.max_extrapolation*height)
        widths = np.array([right.x(y)-left.x(y) for y in (low,high)])
        if np.any(widths <= max(2*self.margin,.015*width)) or np.any(widths > .95*width):
            self.diagnostics['geometry_reason'] = 'implausible_lane_width'
            return
        self.boundaries = (left,right,low,high)
        self.diagnostics = dict(geometry_reason='valid',supported_y=[low,high])

    def classify(self, bbox, resolution):
        if self.boundaries is None:
            return 'unknown'
        try:
            x1,y1,x2,y2 = map(float,bbox)
        except (TypeError,ValueError):
            return 'unknown'
        width,height = resolution
        # Partial boxes may have an unreliable ground-contact point.
        if not np.isfinite([x1,y1,x2,y2]).all() or not (0 < x1 < x2 < width-1 and 0 < y1 < y2 < height-1):
            return 'unknown'
        left,right,low,high = self.boundaries
        if not low <= y2 <= high:
            return 'unknown'
        x = (x1+x2)/2
        lx,rx = left.x(y2),right.x(y2)
        if abs(x-lx) <= self.margin or abs(x-rx) <= self.margin:
            return 'unknown'
        return 'inside' if lx+self.margin < x < rx-self.margin else 'outside'

    def observe(self, bbox, resolution, inside_margin=.10, outside_margin=.05, crop_clipped=False):
        """Current geometric evidence, separate from per-vehicle stable state."""
        result = dict(lane_observation='unknown',lane_u=None,
                      lane_reason=self.diagnostics.get('geometry_reason','unavailable'),
                      lane_geometry_valid=self.boundaries is not None)
        if crop_clipped:
            return dict(result,lane_reason='roi_clipped',hard_invalid=True)
        try:
            box = np.asarray(bbox,dtype=float)
            width,height = resolution
            if box.shape != (4,) or not np.isfinite(box).all():
                raise ValueError('invalid')
            x1,y1,x2,y2 = box
            if not (0 < x1 < x2 < width-1 and 0 < y1 < y2 < height-1):
                raise ValueError('partial')
        except (TypeError,ValueError):
            return dict(result,lane_reason='invalid_or_partial_box',hard_invalid=True)
        if self.boundaries is None:
            return result
        left,right,_,_ = self.boundaries
        # Use the observed support only: no added extrapolation for new evidence.
        if not max(left.low,right.low) <= y2 <= min(left.high,right.high):
            return dict(result,lane_reason='bottom_outside_supported_y')
        lx,rx = left.x(y2),right.x(y2)
        lane_width = rx-lx
        if not max(2*self.margin,.015*width) < lane_width <= .95*width:
            return dict(result,lane_reason='implausible_lane_width')
        u = float(((x1+x2)/2-lx)/lane_width)
        margin = max(inside_margin,self.margin/lane_width)
        outside = max(outside_margin,self.margin/lane_width)
        observation = ('inside' if margin <= u <= 1-margin else
                       'outside' if u < -outside or u > 1+outside else 'uncertain')
        return dict(result,lane_observation=observation,lane_u=u,
                    lane_reason='boundary_zone' if observation == 'uncertain' else 'valid',
                    implausible_association=u < -.5 or u > 1.5)

    @staticmethod
    def iou(a,b):
        x1,y1 = max(a[0],b[0]),max(a[1],b[1])
        x2,y2 = min(a[2],b[2]),min(a[3],b[3])
        intersection = max(0,x2-x1)*max(0,y2-y1)
        union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
        return intersection/union if union > 0 else 0

    def classify_all(self, detections, lanes, resolution):
        self.prepare(lanes,resolution)
        results,history,used = [],[],set()
        for d in detections:
            raw = 'unknown' if d.get('crop_clipped') else self.classify(d.get('bbox'),resolution)
            count = 1 if raw == 'inside' else 0
            if self.smooth and raw == 'inside':
                matches = [(self.iou(d['bbox'],p['bbox']),i,p) for i,p in enumerate(self.previous)
                           if i not in used and p['class_name'] == d['class_name']]
                matches.sort(key=lambda v:v[0],reverse=True)
                if matches and matches[0][0] >= .5:
                    _,i,p = matches[0];used.add(i);count = p['count']+1
            status = 'unknown' if self.smooth and raw == 'inside' and count < self.confirmation_frames else raw
            item = dict(d, lane_status=status, distance=None)
            results.append(item)
            # Unknown/missing evidence immediately clears inside confirmation.
            if raw == 'inside':
                history.append(dict(bbox=list(d['bbox']),class_name=d['class_name'],count=count))
        self.previous = history
        return results
