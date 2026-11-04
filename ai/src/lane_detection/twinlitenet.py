"""Official TwinLiteNet+ Medium inference and straight ego-boundary extraction."""
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .twinlitenet_vendor.model import TwinLiteNetPlus

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LANE_WEIGHTS = ROOT / 'ai/checkpoints/lane_detection/twinlitenet/medium.pth'


@dataclass
class LaneGeometry:
    roi_top: float = .52
    roi_bottom: float = .98
    camera_center: float = .5
    min_span: float = .13
    residual: float = .008
    min_width: float = .12
    max_width: float = .90
    max_slope: float = 2.5
    samples: int = 30
    smooth_alpha: float = .35
    min_points: int = 8

    def __post_init__(self):
        if not (0 <= self.roi_top < self.roi_bottom <= 1 and
                0 < self.camera_center < 1 and 0 < self.min_span <= 1 and
                0 < self.min_width < self.max_width <= 1 and
                self.residual > 0 and self.max_slope > 0 and self.samples >= 2 and self.min_points >= 4 and
                0 < self.smooth_alpha <= 1):
            raise ValueError('Invalid lane geometry configuration')


class EgoLaneExtractor:
    """Row-run candidates, deterministic RANSAC, nearest bracketing boundaries.

    Work at <=640 pixels wide. Never retain a boundary on a missing frame.
    """
    def __init__(self, config=None, smooth=False, validate_pair=True):
        self.config = config or LaneGeometry()
        self.smooth = smooth
        self.validate_pair = validate_pair
        self.previous = {}
        self.diagnostics = {}

    def reset(self):
        self.previous.clear()

    def extract(self, mask):
        h, w = mask.shape
        c = self.config
        scale = min(1., 640 / w)
        small = cv2.resize(mask, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_NEAREST)
        sh, sw = small.shape
        candidates = []
        for y in range(int(sh*c.roi_top), int(sh*c.roi_bottom), max(1, sh//100)):
            xs = np.flatnonzero(small[y])
            for run in np.split(xs, np.flatnonzero(np.diff(xs) > 1)+1):
                if 0 < len(run) <= sw*.05:
                    candidates.append((float(run.mean())/sw, y/sh))
        self.diagnostics = dict(mask_pixels=int(np.count_nonzero(small)),
                                roi_pixels=int(np.count_nonzero(small[int(sh*c.roi_top):int(sh*c.roi_bottom)])),
                                candidate_count=len(candidates), short_support=0, no_consensus=0,
                                insufficient_span=0, invalid_projection=0, pair_rejected=False)
        points = np.asarray(candidates).reshape(-1, 2)
        lines = []
        rng = np.random.default_rng(42)
        for _ in range(8):
            if len(points) < c.min_points:
                self.diagnostics["short_support"] += 1
                break
            best = None
            for _ in range(100):
                p, q = points[rng.choice(len(points), 2, replace=False)]
                if abs(p[1]-q[1]) < c.min_span:
                    continue
                a = (p[0]-q[0])/(p[1]-q[1])
                b = p[0]-a*p[1]
                if abs(a) > c.max_slope:
                    continue
                inside = np.abs(points[:, 0] - a*points[:, 1]-b) < c.residual
                if best is None or inside.sum() > best.sum():
                    best = inside
            if best is None or best.sum() < c.min_points:
                self.diagnostics["no_consensus"] += 1
                break
            support = points[best]
            points = points[~best]
            if np.ptp(support[:, 1]) < c.min_span:
                self.diagnostics["insufficient_span"] += 1
                continue
            a, b = np.polyfit(support[:, 1], support[:, 0], 1)
            low, high = support[:, 1].min(), support[:, 1].max()
            bottom = a*c.roi_bottom+b
            if 0 <= bottom <= 1:
                lines.append((bottom, a, b, low, high))
            else:
                self.diagnostics["invalid_projection"] += 1
        self.diagnostics["line_count"] = len(lines)
        left = sorted((v for v in lines if v[0] < c.camera_center), reverse=True)
        right = sorted(v for v in lines if v[0] > c.camera_center)
        chosen = {'left_ego': left[0] if left else None, 'right_ego': right[0] if right else None}
        if self.validate_pair and left and right:
            l, r = left[0], right[0]
            top = max(l[3], r[3])
            width = r[0]-l[0]
            if not c.min_width <= width <= c.max_width or r[1]*top+r[2] <= l[1]*top+l[2]:
                self.diagnostics["pair_rejected"] = True
                chosen = dict.fromkeys(chosen)
        result = {}
        for side, line in chosen.items():
            result[side] = []
            if line is None:
                self.previous.pop(side, None)
                continue
            _, a, b, low, high = line
            current = np.array([a, b])
            if self.smooth and side in self.previous:
                previous = self.previous[side]
                # Do not blend a large lane change or a newly selected marking.
                if abs(np.dot(current-previous, [c.roi_bottom, 1])) < .08:
                    current = c.smooth_alpha*current + (1-c.smooth_alpha)*previous
            self.previous[side] = current
            for y in np.linspace(low, high, c.samples):
                x = current[0]*y+current[1]
                if 0 <= x < 1:
                    result[side].append((min(w-1, round(x*w)), min(h-1, round(y*h))))
        self.diagnostics['accepted'] = {side: bool(value) for side, value in result.items()}
        self.diagnostics['failure_stage'] = ('mask' if not self.diagnostics['roi_pixels'] else
                                            'postprocess' if not any(result.values()) else 'accepted')
        return result


class TwinLiteNetLaneModel:
    def __init__(self, weights=DEFAULT_LANE_WEIGHTS, device='auto', threshold=.5,
                 smooth=False, geometry=None):
        if not 0 <= threshold <= 1:
            raise ValueError('lane threshold must be in [0,1]')
        self.weights = Path(weights)
        if not self.weights.is_file():
            raise FileNotFoundError(f'Medium checkpoint missing: {self.weights}. See TWINLITENET.md.')
        if str(device).isdigit():
            device = f'cuda:{device}'
        self.device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; use --device cpu')
        if self.device.type not in ('cpu', 'cuda'):
            raise ValueError('device must be auto, cpu, cuda, cuda:N, or a GPU index')
        self.model = TwinLiteNetPlus(SimpleNamespace(config='medium'))
        state = torch.load(self.weights, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        self.threshold = threshold
        self.extractor = EgoLaneExtractor(geometry, smooth)
        self.timings = {}
        self.output_shapes = None
        self.input_shape = None
        self.inference_calls = 0

    def sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    @staticmethod
    def preprocess(frame):
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
            raise ValueError('Expected nonempty uint8 BGR frame (H,W,3)')
        h, w = frame.shape[:2]
        ratio = min(640/h, 640/w)
        nw, nh = round(w*ratio), round(h*ratio)
        dw, dh = (640-nw) % 32 / 2, (640-nh) % 32 / 2
        left, top = round(dw-.1), round(dh-.1)
        image = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        image = cv2.copyMakeBorder(image, top, round(dh+.1), left, round(dw+.1), cv2.BORDER_CONSTANT, value=(114,114,114))
        tensor = torch.from_numpy(np.ascontiguousarray(image[:,:,::-1].transpose(2,0,1))).unsqueeze(0)
        return tensor, (top, left, nh, nw)

    @torch.inference_mode()
    def predict(self, frame):
        self.sync()
        start = perf_counter()
        tensor, (top, left, nh, nw) = self.preprocess(frame)
        tensor = tensor.to(self.device, dtype=torch.float32).div_(255)
        self.input_shape = list(tensor.shape)
        self.sync()
        prepared = perf_counter()
        outputs = self.model(tensor)
        self.inference_calls += 1
        self.sync()
        inferred = perf_counter()
        self.output_shapes = [list(v.shape) for v in outputs]
        logits = outputs[1]
        if logits.ndim != 4 or logits.shape[:2] != (1, 2):
            raise RuntimeError(f'Unexpected lane output: {logits.shape}')
        # Crop exact integer borders, resize logits, then decode foreground.
        logits = logits[:,:,top:top+nh,left:left+nw]
        # Binary softmax foreground probability is sigmoid(logit1-logit0).
        # Linear interpolation commutes with subtraction, so resize one channel.
        margin = F.interpolate(logits[:,1:2]-logits[:,0:1], size=frame.shape[:2],
                               mode='bilinear', align_corners=False)
        cutoff = (-float('inf') if self.threshold == 0 else
                  float('inf') if self.threshold == 1 else
                  float(np.log(self.threshold/(1-self.threshold))))
        foreground = (margin > 0) & (margin >= cutoff)
        mask = foreground[0,0].to(torch.uint8).cpu().numpy()
        decoded = perf_counter()
        lanes = self.extractor.extract(mask)
        end = perf_counter()
        self.timings = dict(lane_preprocess_ms=(prepared-start)*1000,
                            lane_inference_ms=(inferred-prepared)*1000,
                            lane_decode_ms=(decoded-inferred)*1000,
                            lane_postprocess_ms=(end-inferred)*1000)
        return mask, lanes
