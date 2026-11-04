"""Standalone CULane UFLDv2 ONNX inference (Vehicle-CV-ADAS preprocessing).

No dependency on the project's previous lane models or selection logic.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LANE_WEIGHTS = ROOT / "ai/checkpoints/lane_detection/ufldv2/culane_res18.onnx"
OUTPUT_SHAPES = {
    "loc_row": (1, 200, 72, 4),
    "loc_col": (1, 100, 81, 4),
    "exist_row": (1, 2, 72, 4),
    "exist_col": (1, 2, 81, 4),
}
LANE_NAMES = ("left-side", "left-ego", "right-ego", "right-side")


@dataclass
class LaneDetection:
    lane_id: int
    name: str
    points: list[tuple[int, int]]

    @property
    def detected(self):
        return len(self.points) > 2


def parse_device(device):
    """Return (cpu|cuda|auto, GPU index) for both inference engines."""
    value = str(device).lower()
    if value in ("cpu", "auto"):
        return value, 0
    if value.isdigit():
        return "cuda", int(value)
    if value == "cuda":
        return "cuda", 0
    if value.startswith("cuda:") and value[5:].isdigit():
        return "cuda", int(value[5:])
    raise ValueError("device must be cpu, auto, cuda, cuda:N, or a GPU index")


class UFLDv2:
    """CULane ResNet18/34 UFLDv2 export, input NCHW 1x3x320x1600.

    Returns four LaneDetection objects in model order, with empty points for
    missing lanes. Coordinates refer to the complete original image.
    """

    def __init__(self, weights=DEFAULT_LANE_WEIGHTS, device="auto"):
        weights = Path(weights)
        if not weights.is_file():
            raise FileNotFoundError(
                f"UFLDv2 ONNX weights not found: {weights}. "
                "See ai/src/lane_detection/UFLDV2.md for download/export instructions."
            )
        import onnxruntime as ort

        kind, index = parse_device(device)
        available = ort.get_available_providers()
        if kind == "cuda" and "CUDAExecutionProvider" not in available:
            raise RuntimeError("CUDA requested but ONNX Runtime CUDA provider is unavailable")
        providers = ["CPUExecutionProvider"]
        if kind != "cpu" and "CUDAExecutionProvider" in available:
            providers.insert(0, ("CUDAExecutionProvider", {"device_id": index}))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(weights), sess_options=options, providers=providers)
        if kind == "cuda" and self.session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError("ONNX Runtime could not initialize CUDA; use --device cpu")
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or inputs[0].shape != [1, 3, 320, 1600]:
            raise ValueError("Expected a CULane UFLDv2 input of shape [1,3,320,1600]")
        self.input_name = inputs[0].name
        types = {"tensor(float)": np.float32, "tensor(float16)": np.float16}
        if inputs[0].type not in types:
            raise ValueError("Expected float32 or float16 ONNX input")
        self.input_dtype = types[inputs[0].type]
        # Shape matching also supports upstream exports with numeric output names.
        outputs = self.session.get_outputs()
        self.output_names = []
        for name, shape in OUTPUT_SHAPES.items():
            matches = [out.name for out in outputs if tuple(out.shape) == shape]
            if len(matches) != 1:
                raise ValueError(f"Expected one {name} output with shape {shape}")
            self.output_names.append(matches[0])
        if len(outputs) != 4:
            raise ValueError("Expected exactly four UFLDv2 output tensors")

    @staticmethod
    def preprocess(frame):
        if (not isinstance(frame, np.ndarray) or frame.ndim != 3
                or frame.shape[2] != 3 or frame.size == 0 or frame.dtype != np.uint8):
            raise ValueError("Expected a non-empty uint8 OpenCV BGR image (H,W,3)")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (1600, int(320 / 0.6)))
        cropped = resized[-320:].astype(np.float32) / 255.0
        normalized = (cropped - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
            [0.229, 0.224, 0.225], np.float32
        )
        return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])

    @staticmethod
    def decode(outputs, image_shape):
        """Local softmax around argmax (+/-1), using upstream existence gates."""
        for name, shape in OUTPUT_SHAPES.items():
            if name not in outputs or outputs[name].shape != shape:
                raise ValueError(f"Invalid {name} tensor; expected {shape}")
            if not np.isfinite(outputs[name]).all():
                raise ValueError(f"Non-finite {name} tensor")
        height, width = image_shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError("Image dimensions must be positive")
        lanes = [LaneDetection(i, name, []) for i, name in enumerate(LANE_NAMES)]
        for axis, ids, anchors, threshold in (
            ("row", (1, 2), np.linspace(0.42, 1, 72), 72 / 2),
            ("col", (0, 3), np.linspace(0, 1, 81), 81 / 4),
        ):
            loc = outputs[f"loc_{axis}"][0].astype(np.float32)
            valid = outputs[f"exist_{axis}"].argmax(1)[0]
            peaks = loc.argmax(0)
            grid_size = loc.shape[0]
            for lane_id in ids:
                if valid[:, lane_id].sum() <= threshold:
                    continue
                for k in np.flatnonzero(valid[:, lane_id]):
                    peak = peaks[k, lane_id]
                    indices = np.arange(max(0, peak - 1), min(grid_size - 1, peak + 1) + 1)
                    logits = loc[indices, k, lane_id]
                    probabilities = np.exp(logits - logits.max())
                    coordinate = (np.dot(probabilities / probabilities.sum(), indices) + 0.5) / (grid_size - 1)
                    if axis == "row":
                        x, y = int(coordinate * width), int(anchors[k] * height)
                    else:
                        x, y = int(anchors[k] * width), int(coordinate * height)
                    # Upstream's normalized endpoints can lie at width/height.
                    # Drop out-of-frame samples; never invent or clamp points.
                    if 0 <= x < width and 0 <= y < height:
                        lanes[lane_id].points.append((x, y))
                if not lanes[lane_id].detected:
                    lanes[lane_id].points.clear()
        return lanes

    def predict(self, frame):
        tensor = self.preprocess(frame).astype(self.input_dtype, copy=False)
        values = self.session.run(self.output_names, {self.input_name: tensor})
        return self.decode(dict(zip(OUTPUT_SHAPES, values)), frame.shape)
