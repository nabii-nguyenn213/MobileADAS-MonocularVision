"""One-time CPU export of official pretrained CULane ResNet18 UFLDv2."""
import argparse
from pathlib import Path

import numpy as np
import torch

from ai.src.lane_detection.ufldv2 import DEFAULT_LANE_WEIGHTS, OUTPUT_SHAPES
from ai.src.lane_detection.ufldv2_export.model_culane import parsingNet


def export(weights, output):
    torch.set_num_threads(4)
    # Construct the reference architecture directly: get_model() hardcodes CUDA.
    # pretrained=False avoids an unrelated ImageNet download; all weights below
    # must match the official lane checkpoint with strict=True.
    model = parsingNet(
        pretrained=False, backbone="18", num_grid_row=200, num_cls_row=72,
        num_grid_col=100, num_cls_col=81, num_lane_on_row=4, num_lane_on_col=4,
        use_aux=False, input_height=320, input_width=1600, fc_norm=True,
    ).eval()
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    state = {k.removeprefix("module."): v for k, v in checkpoint["model"].items()}
    model.load_state_dict(state, strict=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sample = torch.randn(1, 3, 320, 1600, generator=torch.Generator().manual_seed(0))
    with torch.inference_mode():
        expected = model(sample)
        torch.onnx.export(
            model, sample, str(output), input_names=["input"],
            output_names=list(OUTPUT_SHAPES), opset_version=17, dynamo=False,
        )
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(str(output))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(output), sess_options=options, providers=["CPUExecutionProvider"])
    actual = session.run(list(OUTPUT_SHAPES), {"input": sample.numpy()})
    for name, value in zip(OUTPUT_SHAPES, actual):
        np.testing.assert_allclose(value, expected[name].numpy(), rtol=1e-3, atol=1e-4)
        print(f"{name}: {value.shape}; PyTorch/ONNX parity passed")
    print(f"Exported strictly loaded official checkpoint to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_LANE_WEIGHTS.with_suffix(".pth"))
    parser.add_argument("--output", type=Path, default=DEFAULT_LANE_WEIGHTS)
    args = parser.parse_args()
    export(args.weights, args.output)


if __name__ == "__main__":
    main()
