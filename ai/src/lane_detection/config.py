from dataclasses import dataclass
import torch

@dataclass
class LaneDetectionConfig: 
    model_name = "ufldv2"
    backbone = "resnet18"

    input_width = 800
    input_height = 320

    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = f"checkpoints/lane_detection/{model_name}_{backbone}.pth"

    confidence_threshold = 0.5
