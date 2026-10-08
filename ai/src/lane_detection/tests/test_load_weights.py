from ai.src.lane_detection.config import LaneDetectionConfig
from ai.src.lane_detection.detector import LaneDetector
from ai.src.lane_detection.models.ufldv2 import UFLDv2

config = LaneDetectionConfig()
config.checkpoint_path = "ai/checkpoints/lane_detection/tusimple_res18.pth"

model = UFLDv2()

detector = LaneDetector(
    model=model,
    config=config,
)

detector.load_weights()

print("Checkpoint loaded successfully")
