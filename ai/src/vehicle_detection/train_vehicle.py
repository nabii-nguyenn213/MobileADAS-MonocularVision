import gc
import torch
from ultralytics import YOLO

# Choose "small" (1,950 images) or "large" (6,525 images).
DATASET = "large"

FOLDERS = {
    "small": "vehicle_1950",
    "large": "vehicle_large",
}

data = f"/kaggle/working/{FOLDERS[DATASET]}/data.yaml"
project = f"/kaggle/working/vehicle_comparison_{DATASET}"

models = ["yolov8n", "yolo11n", "yolo26n"]

if not torch.cuda.is_available():
    raise RuntimeError("Enable GPU in Kaggle settings.")

for name in models:
    model = YOLO(f"{name}.pt")

    model.train(
        data=data,
        epochs=30,
        imgsz=640,
        batch=16,
        device=0,
        workers=2,
        patience=0,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        seed=42,
        deterministic=True,
        amp=True,
        cache=False,
        save=True,
        plots=True,
        project=project,
        name=f"{name}_{DATASET}",
        exist_ok=False,
    )

    print("Results:", model.trainer.save_dir)

    del model
    gc.collect()
    torch.cuda.empty_cache()