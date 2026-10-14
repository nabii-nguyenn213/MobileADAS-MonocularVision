# Vietnamese-Road-Vehicle-Detection-A-Comparison-of-YOLO-Models
Comparison of YOLOv8n, YOLO11n, and YOLO26n for vehicle detection on Vietnamese roads using datasets of 1,950 and 6,525 images.
## Datasets

| Dataset | Images | Download |
| --- | ---: | --- |
| Small dataset | 1,950 | [Kaggle](https://www.kaggle.com/datasets/phmc2511/test-6) |
| Large dataset | 6,525 | [Kaggle](https://www.kaggle.com/datasets/phmc2511/vehicle-ver-2) |
## Validation Results — Small Dataset

Dataset: **1,950 images**
- Training: 1,365 images
- Validation: 390 images
- Test: 195 images

Models were trained for 30 epochs with an image size of 640 and a batch size of 16. Results below are measured on the validation split.

| Model | Precision (%) | Recall (%) | mAP@50 (%) | mAP@50–95 (%) |
| --- | ---: | ---: | ---: | ---: |
| YOLOv8n | **87.45** | **75.38** | 83.96 | **57.02** |
| YOLO11n | 85.78 | 72.66 | **84.00** | 52.76 |
| YOLO26n | 85.53 | 73.07 | 83.97 | 56.85 |
## Test Results — Small Dataset

Dataset: **1,950 images**
- Training: 1,365 images
- Validation: 390 images
- Test: 195 images

Results below are measured on the **195-image test split** using each model’s best checkpoint.
| Model | Precision (%) | Recall (%) | mAP@50 (%) | mAP@50–95 (%) |
| --- | --- | --- | --- | --- |
| yolov8n | 89.88 | 74.62 | 81.22 | 57.14 |
| yolo11n | 87.01 | 72.8 | 79.07 | 54.66 |
| yolo26n | 88.33 | 71.63 | 79.71 | 55.88 |
## Validation Results — Large Dataset

Dataset: **6,525 images**
- Training: 4,567 images
- Validation: 1,305 images
- Test: 653 images

Models were trained for 30 epochs with an image size of 640 and a batch size of 16. Results below are measured on the validation split.

| Model | Precision (%) | Recall (%) | mAP@50 (%) | mAP@50–95 (%) |
| --- | ---: | ---: | ---: | ---: |
| YOLOv8n | 85.97 | **79.97** | 87.27 | 60.72 |
| YOLO11n | 86.73 | 78.45 | 86.51 | 60.19 |
| YOLO26n | **89.35** | 78.45 | **87.47** | **61.21** |
## Test Results — Large Dataset

Dataset: **1,950 images**
- Training: 4,567 images
- Validation: 1,305 images
- Test: 653 images

Results below are measured on the **195-image test split** using each model’s best checkpoint.
| Model | Precision (%) | Recall (%) | mAP@50 (%) | mAP@50–95 (%) |
| --- | --- | --- | --- | --- |
| yolov8n | **88.57** | **78.32** | **84.53** | **58.93** |
| yolo11n | 87.56 | 78.0 | 83.45 | 58.46 |
| yolo26n | 88.23 | 76.98 | 84.35 | 58.72 |

