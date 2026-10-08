from ultralytics.models import YOLO
import numpy as np 

CLASS_NAMES = {
    0: "car",
    1: "motorbike",
    2: "bus",
    3: "pickup_truck",
    4: "truck",
}

class VehicleModel: 
    def __init__(self, model_path=None, img_size=640, confidence_threshold=0.25): 
        self.model_path = "ai/checkpoints/vehicle_detection/26n_best.pt" if model_path is None else model_path
        self.img_size = img_size
        self.confidence_threshold = confidence_threshold

        self.model = YOLO(self.model_path)

    def predict(self, frame:np.ndarray): 
        if frame is None or frame.size==0: 
            raise ValueError("Input frame is empty")
        results = self.model.predict(source=frame, imgsz=self.img_size, conf=self.confidence_threshold, verbose=False)
        detections = []
        for result in results: 
            if result.boxes is None: 
                continue
            for box in result.boxes: 
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                detections.append({
                    "class_id": class_id,
                    "class_name": CLASS_NAMES.get(
                        class_id,
                        str(class_id),
                    ),
                    "confidence": confidence,
                    "bbox": [
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                    ],
                })
        return detections
