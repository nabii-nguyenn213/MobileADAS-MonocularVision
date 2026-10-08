from dataclasses import dataclass

@dataclass
class CarDetectionConfig: 
    model_path = "26n_best.pt"
    img_size = 640

    class_name = {
            0: "car", 
            1: "motobike", 
            2: "bus", 
            3: "pickup_truck", 
            4: "truck", 
    }

    confidence_threshold = 0.25
