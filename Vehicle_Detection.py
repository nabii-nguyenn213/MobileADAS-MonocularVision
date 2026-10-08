from ultralytics import YOLO
import cv2


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "/content/26n_best.pt"
IMAGE_PATH = "/content/Test_060.jpg"

OUTPUT_PATH = "/content/result.jpg"

IMG_SIZE = 640
CONF_THRESHOLD = 0.25

CLASS_NAMES = {
    0: "car",
    1: "motorbike",
    2: "bus",
    3: "pickup_truck",
    4: "truck"
}


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading YOLO26n model...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# VEHICLE DETECTION
# ============================================================

def detect_vehicles(frame):

    results = model.predict(
        source=frame,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        verbose=False
    )

    detections = []

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "class_id": class_id,
                "class_name": CLASS_NAMES.get(
                    class_id,
                    str(class_id)
                ),
                "confidence": confidence,
                "bbox": [
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2)
                ]
            })

    return detections


# ============================================================
# DRAW DETECTIONS
# ============================================================

def draw_detections(frame, detections):

    output = frame.copy()

    for detection in detections:

        x1, y1, x2, y2 = detection["bbox"]

        class_name = detection["class_name"]
        confidence = detection["confidence"]

        label = f"{class_name} {confidence:.2f}"

        # Bounding box
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2
        )

        text_y = max(y1, text_height + 10)

        # Text background
        cv2.rectangle(
            output,
            (x1, text_y - text_height - 10),
            (x1 + text_width, text_y),
            (0, 255, 0),
            -1
        )

        # Label
        cv2.putText(
            output,
            label,
            (x1, text_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

    return output


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n========================================")
    print("YOLO26n VEHICLE DETECTION")
    print("========================================")

    # Read image
    frame = cv2.imread(IMAGE_PATH)

    if frame is None:
        raise FileNotFoundError(
            f"Cannot read image: {IMAGE_PATH}"
        )

    print(f"Image: {IMAGE_PATH}")
    print(
        f"Original size: "
        f"{frame.shape[1]} x {frame.shape[0]}"
    )

    print(f"Inference size: {IMG_SIZE} x {IMG_SIZE}")

    # Detect
    detections = detect_vehicles(frame)

    # Draw
    output = draw_detections(
        frame,
        detections
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\nDetections:")
    print("----------------------------------------")

    if len(detections) == 0:

        print("No vehicles detected.")

    else:

        for i, detection in enumerate(
            detections,
            start=1
        ):

            print(
                f"{i}. "
                f"{detection['class_name']} | "
                f"Confidence: "
                f"{detection['confidence']:.3f} | "
                f"BBox: "
                f"{detection['bbox']}"
            )

    print("----------------------------------------")
    print(
        f"Total vehicles: "
        f"{len(detections)}"
    )

    # ========================================================
    # SAVE RESULT
    # ========================================================

    success = cv2.imwrite(
        OUTPUT_PATH,
        output
    )

    if success:
        print(
            f"\nResult saved successfully:"
            f"\n{OUTPUT_PATH}"
        )
    else:
        print("\nERROR: Could not save result image.")