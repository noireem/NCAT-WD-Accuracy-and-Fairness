from ultralytics import YOLO


def detect_objects(image_path: str, weights_path: str, conf: float = 0.7):
    model = YOLO(weights_path)
    results = model(image_path, conf=conf)

    detections = []

    for box in results[0].boxes:
        detections.append({
            "class_id": int(box.cls),
            "confidence": float(box.conf),
            "bbox_xyxy": box.xyxy.tolist()[0]
        })

    return detections


if __name__ == "__main__":
    detections = detect_objects(
        image_path="data/sample.jpg",
        weights_path="runs/detect/train/weights/best.pt",
        conf=0.7,
    )

    for d in detections:
        print(d)