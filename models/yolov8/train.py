from ultralytics import YOLO
from pathlib import Path
import yaml


def train_yolov8(config_path: str, data_yaml: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    model = YOLO(cfg["model"])

    model.train(
        data=data_yaml,
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        optimizer=cfg["optimizer"],
    )


if __name__ == "__main__":
    train_yolov8(
        config_path="src/models/yolov8/config.yaml",
        data_yaml="data/weapon_detection/data.yaml",
    )
