"""Quick 1-epoch validation of the distillation pipeline."""
import traceback
from pathlib import Path
import yaml
from models.yolov8.train import DistillationTrainer, _resolve_path
from src.shared_paths import require_weights_path

cfg_path = Path("models/yolov8/config.yaml").resolve()
base_dir = cfg_path.parent
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

data_yaml = _resolve_path(base_dir, cfg.get("data"), base_dir / "data.yaml")
teacher_weights = _resolve_path(base_dir, cfg.get("teacher_model"))
teacher_weights = str(require_weights_path(teacher_weights))

overrides = {
    "model": _resolve_path(base_dir, cfg.get("model")),
    "data": data_yaml,
    "epochs": 1,
    "imgsz": 640,
    "batch": 4,
    "optimizer": "auto",
    "patience": 20,
    "project": _resolve_path(base_dir, cfg.get("project")),
    "name": "distill_validate",
    "workers": 0,
    "pretrained": True,
    "amp": True,
    "exist_ok": True,
}
overrides["model"] = str(require_weights_path(overrides["model"]))
overrides = {k: v for k, v in overrides.items() if v is not None}

try:
    trainer = DistillationTrainer(
        teacher_weights=teacher_weights,
        temperature=2.0,
        cls_w=0.5,
        box_w=0.25,
        overrides=overrides,
    )
    trainer.train()
    print("\n=== VALIDATION RUN COMPLETED SUCCESSFULLY ===")
except Exception:
    traceback.print_exc()
    print("\n=== VALIDATION RUN FAILED ===")
