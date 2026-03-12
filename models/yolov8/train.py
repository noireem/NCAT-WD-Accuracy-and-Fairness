"""Knowledge-distillation training for YOLOv8.

Teacher = old model trained on DatasetNinja (train13/weights/best.pt)
Student = fresh yolov8n initialised from COCO pretrained weights

The student learns from both the ground-truth labels AND the teacher's
soft predictions (classification KL-div + box smooth-L1).
"""
from __future__ import annotations

import argparse

from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors
try:
    from ultralytics.utils.torch_utils import de_parallel
except ImportError:
    from ultralytics.utils.torch_utils import unwrap_model as de_parallel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_path(base_dir: Path, value: str | None, default: Path | None = None) -> str | None:
    if value is None:
        return str(default.resolve()) if default is not None else None
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def _default_resume_checkpoint(base_dir: Path, cfg: dict) -> Path:
    project_dir = Path(_resolve_path(base_dir, cfg.get("project"), base_dir / "runs" / "detect"))
    run_name = cfg.get("name", "train")
    return (project_dir / run_name / "weights" / "last.pt").resolve()


def load_teacher(weights_path: str, device: torch.device) -> DetectionModel:
    """Load the teacher model in eval mode with frozen weights."""
    teacher = YOLO(weights_path).model.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    # Keep detect head in training mode so it returns raw feature maps
    teacher.model[-1].training = True
    return teacher


def _extract_preds(preds, no, reg_max, nc):
    """Turn predictions into (pred_distri, pred_scores, feats) across ultralytics API variants."""
    if isinstance(preds, dict):
        return (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
            preds["feats"],
        )

    feats = preds[1] if isinstance(preds, tuple) else preds
    raw = torch.cat([xi.view(feats[0].shape[0], no, -1) for xi in feats], 2)
    pred_distri, pred_scores = raw.split((reg_max * 4, nc), 1)
    pred_scores = pred_scores.permute(0, 2, 1).contiguous()
    pred_distri = pred_distri.permute(0, 2, 1).contiguous()
    return pred_distri, pred_scores, feats


# ---------------------------------------------------------------------------
# Distillation loss  (wraps the standard v8 detection loss)
# ---------------------------------------------------------------------------

class DistillationLoss(v8DetectionLoss):
    """Standard YOLOv8 loss + soft-label distillation from teacher."""

    def __init__(self, model, teacher, temperature=2.0, cls_w=0.5, box_w=0.25):
        super().__init__(model)
        self.teacher = teacher
        self.T = temperature
        self.cls_w = cls_w
        self.box_w = box_w

    def __call__(self, preds, batch):
        # ---- normal detection loss (unchanged) ----
        total_loss, loss_items = super().__call__(preds, batch)

        # ---- teacher forward ----
        imgs = batch["img"]
        with torch.no_grad():
            t_preds = self.teacher(imgs)

        # ---- extract student & teacher raw predictions ----
        s_distri, s_scores, s_feats = _extract_preds(preds, self.no, self.reg_max, self.nc)
        t_distri, t_scores, _ = _extract_preds(t_preds, self.no, self.reg_max, self.nc)

        # ---- classification KL divergence ----
        T = self.T
        # weight by teacher confidence so we focus on "useful" anchors
        t_conf = t_scores.detach().sigmoid().amax(dim=-1).clamp_min(1e-4)
        kl = F.kl_div(
            F.log_softmax(s_scores / T, dim=-1),
            F.softmax(t_scores.detach() / T, dim=-1),
            reduction="none",
        ).sum(dim=-1)                          # (B, num_anchors)
        cls_kd = (kl * t_conf).sum() / t_conf.sum() * (T ** 2)

        # ---- box smooth-L1 ----
        anchor_points, stride_tensor = make_anchors(s_feats, self.stride, 0.5)
        s_boxes = self.bbox_decode(anchor_points, s_distri) * stride_tensor
        t_boxes = self.bbox_decode(anchor_points, t_distri) * stride_tensor
        box_l1 = F.smooth_l1_loss(s_boxes, t_boxes, reduction="none").mean(dim=-1)
        box_kd = (box_l1 * t_conf).sum() / t_conf.sum()

        batch_size = s_scores.shape[0]
        kd_loss = (self.cls_w * cls_kd + self.box_w * box_kd) * batch_size
        total_loss = total_loss + kd_loss
        return total_loss, loss_items


# ---------------------------------------------------------------------------
# Trainer subclass
# ---------------------------------------------------------------------------

class DistillationTrainer(DetectionTrainer):
    def __init__(self, teacher_weights, temperature, cls_w, box_w,
                 overrides=None, _callbacks=None):
        self._teacher_weights = teacher_weights
        self._temperature = temperature
        self._cls_w = cls_w
        self._box_w = box_w
        super().__init__(overrides=overrides, _callbacks=_callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose)
        if weights:
            model.load(weights)
        return model

    def _setup_train(self, world_size=None):
        """Run normal setup, then swap in the distillation loss once the model is initialized."""
        if world_size is None:
            super()._setup_train()
        else:
            super()._setup_train(world_size)
        model = de_parallel(self.model)
        teacher = load_teacher(self._teacher_weights, self.device)
        model.criterion = DistillationLoss(
            model, teacher,
            temperature=self._temperature,
            cls_w=self._cls_w,
            box_w=self._box_w,
        )
        print(f"\n[distill] Teacher loaded from {self._teacher_weights}")
        print(f"[distill] T={self._temperature}  cls_w={self._cls_w}  box_w={self._box_w}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def train_yolov8(config_path: str, resume: bool | str = False):
    config_file = Path(config_path).resolve()
    base_dir = config_file.parent
    with config_file.open("r", encoding="utf-8") as cfg_f:
        cfg = yaml.safe_load(cfg_f)

    data_yaml = _resolve_path(base_dir, cfg.get("data"), base_dir / "data.yaml")
    teacher_weights = _resolve_path(base_dir, cfg.get("teacher_model"))
    if not teacher_weights:
        raise ValueError("config.yaml must define 'teacher_model' for distillation.")

    overrides = {
        "model": _resolve_path(base_dir, cfg.get("model")),
        "data": data_yaml,
        "epochs": cfg.get("epochs", 75),
        "imgsz": cfg.get("imgsz", 640),
        "batch": cfg.get("batch", 16),
        "optimizer": cfg.get("optimizer", "auto"),
        "patience": cfg.get("patience", 20),
        "project": _resolve_path(base_dir, cfg.get("project")) if cfg.get("project") else None,
        "name": cfg.get("name"),
        "device": cfg.get("device"),
        "workers": cfg.get("workers", 0),
        "pretrained": cfg.get("pretrained", True),
        "amp": cfg.get("amp", True),
        "exist_ok": cfg.get("exist_ok", False),
    }

    if resume:
        resume_path = _default_resume_checkpoint(base_dir, cfg) if resume is True else Path(str(resume))
        if not resume_path.is_absolute():
            resume_path = (Path.cwd() / resume_path).resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        overrides["resume"] = str(resume_path)

    overrides = {k: v for k, v in overrides.items() if v is not None}

    trainer = DistillationTrainer(
        teacher_weights=teacher_weights,
        temperature=float(cfg.get("distill_temperature", 2.0)),
        cls_w=float(cfg.get("distill_cls_weight", 0.5)),
        box_w=float(cfg.get("distill_box_weight", 0.25)),
        overrides=overrides,
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or resume YOLOv8 distillation.")
    parser.add_argument(
        "--cfg",
        default=str(Path(__file__).resolve().parent / "config.yaml"),
        help="Path to the training config YAML.",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        help="Resume from the default checkpoint for this run, or provide a checkpoint path.",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Display training progress in the terminal.",
    )
    args = parser.parse_args()

    if args.show_progress:
        import logging
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        logging.getLogger("ultralytics").setLevel(logging.INFO)

    train_yolov8(args.cfg, resume=args.resume)
