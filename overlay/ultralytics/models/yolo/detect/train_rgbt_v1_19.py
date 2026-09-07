# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""DetectionTrainer for YOLO26 RGBT training — v1.19 门控 CMEE + fp32 存档版。

构成 = v1.17b trainer 原样 + 存档安全补丁：
  1. 覆写 save_model：去 .half() fp32 存档
  2. 模块加载时把 engine.trainer 的 strip_optimizer 换成 fp32 版
  3. 每轮 save_model 时把模型 _guard_stats 落盘 guard_stats.json

环境变量全集：PHY_STAGE / PHY_PSCMT_RES / PHY_TFAM_IR / PHY_DUALSTAL / PHY_STALW
  / PHY_SMOKESTAL / PHY_SMOKEW / PHY_SMOKEDIL / PHY_AUXW
"""

from __future__ import annotations

import io
import json
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

import ultralytics.engine.trainer as _engine_trainer
from ultralytics import __version__
from ultralytics.data.rgbt_dataset import RGBTYOLODataset
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks_rgbt_v1_19 import PhyBridgeYOLO26
from ultralytics.utils import GIT, LOGGER, RANK, colorstr
from ultralytics.utils.patches import torch_load
from ultralytics.utils.torch_utils import convert_optimizer_state_dict_to_fp16, unwrap_model


def _strip_optimizer_fp32(f: str | Path, s: str = "", updates: dict[str, Any] | None = None):
    """strip_optimizer 的 fp32 版：流程与原版一致，但不做 x["model"].half()。"""
    try:
        x = torch_load(f, map_location=torch.device("cpu"))
        assert isinstance(x, dict), "checkpoint is not a Python dictionary"
        assert "model" in x, "'model' missing from checkpoint"
    except Exception as e:
        LOGGER.warning(f"Skipping {f}, not a valid Ultralytics model: {e}")
        return {}
    if x.get("ema"):
        x["model"] = x["ema"]  # replace model with EMA
    if hasattr(x["model"], "criterion"):
        x["model"].criterion = None
    for p in x["model"].parameters():
        p.requires_grad = False
    # 不 half()：融合路径 BN running stats 可能超 fp16 上限，保持 fp32 保真
    for k in ("optimizer", "best_fitness", "ema", "updates", "scaler"):
        x[k] = None
    x["epoch"] = -1
    combined = {**x, **(updates or {})}
    torch.save(combined, s or f)
    mb = os.path.getsize(s or f) / 1e6
    LOGGER.info(f"Optimizer stripped (fp32, v1.19) from {f},{f' saved as {s},' if s else ''} {mb:.1f}MB")
    return combined


# final_eval 在 engine.trainer 模块命名空间里查 strip_optimizer，必须在
# model.train() 之前换掉（本模块由根目录训练脚本在 train 前 import，时序安全）
_engine_trainer.strip_optimizer = _strip_optimizer_fp32


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip() in {"1", "true", "True"}


class RGBTDetectionTrainer(DetectionTrainer):
    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        raw = yaml.safe_load(open(self.args.data, encoding="utf-8"))
        ir_images = raw.get("ir_images", "image")
        ir_root = str(Path(raw.get("path", self.data.get("path", ""))) / ir_images)
        if self.args.cache:
            LOGGER.warning("RGBT patch forces cache=False to avoid stale RGB-only image caches.")
        return RGBTYOLODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect or mode == "val",
            cache=False,
            single_cls=self.args.single_cls or False,
            stride=gs,
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
            ir_root=ir_root,
            mode=mode,
            ir_channel_mode="rgb3",
            drop_unpaired=True,
        )

    def save_model(self):
        """fp32 存档：与 engine.trainer.save_model 同流程，仅去掉 .half()；
        附带把 guard 计数落盘。"""
        ema = unwrap_model(self.ema.ema)
        if not all(torch.isfinite(v).all() for v in ema.state_dict().values() if isinstance(v, torch.Tensor)):
            model_sd = unwrap_model(self.model).state_dict()
            for k, v in ema.state_dict().items():
                if isinstance(v, torch.Tensor) and not torch.isfinite(v).all() and torch.isfinite(model_sd[k]).all():
                    v.copy_(model_sd[k])
        ema = deepcopy(ema).to(memory_format=torch.contiguous_format)  # 保持 fp32
        if hasattr(ema, "criterion"):
            ema.criterion = None
        for v in ema.state_dict().values():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                torch.nan_to_num_(v)  # fp32 下只兜底非有限值

        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                "model": None,  # resume and final checkpoints derive from EMA
                "ema": ema,
                "updates": self.ema.updates,
                "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                "scaler": self.scaler.state_dict(),
                "train_args": vars(self.args),
                "train_metrics": {**self.metrics, "fitness": self.fitness},
                "train_results": self.read_results_csv(),
                "date": datetime.now().astimezone().isoformat(),
                "version": __version__,
                "git": {
                    "root": str(GIT.root),
                    "branch": GIT.branch,
                    "commit": GIT.commit,
                    "message": GIT.message,
                    "origin": GIT.origin,
                },
                "license": "AGPL-3.0 (https://www.gnu.org/licenses/agpl-3.0.html)",
                "docs": "https://docs.ultralytics.com",
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()

        self.wdir.mkdir(parents=True, exist_ok=True)
        self.last.write_bytes(serialized_ckpt)
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)

        # guard 计数落盘：每轮覆盖写，训练中断也留有最新现场
        try:
            stats = getattr(unwrap_model(self.model), "_guard_stats", None)
            if stats:
                (self.wdir.parent / "guard_stats.json").write_text(json.dumps(stats))
        except Exception:
            pass
        return True

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        stage = int(os.getenv("PHY_STAGE", "3"))
        pscmt_res = _env_flag("PHY_PSCMT_RES")
        tfam_ir = _env_flag("PHY_TFAM_IR", "1")
        dual_stal = _env_flag("PHY_DUALSTAL")
        stal_w = float(os.getenv("PHY_STALW", "0.25"))
        smoke_stal = _env_flag("PHY_SMOKESTAL")
        smoke_w = float(os.getenv("PHY_SMOKEW", "0.25"))
        smoke_dilate = float(os.getenv("PHY_SMOKEDIL", "3.0"))
        model = self.set_model_names_for_load(
            PhyBridgeYOLO26(
                cfg=cfg or "yolo26m-p2.yaml",
                ch=3,
                nc=self.data["nc"],
                verbose=verbose and RANK == -1,
                scale="m",
                stage=stage,
                pscmt_res=pscmt_res,
                tfam_ir=tfam_ir,
                dual_stal=dual_stal,
                stal_w=stal_w,
                smoke_stal=smoke_stal,
                smoke_w=smoke_w,
                smoke_dilate=smoke_dilate,
            )
        )
        if weights:
            model.load_pretrained(weights)
        LOGGER.info(f"PhyBridgeYOLO26 v1.19 (gated CMEE + fp32 save) stage={stage} "
                    f"pscmt_res={pscmt_res} tfam_ir={tfam_ir} "
                    f"dual_stal={dual_stal}(w={stal_w}) "
                    f"smoke_stal={smoke_stal}(w={smoke_w},dil={smoke_dilate}) "
                    f"aux_w={os.getenv('PHY_AUXW', '0.0')}")
        return model
