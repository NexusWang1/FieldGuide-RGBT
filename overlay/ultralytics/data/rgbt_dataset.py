# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""RGBT dataset patch for YOLO26.

Layout assumption:
    RGBT/
      images/train/*.png   # RGB
      image/train/*.png    # IR, same frame name as RGB
      labels/train/*.txt   # YOLO labels driven by RGB image names

The dataset scans RGB images as the master list, keeps only samples that also have
same-name IR images, and loads RGB+IR as one 6-channel image before augmentation,
so geometric transforms are applied to both modalities consistently.
"""

from __future__ import annotations

import glob
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics.data.augment import Format
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import LOGGER


class RGBTFormat(Format):
    """Format 6/4-channel RGBT images without breaking channel order.

    Ultralytics Format only reverses channels when C==3. For C==6 it leaves the
    RGB part in BGR order. This class converts only the first 3 channels BGR->RGB;
    IR channels are replicated grayscale, so channel order there is irrelevant.
    """

    def _format_img(self, img: np.ndarray) -> torch.Tensor:
        if len(img.shape) < 3:
            img = img[..., None]
        img = img.transpose(2, 0, 1)
        if random.uniform(0, 1) > self.bgr and img.shape[0] >= 3:
            img[:3] = img[:3][::-1]
        img = np.ascontiguousarray(img)
        return torch.from_numpy(img)


class RGBTYOLODataset(YOLODataset):
    """YOLO dataset that loads paired RGB + IR frames as one image tensor.

    Args:
        ir_root: IR root directory, usually `<dataset_root>/image`.
        mode: 'train' or 'val'.
        ir_channel_mode: 'rgb3' returns RGB(3) + IR gray replicated to 3 channels = 6 channels.
            'gray1' returns RGB(3) + IR gray(1) = 4 channels. Default is 'rgb3'.
        drop_unpaired: if True, RGB frames without same-name IR frames are dropped before label scan.
    """

    def __init__(self, *args, ir_root=None, mode="train", ir_channel_mode="rgb3", drop_unpaired=True, **kwargs):
        self.ir_root = ir_root
        self.mode = mode
        self.ir_channel_mode = ir_channel_mode
        self.drop_unpaired = drop_unpaired
        self.ir_files: list[str] = []
        self.unpaired_rgb: list[str] = []
        super().__init__(*args, **kwargs)

    def get_img_files(self, img_path):
        im_files = super().get_img_files(img_path)
        self._build_ir_mapping(im_files)
        return self.im_files

    def _rgb_to_ir_path(self, rgb_path: str) -> Path | None:
        p = Path(rgb_path)
        parts = list(p.parts)
        for i in range(len(parts) - 1, -1, -1):
            if parts[i] == "images":
                parts[i] = "image"
                return Path(*parts)
        if self.ir_root:
            return Path(self.ir_root) / self.mode / p.name
        return None

    def _build_ir_mapping(self, im_files: list[str]) -> None:
        paired_rgb, paired_ir, missing = [], [], []
        for rgb in im_files:
            ir = self._rgb_to_ir_path(rgb)
            ok = ir is not None and ir.exists()
            if not ok and ir is not None:
                # Fallback: same stem, any image suffix in the IR directory.
                matches = glob.glob(str(ir.parent / f"{Path(rgb).stem}.*"))
                matches = [m for m in matches if Path(m).suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}]
                if len(matches) == 1:
                    ir, ok = Path(matches[0]), True
            if ok:
                paired_rgb.append(rgb)
                paired_ir.append(str(ir))
            else:
                missing.append(rgb)
        self.unpaired_rgb = missing
        if missing:
            LOGGER.warning(
                f"{self.prefix}RGB-IR pairing: {len(missing)}/{len(im_files)} RGB frames have no same-name IR frame and were skipped. "
                f"Example: {missing[0]}"
            )
        if self.drop_unpaired:
            self.im_files = paired_rgb
            self.ir_files = paired_ir
        else:
            self.im_files = im_files
            self.ir_files = [str(self._rgb_to_ir_path(x)) for x in im_files]
        if not self.im_files:
            raise FileNotFoundError(f"{self.prefix}No paired RGB-IR samples found from {self.img_path}")

    def get_cache_hash(self) -> str:
        return super().get_cache_hash() + str(len(self.ir_files))

    def load_image(self, i: int, rect_mode: bool = True, resize_short: bool = False):
        rgb, hw_original, hw_resized = super().load_image(i, rect_mode=rect_mode, resize_short=resize_short)
        ir_path = self.ir_files[i]
        ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
        if ir is None:
            raise FileNotFoundError(f"IR image Not Found {ir_path}")
        if ir.shape[:2] != rgb.shape[:2]:
            ir = cv2.resize(ir, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        if self.ir_channel_mode == "gray1":
            img = np.concatenate([rgb, ir[..., None]], axis=2)
        else:
            img = np.concatenate([rgb, cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)], axis=2)
        return img, hw_original, hw_resized

    def build_transforms(self, hyp=None):
        transforms = super().build_transforms(hyp)
        if getattr(transforms, "transforms", None) and isinstance(transforms.transforms[-1], Format):
            old = transforms.transforms[-1]
            transforms.transforms[-1] = RGBTFormat(
                bbox_format=old.bbox_format,
                normalize=old.normalize,
                return_mask=old.return_mask,
                return_keypoint=old.return_keypoint,
                return_obb=old.return_obb,
                mask_ratio=old.mask_ratio,
                mask_overlap=old.mask_overlap,
                batch_idx=old.batch_idx,
                bgr=old.bgr,
            )
        return transforms
