#!/usr/bin/env python3
"""Single source of truth for image preprocessing.

Both the training dataset and inference must use IDENTICAL preprocessing,
otherwise we get silent train/serve drift. This module is imported by both.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image
from torchvision import transforms

# ImageNet normalisation stats (MobileNetV3-Small was pretrained with these).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMG_SIZE = 224


def build_eval_transform() -> transforms.Compose:
    """Deterministic eval/inference transform: resize → tensor → normalise."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_train_transform() -> transforms.Compose:
    """Stochastic training transform: + colour jitter, motion blur, light noise."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def preprocess_bgr(frame_bgr) -> np.ndarray:
    """BGR ndarray (from OpenCV) → normalised float32 (1, 3, 224, 224) numpy.

    Used by inference pipelines (ONNX / Torch). Mirrors the eval transform
    above so train + inference are identical.
    """
    import cv2  # local import — vision module may be imported in headless tests
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    t = build_eval_transform()
    return t(img).unsqueeze(0).numpy().astype(np.float32)


def preprocess_pair(frame_bgr_now, frame_bgr_prev: Optional[np.ndarray]) -> np.ndarray:
    """Stack current + previous frame → (1, 6, 224, 224) for temporal context.

    If prev is None (first frame), prev is set equal to current → zero motion.
    """
    cur = preprocess_bgr(frame_bgr_now)             # (1, 3, 224, 224)
    if frame_bgr_prev is None:
        prev = cur.copy()
    else:
        prev = preprocess_bgr(frame_bgr_prev)
    return np.concatenate([cur, prev], axis=1)      # (1, 6, 224, 224)
