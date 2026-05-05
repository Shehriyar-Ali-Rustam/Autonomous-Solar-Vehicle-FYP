#!/usr/bin/env python3
"""Production inference wrapper. Loads ONNX (fast) or PyTorch checkpoint.

Both backends use the SAME preprocessing as the training pipeline
(vision/transforms.py) — no train/serve drift.

Public API (unchanged from before):
    Predictor(model_path).predict(frame_bgr, sensors, gps_valid, gps_speed,
                                  gps_heading, prev_action, yolo,
                                  prev_frame_bgr=None) -> dict
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from vision.transforms import preprocess_pair
from .actions import ACTION_NAMES, STOP
from .dataset import build_state_vector

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import torch
    from .model import DecisionModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ---------------- Backends ----------------
class ONNXInference:
    def __init__(self, onnx_path: str, providers=None):
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime not installed")
        providers = providers or ['CPUExecutionProvider']
        self.sess = ort.InferenceSession(onnx_path, providers=providers)

    def predict_logits(self, image_np: np.ndarray, state_np: np.ndarray) -> np.ndarray:
        out = self.sess.run(['logits'], {'image': image_np, 'state': state_np})[0]
        return out


class TorchInference:
    def __init__(self, ckpt_path: str, device: str = 'cpu'):
        if not TORCH_AVAILABLE:
            raise ImportError("torch not installed")
        self.device = device
        self.model = DecisionModel(pretrained_backbone=False).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        # Both raw state_dicts and metadata-wrapped checkpoints
        sd = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt
        self.model.load_state_dict(sd)
        self.model.eval()

    def predict_logits(self, image_np: np.ndarray, state_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            img = torch.from_numpy(image_np).to(self.device)
            st  = torch.from_numpy(state_np.astype(np.float32)).to(self.device)
            logits = self.model(img, st).cpu().numpy()
        return logits


# ---------------- Predictor ----------------
def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


class Predictor:
    """Unified ONNX/Torch predictor. Path ending in `.onnx` → ONNX, else Torch."""

    def __init__(self, model_path: str, device: str = 'cpu'):
        self.model_path = model_path
        self.device = device
        if model_path.endswith('.onnx'):
            self.backend = ONNXInference(model_path)
            self.kind = 'onnx'
        else:
            self.backend = TorchInference(model_path, device)
            self.kind = 'torch'

    def predict(self,
                frame_bgr,
                sensors: dict,
                gps_valid: int,
                gps_speed: float,
                gps_heading_deg: float,
                prev_action: int,
                yolo: Optional[dict] = None,
                prev_frame_bgr: Optional[np.ndarray] = None) -> dict:
        """Returns {action_id, action_name, probs, confidence}."""
        if frame_bgr is None:
            return {'action_id': STOP, 'action_name': ACTION_NAMES[STOP],
                    'probs': None, 'confidence': 0.0, 'reason': 'no_frame'}

        img = preprocess_pair(frame_bgr, prev_frame_bgr)         # (1, 6, 224, 224)
        state = build_state_vector(sensors, gps_valid, gps_speed,
                                   gps_heading_deg, prev_action, yolo)
        state = state[np.newaxis, :].astype(np.float32)          # (1, state_dim)

        logits = self.backend.predict_logits(img, state)[0]
        probs = _softmax(logits)
        action_id = int(np.argmax(probs))
        return {
            'action_id': action_id,
            'action_name': ACTION_NAMES[action_id],
            'probs': probs.tolist(),
            'confidence': float(probs[action_id]),
        }
