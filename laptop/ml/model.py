#!/usr/bin/env python3
"""Decision Model Architecture.

Two-stream:
    Stream A (vision): MobileNetV3-Small backbone, accepts 6 input channels
        (current frame + previous frame stacked) for temporal context.
    Stream B (state): sensors + (optional) GPS + YOLO + prev-action one-hot.

State vector layout (default 24 dims):
    6  ultrasonic distances        (FL, FR, FW, BC, LS, RS, normalised 0..1)
    2  derived mins                (front_min, back_min)
    1  GPS valid flag              (0 or 1; multiplied with the next three)
    1  GPS speed (normalised)      (0..1) — gated by gps_valid
    2  GPS heading (sin, cos)      (-1..1) — gated by gps_valid
    1  YOLO person_detected        (0 or 1)
    1  YOLO object_detected        (0 or 1)
    1  YOLO nearest_area_ratio     (0..1)
    1  YOLO nearest_position_x     (-1..1, continuous bbox-centre x)
    8  prev_action one-hot         (NUM_ACTIONS)
   ───
   24 total

Compared to v1: replaced the single 0/1/2 YOLO position with a continuous
value, replaced 1D heading angle with [sin, cos], and gated GPS by validity
so the model learns to ignore noise when no fix is available.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm

from .actions import NUM_ACTIONS

# ---------------- Geometry constants -----------------------
SENSOR_DIM       = 6      # FL, FR, FW, BC, LS, RS
DERIVED_DIM      = 2      # front_min, back_min
GPS_DIM          = 4      # valid, speed, sin(heading), cos(heading)
YOLO_DIM         = 4      # person_detected, object_detected, area, pos_x
PREV_ACTION_DIM  = NUM_ACTIONS

STATE_DIM = SENSOR_DIM + DERIVED_DIM + GPS_DIM + YOLO_DIM + PREV_ACTION_DIM  # 24

# Visual stream: 6 channels (3 RGB current + 3 RGB previous)
IMAGE_CHANNELS = 6


def _make_backbone(pretrained: bool, in_channels: int):
    """MobileNetV3-Small with custom in_channels (6 for frame-stacking).

    Pretrained weights are 3-channel; we tile them across the extra input
    channels so initial response is sensible.
    """
    weights = tvm.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    net = tvm.mobilenet_v3_small(weights=weights)
    feat_dim = net.classifier[0].in_features  # 576

    if in_channels != 3:
        old_conv = net.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )
        with torch.no_grad():
            if pretrained:
                rep = (in_channels + 2) // 3
                tiled = old_conv.weight.repeat(1, rep, 1, 1)[:, :in_channels]
                new_conv.weight.copy_(tiled / rep)
                if old_conv.bias is not None and new_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)
        net.features[0][0] = new_conv

    net.classifier = nn.Identity()
    return net, feat_dim


class DecisionModel(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM,
                 num_actions: int = NUM_ACTIONS,
                 in_channels: int = IMAGE_CHANNELS,
                 pretrained_backbone: bool = True,
                 freeze_backbone: bool = True,
                 dropout: float = 0.3):
        super().__init__()
        self.in_channels = in_channels
        self.backbone, vision_feat_dim = _make_backbone(pretrained_backbone, in_channels)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        fused_dim = vision_feat_dim + state_dim
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(max(0.0, dropout - 0.1)),
            nn.Linear(128, num_actions),
        )
        self.state_dim = state_dim
        self.num_actions = num_actions

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """image: (B, in_channels, 224, 224)  state: (B, state_dim)  → logits (B, num_actions)."""
        v = self.backbone(image)
        x = torch.cat([v, state], dim=1)
        return self.head(x)

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True


if __name__ == "__main__":
    m = DecisionModel()
    img = torch.randn(2, IMAGE_CHANNELS, 224, 224)
    st = torch.randn(2, STATE_DIM)
    out = m(img, st)
    print(f"Output shape: {out.shape}")
    print(f"STATE_DIM={STATE_DIM}, IMAGE_CHANNELS={IMAGE_CHANNELS}")
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Total params: {total:,}  Trainable: {trainable:,}")
