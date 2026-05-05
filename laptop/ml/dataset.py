#!/usr/bin/env python3
"""Dataset, state-vector builder, and split utilities.

Highlights:
  * Per-session train/val/test split (no scene leakage between splits).
  * Class-stratified within sessions for the train/val side; rare classes
    are guaranteed to appear in val and test by promoting samples when needed.
  * Frame pair loading (current + previous from the same session).
  * Centralised image preprocessing (vision/transforms.py).
  * Gated GPS: features multiplied by gps_valid so invalid GPS contributes zero.
  * Continuous YOLO position (-1..1, bbox centre x).
"""

from __future__ import annotations

import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from vision.transforms import build_eval_transform, build_train_transform
from .actions import (FORWARD, SLOW_DOWN, TURN_LEFT, TURN_RIGHT, STOP,
                      REVERSE_LEFT, REVERSE_RIGHT, REVERSE, NUM_ACTIONS)


# ---------------- Normalisation constants ----------------
MAX_DISTANCE_CM = 400.0
MAX_SPEED_MPS = 5.0


# ---------------- Flip augmentation maps ----------------
FLIP_ACTION_MAP = {
    FORWARD:       FORWARD,
    SLOW_DOWN:     SLOW_DOWN,
    TURN_LEFT:     TURN_RIGHT,
    TURN_RIGHT:    TURN_LEFT,
    STOP:          STOP,
    REVERSE_LEFT:  REVERSE_RIGHT,
    REVERSE_RIGHT: REVERSE_LEFT,
    REVERSE:       REVERSE,
}


def flip_sensors(sensors: dict) -> dict:
    flipped = dict(sensors)
    flipped['FL'], flipped['FR'] = sensors['FR'], sensors['FL']
    flipped['LS'], flipped['RS'] = sensors['RS'], sensors['LS']
    return flipped


# ---------------- State vector ----------------
def build_state_vector(sensors: dict, gps_valid: int, gps_speed: float,
                       gps_heading_deg: float, prev_action: int,
                       yolo: Optional[dict] = None) -> np.ndarray:
    """Normalised state vector matching ml.model.STATE_DIM = 24.

    yolo dict keys:
        person_detected (0/1), object_detected (0/1),
        nearest_area_ratio (0..1), nearest_position_x (-1..1)
    """
    # Sensors (clamped, normalised)
    raw = [sensors['FL'], sensors['FR'], sensors['FW'],
           sensors['BC'], sensors['LS'], sensors['RS']]
    u = [max(0.0, min(MAX_DISTANCE_CM, v)) / MAX_DISTANCE_CM for v in raw]

    # Front/back min (only valid sensors)
    valid_front = [v for v in [sensors['FL'], sensors['FR'], sensors['FW']]
                   if 2 <= v <= MAX_DISTANCE_CM]
    valid_back  = [v for v in [sensors['BC']]
                   if 2 <= v <= MAX_DISTANCE_CM]
    front_min = (min(valid_front) / MAX_DISTANCE_CM) if valid_front else 1.0
    back_min  = (min(valid_back) / MAX_DISTANCE_CM) if valid_back else 1.0

    # GPS — gated by validity flag so invalid GPS contributes zeros.
    valid = 1.0 if int(gps_valid) > 0 else 0.0
    speed_n = max(0.0, min(MAX_SPEED_MPS, gps_speed)) / MAX_SPEED_MPS * valid
    heading = math.radians(gps_heading_deg % 360.0)
    sin_h = math.sin(heading) * valid
    cos_h = math.cos(heading) * valid

    # YOLO (with safe defaults)
    if yolo is None:
        yolo = {}
    yolo_person  = float(yolo.get('person_detected', 0))
    yolo_obj     = float(yolo.get('object_detected', 0))
    yolo_area    = float(yolo.get('nearest_area_ratio', 0.0))
    yolo_pos_x   = float(yolo.get('nearest_position_x', 0.0))  # -1..1
    yolo_pos_x   = max(-1.0, min(1.0, yolo_pos_x))

    # Prev action one-hot
    prev_oh = np.zeros(NUM_ACTIONS, dtype=np.float32)
    if 0 <= prev_action < NUM_ACTIONS:
        prev_oh[prev_action] = 1.0

    head = np.array(u + [front_min, back_min,
                         valid, speed_n, sin_h, cos_h,
                         yolo_person, yolo_obj, yolo_area, yolo_pos_x],
                    dtype=np.float32)
    return np.concatenate([head, prev_oh])


# ---------------- Sample dataclass ----------------
@dataclass
class Sample:
    image_path: str
    prev_image_path: Optional[str]   # frame from same session, ~100ms earlier
    sensors: dict
    gps_valid: int
    gps_speed: float
    gps_heading: float
    prev_action: int
    label: int
    yolo: dict
    session: str


# ---------------- Dataset ----------------
class DrivingDataset(Dataset):
    """Supervised dataset: (frame_pair, state) → action label."""

    def __init__(self, csv_path: str, train: bool = True,
                 sensor_noise_std: float = 0.02, flip_prob: float = 0.5):
        self.csv_path = csv_path
        self.train = train
        self.sensor_noise_std = sensor_noise_std
        self.flip_prob = flip_prob
        self.transform = build_train_transform() if train else build_eval_transform()
        self.samples: List[Sample] = self._load(csv_path)

    @staticmethod
    def _yolo_pos_x(row) -> float:
        """Convert YOLO position to continuous (-1..1).

        Backwards-compatible: old data uses yolo_pos in {0, 1, 2}; new data
        will use yolo_pos_x in [-1, 1]. We support both.
        """
        if 'yolo_pos_x' in row and pd.notna(row.get('yolo_pos_x')):
            return float(row['yolo_pos_x'])
        legacy = row.get('yolo_pos', 1)
        try:
            legacy = int(legacy)
        except (TypeError, ValueError):
            legacy = 1
        # 0=left → -1, 1=center → 0, 2=right → +1
        return float(legacy) - 1.0

    @classmethod
    def _load(cls, csv_path: str) -> List[Sample]:
        df = pd.read_csv(csv_path)
        # Build per-session previous-frame index for frame stacking.
        prev_path: Dict[int, Optional[str]] = {}
        last_in_session: Dict[str, str] = {}
        for idx, row in df.iterrows():
            sess = str(row.get('session', ''))
            prev_path[idx] = last_in_session.get(sess)
            last_in_session[sess] = row['frame_path']

        samples = []
        for idx, r in df.iterrows():
            samples.append(Sample(
                image_path=r['frame_path'],
                prev_image_path=prev_path.get(idx),
                sensors={'FL': float(r['FL']), 'FR': float(r['FR']),
                         'FW': float(r['FW']), 'BC': float(r['BC']),
                         'LS': float(r['LS']), 'RS': float(r['RS'])},
                gps_valid=int(r.get('gps_valid', 0)),
                gps_speed=float(r.get('gps_speed', 0.0)),
                gps_heading=float(r.get('gps_heading', 0.0)),
                prev_action=int(r.get('prev_action', STOP)),
                label=int(r['action_label']),
                yolo={
                    'person_detected':    int(r.get('yolo_person', 0)),
                    'object_detected':    int(r.get('yolo_object', 0)),
                    'nearest_area_ratio': float(r.get('yolo_area', 0.0)),
                    'nearest_position_x': cls._yolo_pos_x(r),
                },
                session=str(r.get('session', '')),
            ))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Optional[str], fallback: Image.Image) -> Image.Image:
        if path and os.path.exists(path):
            try:
                return Image.open(path).convert('RGB')
            except (OSError, IOError):
                return fallback
        return fallback

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        img = self._load_image(s.image_path, Image.new('RGB', (224, 224)))
        prev = self._load_image(s.prev_image_path, img)  # fallback to current

        sensors = s.sensors
        label = s.label
        prev_action = s.prev_action
        yolo = dict(s.yolo)

        # Horizontal flip augmentation (training only)
        if self.train and random.random() < self.flip_prob:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            prev = prev.transpose(Image.FLIP_LEFT_RIGHT)
            sensors = flip_sensors(sensors)
            label = FLIP_ACTION_MAP[label]
            prev_action = FLIP_ACTION_MAP[prev_action]
            yolo['nearest_position_x'] = -float(yolo.get('nearest_position_x', 0.0))

        img_t = self.transform(img)
        prev_t = self.transform(prev)
        stacked = torch.cat([img_t, prev_t], dim=0)  # (6, 224, 224)

        state = build_state_vector(sensors, s.gps_valid, s.gps_speed,
                                   s.gps_heading, prev_action, yolo)
        if self.train and self.sensor_noise_std > 0:
            # Add noise only to the 8 distance-derived features
            noise = np.random.normal(0, self.sensor_noise_std, 8).astype(np.float32)
            state[:8] = np.clip(state[:8] + noise, 0.0, 1.0)

        return {
            'image': stacked,
            'state': torch.from_numpy(state).float(),
            'label': torch.tensor(label, dtype=torch.long),
        }


# ---------------- Class weights ----------------
def class_weights(labels: List[int], num_classes: int = NUM_ACTIONS) -> torch.Tensor:
    """Inverse-frequency class weights. Classes with 0 samples → weight 0."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    present = counts > 0
    w = np.zeros(num_classes, dtype=np.float32)
    if present.any():
        w[present] = 1.0 / counts[present]
        w = w / w[present].mean()
    return torch.from_numpy(w).float()


def sample_weights_for_oversampling(labels: List[int],
                                    num_classes: int = NUM_ACTIONS) -> np.ndarray:
    """Per-sample weights for use with torch.utils.data.WeightedRandomSampler.

    Each sample's weight = 1 / count(its class). After WeightedRandomSampler,
    minority classes are sampled more often, balancing each batch.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    inv = 1.0 / counts
    return np.array([inv[c] for c in labels], dtype=np.float64)


# ---------------- Splits ----------------
def split_csv(csv_path: str, out_dir: str,
              train_ratio: float = 0.70, val_ratio: float = 0.15,
              seed: int = 42, min_per_class_per_split: int = 2,
              mode: str = 'time') -> dict:
    """Split a dataset CSV.

    Modes:
      'time'    (default) — within each session, sort by timestamp and take
                first `train_ratio` as train, next `val_ratio` as val, rest
                as test. A small temporal buffer (1% of the session length on
                each side of a split boundary) is dropped to reduce
                near-frame leakage between splits. Recommended when you only
                have a handful of sessions; gives every class plenty of
                support in every split while still avoiding random shuffle
                leakage.
      'session' — entire sessions go into one bucket (no scene leakage at
                all but unbalanced if you have <5 sessions).

    Returns a dict {'train': path, 'val': path, 'test': path}.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)

    df = pd.read_csv(csv_path)
    if 'session' in df.columns and mode == 'time':
        return _split_time_within_sessions(df, out_dir, train_ratio,
                                           val_ratio, min_per_class_per_split)

    if 'session' not in df.columns:
        # Fall back to random split if no session column
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        n = len(df)
        n_tr, n_va = int(n * train_ratio), int(n * val_ratio)
        train_df, val_df, test_df = df[:n_tr], df[n_tr:n_tr + n_va], df[n_tr + n_va:]
    else:
        sessions = list(df['session'].unique())
        rng.shuffle(sessions)
        n_total = len(df)
        n_train_target = int(n_total * train_ratio)
        n_val_target   = int(n_total * val_ratio)

        # Greedy: assign whole sessions to train, then val, then test
        # while keeping ratios close to targets.
        sess_sizes = df.groupby('session').size().to_dict()
        train_sess: List[str] = []
        val_sess: List[str] = []
        test_sess: List[str] = []
        n_tr = n_va = n_te = 0
        for s in sessions:
            sz = sess_sizes[s]
            # pick the bucket whose deficit is largest (proportional to target)
            deficits = [
                (n_train_target - n_tr, 'train'),
                (n_val_target - n_va, 'val'),
                (max(0, n_total - n_train_target - n_val_target) - n_te, 'test'),
            ]
            deficits.sort(key=lambda x: x[0], reverse=True)
            target = deficits[0][1]
            if target == 'train':
                train_sess.append(s); n_tr += sz
            elif target == 'val':
                val_sess.append(s); n_va += sz
            else:
                test_sess.append(s); n_te += sz

        # Force at least one session into val and test if available
        if not val_sess and len(train_sess) > 1:
            moved = train_sess.pop()
            val_sess.append(moved); n_va += sess_sizes[moved]; n_tr -= sess_sizes[moved]
        if not test_sess and len(train_sess) > 1:
            moved = train_sess.pop()
            test_sess.append(moved); n_te += sess_sizes[moved]; n_tr -= sess_sizes[moved]

        train_df = df[df['session'].isin(train_sess)].reset_index(drop=True)
        val_df   = df[df['session'].isin(val_sess)].reset_index(drop=True)
        test_df  = df[df['session'].isin(test_sess)].reset_index(drop=True)

    # Stratification check: warn (and promote a handful of train rows) if
    # any class has < min_per_class_per_split in val/test. We promote
    # individual rows here as a *second-best* fix to preserve evaluation
    # signal on rare classes; per-session purity is preferred but we'd
    # otherwise have classes with zero test support.
    train_df, val_df  = _promote_rare_class_rows(train_df, val_df,
                                                 min_per_class_per_split)
    train_df, test_df = _promote_rare_class_rows(train_df, test_df,
                                                 min_per_class_per_split)

    paths = {
        'train': os.path.join(out_dir, 'train.csv'),
        'val':   os.path.join(out_dir, 'val.csv'),
        'test':  os.path.join(out_dir, 'test.csv'),
    }
    train_df.to_csv(paths['train'], index=False)
    val_df.to_csv(paths['val'], index=False)
    test_df.to_csv(paths['test'], index=False)
    print(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print("Sessions: train={}, val={}, test={}".format(
        train_df['session'].nunique() if 'session' in train_df else '?',
        val_df['session'].nunique()   if 'session' in val_df   else '?',
        test_df['session'].nunique()  if 'session' in test_df  else '?',
    ))
    return paths


def _split_time_within_sessions(df: pd.DataFrame, out_dir: str,
                                train_ratio: float, val_ratio: float,
                                min_per_class_per_split: int,
                                buffer_frac: float = 0.01) -> dict:
    """For each session, sort by timestamp and take first/middle/last slices.

    Adjacent frames are nearly identical (10Hz sampling), so we drop a small
    buffer (1% of session length) at each split boundary to reduce direct
    leakage between train and test.
    """
    train_parts, val_parts, test_parts = [], [], []
    for sess, sub in df.groupby('session'):
        sub = sub.sort_values('timestamp').reset_index(drop=True)
        n = len(sub)
        if n < 10:
            # too small to split meaningfully — put it all in train
            train_parts.append(sub); continue

        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        buf = max(1, int(n * buffer_frac))

        train_end = max(0, n_train - buf)
        val_start = min(n, n_train + buf)
        val_end   = min(n, n_train + n_val - buf)
        test_start = min(n, n_train + n_val + buf)

        train_parts.append(sub.iloc[:train_end])
        val_parts.append(sub.iloc[val_start:val_end])
        test_parts.append(sub.iloc[test_start:])

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame()
    val_df   = pd.concat(val_parts,   ignore_index=True) if val_parts   else pd.DataFrame()
    test_df  = pd.concat(test_parts,  ignore_index=True) if test_parts  else pd.DataFrame()

    train_df, val_df  = _promote_rare_class_rows(train_df, val_df,
                                                 min_per_class_per_split)
    train_df, test_df = _promote_rare_class_rows(train_df, test_df,
                                                 min_per_class_per_split)

    paths = {
        'train': os.path.join(out_dir, 'train.csv'),
        'val':   os.path.join(out_dir, 'val.csv'),
        'test':  os.path.join(out_dir, 'test.csv'),
    }
    train_df.to_csv(paths['train'], index=False)
    val_df.to_csv(paths['val'], index=False)
    test_df.to_csv(paths['test'], index=False)
    print(f"Split (time-axis): train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    if 'session' in train_df:
        print(f"Sessions appearing in all splits: {train_df['session'].nunique()}")
    return paths


def _promote_rare_class_rows(train_df: pd.DataFrame, target_df: pd.DataFrame,
                             min_per_class: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """If the target split has < `min_per_class` of any present class, pull
    that many rows from train for those classes only. Returns updated dfs.
    """
    if len(train_df) == 0 or len(target_df) == 0:
        return train_df, target_df
    counts_train = train_df['action_label'].value_counts().to_dict()
    counts_target = target_df['action_label'].value_counts().to_dict()
    moves: List[int] = []
    for cls, n_train in counts_train.items():
        n_target = counts_target.get(cls, 0)
        deficit = max(0, min_per_class - n_target)
        if deficit > 0 and n_train > deficit:
            cls_rows = train_df[train_df['action_label'] == cls].head(deficit).index.tolist()
            moves.extend(cls_rows)
    if moves:
        moved = train_df.loc[moves]
        train_df = train_df.drop(index=moves).reset_index(drop=True)
        target_df = pd.concat([target_df, moved], ignore_index=True)
    return train_df, target_df


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m laptop.ml.dataset <csv_path>")
        raise SystemExit(0)
    ds = DrivingDataset(sys.argv[1], train=True)
    print(f"Dataset size: {len(ds)}")
    s = ds[0]
    print(f"image: {s['image'].shape}  state: {s['state'].shape}  label: {s['label'].item()}")
