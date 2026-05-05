#!/usr/bin/env python3
"""Training pipeline for the autonomous driving decision model.

Proper ML practice:
    * Per-session, stratified train / val / test split (no scene leakage).
    * WeightedRandomSampler oversampling for rare classes.
    * AdamW + CosineAnnealing LR.
    * Class-weighted CrossEntropy with label smoothing (defence in depth on top of oversampling).
    * Two-stage training (frozen backbone → fine-tune).
    * Model versioning: timestamped subdirectory, never overwrite previous runs.
    * Experiment tracking: writes train_config.json, metrics.json, splits.json
      into the version dir + TensorBoard.

Usage:
    python -m ml.train --csv data/dataset.csv --out models
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from utils.logger import get_logger
from utils.config import load_config
from .actions import NUM_ACTIONS, ACTION_NAMES
from .dataset import (DrivingDataset, class_weights, sample_weights_for_oversampling,
                      split_csv)
from .model import DecisionModel

log = get_logger(__name__)


@dataclass
class TrainConfig:
    csv_path: str
    out_dir: str
    batch_size: int = 32
    num_workers: int = 2
    epochs_frozen: int = 10
    epochs_unfrozen: int = 20
    lr_head: float = 1e-3
    lr_backbone: float = 1e-4
    weight_decay: float = 1e-4
    early_stop_patience: int = 6
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 42
    test_ratio: float = 0.15
    val_ratio: float = 0.15
    oversample: bool = True
    split_mode: str = 'time'   # 'time' | 'session'


def set_seed(seed: int) -> None:
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total = correct = 0
    loss_sum = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            img   = batch['image'].to(device, non_blocking=True)
            state = batch['state'].to(device, non_blocking=True)
            y     = batch['label'].to(device, non_blocking=True)

            logits = model(img, state)
            loss = criterion(logits, y)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            bs = y.size(0)
            loss_sum += loss.item() * bs
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += bs
    return loss_sum / max(total, 1), correct / max(total, 1)


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)

    # Versioned output directory: never overwrite previous runs.
    run_id = time.strftime('v_%Y%m%d_%H%M%S')
    run_dir = os.path.join(cfg.out_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    log.info(f"Device: {cfg.device}")
    log.info(f"Run directory: {run_dir}")

    writer = SummaryWriter(os.path.join(run_dir, 'tb'))

    # 1) Per-session stratified split → versioned splits dir
    splits_dir = os.path.join(run_dir, 'splits')
    paths = split_csv(cfg.csv_path, splits_dir, seed=cfg.seed,
                      train_ratio=1.0 - cfg.test_ratio - cfg.val_ratio,
                      val_ratio=cfg.val_ratio,
                      mode=cfg.split_mode)

    train_ds = DrivingDataset(paths['train'], train=True)
    val_ds   = DrivingDataset(paths['val'],   train=False)
    test_ds  = DrivingDataset(paths['test'],  train=False)

    # 2) Oversampling sampler (or shuffle=True fallback)
    train_labels = [s.label for s in train_ds.samples]
    if cfg.oversample and len(train_labels) > 0:
        weights = sample_weights_for_oversampling(train_labels, NUM_ACTIONS)
        sampler = WeightedRandomSampler(weights=weights.tolist(),
                                        num_samples=len(weights),
                                        replacement=True)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                                  sampler=sampler, num_workers=cfg.num_workers,
                                  pin_memory=True)
        log.info("Using WeightedRandomSampler oversampling for class balance")
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                                  num_workers=cfg.num_workers, pin_memory=True)

    val_loader  = DataLoader(val_ds,  batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)

    # 3) Class-weighted loss as a second line of defence
    cw = class_weights(train_labels, NUM_ACTIONS).to(cfg.device)
    log.info("Class weights:")
    for i, n in enumerate(ACTION_NAMES):
        log.info(f"  {n:15s}: {cw[i].item():.3f}")
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)

    # 4) Model
    model = DecisionModel(pretrained_backbone=True, freeze_backbone=True).to(cfg.device)

    # 5) Phase 1 — frozen backbone (head-only)
    log.info(f"=== Phase 1: train head only ({cfg.epochs_frozen} epochs) ===")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg.lr_head, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs_frozen)

    best_val = float('inf')
    patience = 0
    best_path = os.path.join(run_dir, 'best.pt')
    last_path = os.path.join(run_dir, 'last.pt')

    def _step(epoch: int, phase: str) -> None:
        nonlocal best_val, patience
        tl, ta = run_epoch(model, train_loader, criterion, opt, cfg.device, train=True)
        vl, va = run_epoch(model, val_loader,   criterion, opt, cfg.device, train=False)
        sched.step()

        for name, val in (('train_loss', tl), ('train_acc', ta),
                          ('val_loss', vl), ('val_acc', va),
                          ('lr', opt.param_groups[0]['lr'])):
            writer.add_scalar(f'{phase}/{name}', val, epoch)

        log.info(f"[{phase}] epoch {epoch:3d} | "
                 f"train_loss={tl:.4f} acc={ta*100:.2f}% | "
                 f"val_loss={vl:.4f} acc={va*100:.2f}% | "
                 f"lr={opt.param_groups[0]['lr']:.2e}")

        # Always save last for resumability
        torch.save({'model': model.state_dict(), 'phase': phase,
                    'epoch': epoch, 'val_loss': vl, 'val_acc': va}, last_path)

        if vl < best_val - 1e-4:
            best_val = vl
            patience = 0
            torch.save({'model': model.state_dict(), 'phase': phase,
                        'epoch': epoch, 'val_loss': vl, 'val_acc': va}, best_path)
            log.info(f"  * saved best to {best_path}")
        else:
            patience += 1

    for e in range(1, cfg.epochs_frozen + 1):
        _step(e, 'phase1')
        if patience >= cfg.early_stop_patience:
            log.info("Early stop in phase 1"); break

    # 6) Phase 2 — fine-tune backbone
    log.info(f"=== Phase 2: fine-tune backbone ({cfg.epochs_unfrozen} epochs) ===")
    model.unfreeze_backbone()
    opt = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': cfg.lr_backbone},
        {'params': model.head.parameters(),     'lr': cfg.lr_head * 0.5},
    ], weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs_unfrozen)
    patience = 0

    for e in range(1, cfg.epochs_unfrozen + 1):
        _step(e, 'phase2')
        if patience >= cfg.early_stop_patience:
            log.info("Early stop in phase 2"); break

    # 7) Evaluate best on test set
    ckpt = torch.load(best_path, map_location=cfg.device)
    model.load_state_dict(ckpt['model'])
    tl, ta = run_epoch(model, test_loader, criterion, opt, cfg.device, train=False)
    log.info("=== TEST RESULTS ===")
    log.info(f"Test loss: {tl:.4f}  accuracy: {ta*100:.2f}%")
    writer.add_scalar('test/loss', tl, 0)
    writer.add_scalar('test/acc',  ta, 0)

    # 8) Save run metadata + create/update 'latest' symlink
    metadata = {
        'config': asdict(cfg),
        'run_id': run_id,
        'splits': paths,
        'test_loss': tl, 'test_acc': ta,
        'best_val_loss': best_val,
        'platform': platform.platform(),
        'python': sys.version,
        'torch': torch.__version__,
        'finished_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(run_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    # 'latest' symlink for convenience (autonomous mode can target it)
    latest = os.path.join(cfg.out_dir, 'latest')
    try:
        if os.path.islink(latest) or os.path.exists(latest):
            os.remove(latest)
        os.symlink(run_id, latest)
    except OSError:
        # Symlink may fail on some FS; not fatal
        pass

    writer.close()
    log.info(f"Best model: {best_path}")
    log.info(f"Run dir:    {run_dir}")
    log.info(f"Latest:     {latest}")


def main() -> None:
    cfg_yaml = load_config().get('train', {})

    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True, help='Path to dataset CSV')
    p.add_argument('--out', default='models', help='Output directory (run_id appended)')
    p.add_argument('--batch-size', type=int, default=cfg_yaml.get('batch_size', 32))
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--epochs-frozen',   type=int, default=cfg_yaml.get('epochs_frozen', 10))
    p.add_argument('--epochs-unfrozen', type=int, default=cfg_yaml.get('epochs_unfrozen', 20))
    p.add_argument('--no-oversample', action='store_true',
                   help='Disable WeightedRandomSampler oversampling')
    p.add_argument('--split-mode', choices=['time', 'session'], default='time',
                   help='time = split each session along time axis (default); '
                        'session = whole sessions per bucket (use only with >=5 sessions)')
    args = p.parse_args()

    cfg = TrainConfig(
        csv_path=args.csv,
        out_dir=args.out,
        batch_size=args.batch_size,
        num_workers=args.workers,
        epochs_frozen=args.epochs_frozen,
        epochs_unfrozen=args.epochs_unfrozen,
        lr_head=cfg_yaml.get('lr_head', 1e-3),
        lr_backbone=cfg_yaml.get('lr_backbone', 1e-4),
        weight_decay=cfg_yaml.get('weight_decay', 1e-4),
        early_stop_patience=cfg_yaml.get('early_stop_patience', 6),
        seed=cfg_yaml.get('seed', 42),
        test_ratio=cfg_yaml.get('test_ratio', 0.15),
        val_ratio=cfg_yaml.get('val_ratio', 0.15),
        oversample=not args.no_oversample,
        split_mode=args.split_mode,
    )
    train(cfg)


if __name__ == "__main__":
    main()
