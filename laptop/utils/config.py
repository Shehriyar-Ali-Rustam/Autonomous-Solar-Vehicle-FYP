#!/usr/bin/env python3
"""Loads the central YAML config, with safe defaults if PyYAML missing.

Usage:
    from utils.config import load_config
    cfg = load_config()                         # uses laptop/config.yaml
    cfg = load_config('path/to/other.yaml')     # override
    threshold = cfg.get('model', {}).get('conf_threshold', 0.55)
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict

_LOCK = threading.Lock()
_CACHE: Dict[str, Dict[str, Any]] = {}

_DEFAULTS: Dict[str, Any] = {
    'pi': {'default_ip': '192.168.1.100', 'port': 5555},
    'web': {'port': 8080, 'bind': '0.0.0.0'},
    'camera': {'device': 1, 'width': 640, 'height': 480, 'fps': 30},
    'recorder': {'sample_hz': 10, 'jpg_quality': 85, 'out_dir': 'data'},
    'model': {'conf_threshold': 0.55, 'loop_hz': 10, 'pulse_steer_ms': 200},
    'train': {
        'batch_size': 32, 'epochs_frozen': 10, 'epochs_unfrozen': 20,
        'lr_head': 1e-3, 'lr_backbone': 1e-4, 'weight_decay': 1e-4,
        'early_stop_patience': 6, 'seed': 42,
        'test_ratio': 0.15, 'val_ratio': 0.15,
    },
    'sensors': {'max_distance_cm': 400.0, 'min_distance_cm': 2.0, 'max_speed_mps': 5.0},
    'pedestrian': {'area_threshold': 0.06, 'any_position': True},
    'yolo': {'conf_threshold': 0.4, 'staleness_seconds': 1.0},
    'logging': {'level': 'INFO', 'dir': 'logs', 'max_bytes': 5_000_000, 'backup_count': 5},
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins). Returns new dict."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str = None) -> Dict[str, Any]:
    """Load + cache config from YAML, merged onto defaults."""
    if path is None:
        # default: laptop/config.yaml relative to this file
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.normpath(os.path.join(here, '..', 'config.yaml'))

    with _LOCK:
        if path in _CACHE:
            return _CACHE[path]

        loaded: Dict[str, Any] = {}
        if os.path.exists(path):
            try:
                import yaml  # type: ignore
                with open(path, 'r') as f:
                    loaded = yaml.safe_load(f) or {}
            except ImportError:
                # PyYAML not installed — return defaults
                pass
            except Exception:
                # malformed yaml — return defaults
                pass

        merged = _deep_merge(_DEFAULTS, loaded)
        _CACHE[path] = merged
        return merged


def reset_cache() -> None:
    """Force re-read of config (for tests)."""
    with _LOCK:
        _CACHE.clear()
