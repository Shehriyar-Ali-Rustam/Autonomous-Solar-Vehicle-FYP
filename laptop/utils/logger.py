#!/usr/bin/env python3
"""Centralised logger. Replaces ad-hoc print() calls.

Usage:
    from utils.logger import get_logger
    log = get_logger(__name__)
    log.info("hello")
    log.warning("careful")
    log.error("broken", exc_info=True)
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys

from utils.config import load_config

_CONFIGURED = False


def _setup_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    cfg = load_config().get('logging', {})
    level = getattr(logging, cfg.get('level', 'INFO').upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    # Clear existing handlers (e.g. from external imports)
    for h in list(root.handlers):
        root.removeHandler(h)

    # Console
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                            datefmt='%H:%M:%S')
    ch = logging.StreamHandler(sys.stderr)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # Rotating file
    log_dir = cfg.get('dir', 'logs')
    here = os.path.dirname(os.path.abspath(__file__))
    abs_dir = log_dir if os.path.isabs(log_dir) else os.path.normpath(os.path.join(here, '..', log_dir))
    try:
        os.makedirs(abs_dir, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            os.path.join(abs_dir, 'app.log'),
            maxBytes=int(cfg.get('max_bytes', 5_000_000)),
            backupCount=int(cfg.get('backup_count', 5)),
        )
        fh.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s [%(threadName)s]: %(message)s'))
        root.addHandler(fh)
    except OSError:
        # Read-only fs / etc — log to console only
        pass

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger; sets up root handlers on first call."""
    _setup_root()
    return logging.getLogger(name)
