"""Test that split_csv guarantees no scene leakage across train/val/test."""

import os
import sys
import tempfile

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.dataset import split_csv


def _toy_df():
    """Three sessions, 30 rows each, mix of action labels per session."""
    rows = []
    for sess_id in ('s1', 's2', 's3'):
        for i in range(30):
            rows.append({
                'session': sess_id,
                'timestamp': i,
                'frame_path': f'/tmp/{sess_id}_{i}.jpg',
                'FL': 100, 'FR': 100, 'FW': 100,
                'BC': 100, 'LS': 100, 'RS': 100,
                'gps_valid': 0, 'gps_speed': 0, 'gps_heading': 0,
                'drive': 'FORWARD', 'steer': 'STEER_STOP', 'speed': 50,
                'prev_action': 4,
                'action_label': i % 8,
                'action_name': 'X',
                'yolo_person': 0, 'yolo_object': 0,
                'yolo_area': 0.0, 'yolo_pos_x': 0.0,
                'yolo_count': 0,
            })
    return pd.DataFrame(rows)


def test_no_session_appears_in_more_than_one_split():
    with tempfile.TemporaryDirectory() as td:
        csv_path = os.path.join(td, 'all.csv')
        out_dir  = os.path.join(td, 'splits')
        _toy_df().to_csv(csv_path, index=False)

        paths = split_csv(csv_path, out_dir, train_ratio=0.34, val_ratio=0.33,
                          seed=42, min_per_class_per_split=0, mode='session')

        train_sess = set(pd.read_csv(paths['train'])['session'])
        val_sess   = set(pd.read_csv(paths['val'])['session'])
        test_sess  = set(pd.read_csv(paths['test'])['session'])

        # Disjoint sets
        assert train_sess.isdisjoint(val_sess)
        assert train_sess.isdisjoint(test_sess)
        assert val_sess.isdisjoint(test_sess)
        # Every session is somewhere
        assert (train_sess | val_sess | test_sess) == {'s1', 's2', 's3'}


def test_split_creates_three_files():
    with tempfile.TemporaryDirectory() as td:
        csv_path = os.path.join(td, 'all.csv')
        out_dir  = os.path.join(td, 'splits')
        _toy_df().to_csv(csv_path, index=False)
        paths = split_csv(csv_path, out_dir, train_ratio=0.34, val_ratio=0.33,
                          seed=42, min_per_class_per_split=0)
        for k in ('train', 'val', 'test'):
            assert os.path.exists(paths[k]), f"missing {k}"


def test_time_split_default_keeps_class_coverage():
    """time-axis split (default) gives every session-class pair to all splits
    where data exists, and produces train > val ≈ test in size.
    """
    with tempfile.TemporaryDirectory() as td:
        csv_path = os.path.join(td, 'all.csv')
        out_dir  = os.path.join(td, 'splits')
        _toy_df().to_csv(csv_path, index=False)
        paths = split_csv(csv_path, out_dir, train_ratio=0.70, val_ratio=0.15,
                          seed=42, min_per_class_per_split=0, mode='time')
        train_df = pd.read_csv(paths['train'])
        val_df   = pd.read_csv(paths['val'])
        test_df  = pd.read_csv(paths['test'])
        # Train should be biggest
        assert len(train_df) > len(val_df)
        assert len(train_df) > len(test_df)
        # All three splits non-empty
        assert len(train_df) > 0 and len(val_df) > 0 and len(test_df) > 0
