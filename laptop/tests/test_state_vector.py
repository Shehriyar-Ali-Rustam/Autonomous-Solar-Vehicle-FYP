"""Tests for state-vector construction in ml/dataset.py."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.actions import STOP, FORWARD, NUM_ACTIONS
from ml.dataset import build_state_vector
from ml.model import STATE_DIM


SAFE_SENSORS = {'FL': 200, 'FR': 200, 'FW': 200,
                'BC': 100, 'LS': 100, 'RS': 100}


def _v(sensors=SAFE_SENSORS, gps_valid=0, gps_speed=0.0, gps_heading=0.0,
       prev=STOP, yolo=None):
    return build_state_vector(sensors, gps_valid, gps_speed, gps_heading, prev, yolo)


def test_state_vector_length_matches_model_state_dim():
    v = _v()
    assert v.shape == (STATE_DIM,), f"got {v.shape}"


def test_state_vector_has_no_nans():
    v = _v(yolo={'person_detected': 1, 'object_detected': 1,
                 'nearest_area_ratio': 0.5, 'nearest_position_x': 0.7})
    assert not np.isnan(v).any()


def test_invalid_gps_zeros_out_speed_and_heading():
    """GPS gating: when gps_valid=0, speed/heading contributions must be zero."""
    invalid = _v(gps_valid=0, gps_speed=4.0, gps_heading=180.0)
    valid   = _v(gps_valid=1, gps_speed=4.0, gps_heading=180.0)
    # GPS occupies 4 dims starting at offset 8 (after 6 sensors + 2 mins)
    assert np.allclose(invalid[8:12], [0.0, 0.0, 0.0, 0.0])
    # Valid GPS produces non-zero values
    assert not np.allclose(valid[8:12], [0.0, 0.0, 0.0, 0.0])


def test_yolo_position_x_clamped_to_unit_range():
    v_low  = _v(yolo={'nearest_position_x': -5.0})
    v_high = _v(yolo={'nearest_position_x': +5.0})
    # Last YOLO field is at offset 8 (gps) + 4 (gps dims) + 3 (yolo before pos_x) = 15
    # Layout: [6 sensors, 2 mins, 4 gps, 4 yolo (person, obj, area, pos_x), 8 prev]
    assert v_low[15] == -1.0
    assert v_high[15] == +1.0


def test_prev_action_one_hot_correct():
    v = _v(prev=FORWARD)
    onehot = v[16:16 + NUM_ACTIONS]
    assert int(onehot.argmax()) == FORWARD
    assert onehot.sum() == pytest.approx(1.0)


def test_distances_normalised_to_unit():
    sensors = {'FL': 400, 'FR': 0, 'FW': 200,
               'BC': 0, 'LS': 0, 'RS': 0}
    v = _v(sensors=sensors)
    # distances are first 6 entries, normalised to 0..1
    assert v[0] == pytest.approx(1.0)        # 400 / 400
    assert v[1] == pytest.approx(0.0)
    assert v[2] == pytest.approx(0.5)
    assert (v[:6] >= 0.0).all() and (v[:6] <= 1.0).all()
