"""Unit tests for hybrid sensor+YOLO decision logic."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.actions import (FORWARD, SLOW_DOWN, TURN_LEFT, TURN_RIGHT, STOP,
                        REVERSE_LEFT, REVERSE_RIGHT, REVERSE)
from autonomous_hybrid import decide


def _yolo(person=0, area=0.0):
    return {'person_detected': person, 'object_detected': 0,
            'nearest_area_ratio': area, 'nearest_position_x': 0.0,
            'num_objects': 1 if person else 0}


def _sensors(fl=200, fr=200, fw=200, bc=200, ls=200, rs=200):
    return {'FL': fl, 'FR': fr, 'FW': fw, 'BC': bc, 'LS': ls, 'RS': rs}


def test_clear_path_is_forward():
    action, reason = decide(_sensors(), _yolo())
    assert action == FORWARD


def test_close_person_stops_even_in_clear_path():
    action, reason = decide(_sensors(), _yolo(person=1, area=0.5))
    assert action == STOP
    assert 'PERSON_CLOSE' in reason


def test_distant_person_does_not_stop():
    action, reason = decide(_sensors(), _yolo(person=1, area=0.001))
    assert action == FORWARD


def test_obstacle_in_front_within_slow_zone_turns_to_clearer_side():
    # left clear (200), right wall (10) → should TURN_LEFT
    # fw=60 is below the front_slow_cm threshold (70 by default)
    action, reason = decide(_sensors(fw=60, ls=200, rs=10), _yolo())
    assert action == TURN_LEFT
    action, reason = decide(_sensors(fw=60, ls=10, rs=200), _yolo())
    assert action == TURN_RIGHT


def test_emergency_front_reverses_to_open_side():
    # Front blocked, back clear, left more open → should REVERSE_RIGHT (curve back-right)
    action, reason = decide(_sensors(fw=20, bc=200, ls=200, rs=10), _yolo())
    assert action == REVERSE_RIGHT


def test_pinned_front_and_back_blocked_is_stop():
    action, reason = decide(_sensors(fw=20, bc=10), _yolo())
    assert action == STOP
    assert 'PINNED' in reason


def test_narrow_corridor_just_slows_down():
    # Front getting close but no side clearance → SLOW_DOWN
    action, reason = decide(_sensors(fw=60, ls=10, rs=10), _yolo())
    assert action == SLOW_DOWN


def test_invalid_zero_sensors_treated_as_clear():
    """Zero or out-of-range readings are ignored; if no front data is valid,
    the function should still drive (fall-through to FORWARD)."""
    action, reason = decide(_sensors(fl=0, fr=0, fw=0), _yolo())
    assert action == FORWARD
