"""Tests for the laptop-side direction-aware vision hard-stop."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.actions import (FORWARD, STOP, TURN_LEFT, TURN_RIGHT,
                        REVERSE_LEFT, REVERSE_RIGHT)
import main_autonomous as ma


def _yolo(person=0, obj=0, area=0.0, pos_x=0.0):
    return {'person_detected': person, 'object_detected': obj,
            'nearest_area_ratio': area, 'nearest_position_x': pos_x,
            'num_objects': 1 if (person or obj) else 0}


def test_no_hardstop_when_intent_is_already_stop():
    reason = ma._hardstop_reason(STOP, _yolo(person=1, area=0.5, pos_x=0.0))
    assert reason == ''


def test_no_hardstop_when_no_object_detected():
    reason = ma._hardstop_reason(FORWARD, _yolo())
    assert reason == ''


def test_no_hardstop_for_distant_object():
    reason = ma._hardstop_reason(FORWARD, _yolo(obj=1, area=0.001, pos_x=0.0))
    assert reason == ''


def test_hardstop_on_close_person_in_front_centre():
    reason = ma._hardstop_reason(FORWARD, _yolo(person=1, area=0.5, pos_x=0.0))
    assert 'PERSON_CLOSE' in reason


def test_hardstop_label_says_obstacle_when_no_person():
    reason = ma._hardstop_reason(FORWARD, _yolo(obj=1, area=0.5, pos_x=0.0))
    assert 'OBSTACLE_CLOSE' in reason


def test_reverse_intent_also_triggers_hardstop():
    """User asked: hard-stop should work for reverse direction too."""
    from ml.actions import REVERSE
    reason = ma._hardstop_reason(REVERSE, _yolo(person=1, area=0.5, pos_x=0.0))
    assert 'PERSON_CLOSE' in reason


def test_turn_left_hardstops_for_obstacle_on_left():
    reason = ma._hardstop_reason(TURN_LEFT, _yolo(obj=1, area=0.5, pos_x=-0.5))
    assert 'OBSTACLE_CLOSE' in reason
