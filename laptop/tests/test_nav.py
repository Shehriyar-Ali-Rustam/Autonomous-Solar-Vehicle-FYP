"""Tests for nav.geo helpers and nav.controller state machine."""

import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nav.geo import haversine_m, bearing_deg, heading_error_deg
from nav.controller import NavController
from ml.actions import FORWARD, SLOW_DOWN, TURN_LEFT, TURN_RIGHT, STOP


# ============== geo helpers =================================================
def test_haversine_zero_distance():
    assert haversine_m(33.6, 73.0, 33.6, 73.0) < 0.01


def test_haversine_known_distance():
    # CUST Islamabad area (approx) — 1 deg lat ≈ 111 km
    d = haversine_m(33.6, 73.0, 33.61, 73.0)
    assert 1100 < d < 1115     # 0.01° lat ≈ 1.11 km


def test_bearing_north_is_zero():
    # Move directly north → bearing should be ~0
    b = bearing_deg(33.6, 73.0, 33.7, 73.0)
    assert abs(b) < 1.0 or abs(b - 360.0) < 1.0


def test_bearing_east_is_ninety():
    # Move east at the equator, ish
    b = bearing_deg(33.6, 73.0, 33.6, 73.1)
    assert abs(b - 90.0) < 1.0


def test_bearing_south_is_180():
    b = bearing_deg(33.6, 73.0, 33.5, 73.0)
    assert abs(b - 180.0) < 1.0


def test_bearing_west_is_270():
    b = bearing_deg(33.6, 73.0, 33.6, 72.9)
    assert abs(b - 270.0) < 1.0


def test_heading_error_no_turn_needed():
    assert abs(heading_error_deg(90.0, 90.0)) < 0.01


def test_heading_error_right_turn():
    # current 0 (north), target 90 (east) → +90° (turn right)
    assert abs(heading_error_deg(0.0, 90.0) - 90.0) < 0.01


def test_heading_error_left_turn():
    # current 90, target 0 → -90 (turn left)
    assert abs(heading_error_deg(90.0, 0.0) - (-90.0)) < 0.01


def test_heading_error_wraps_at_180():
    # current 350, target 10 → +20° (don't go all the way around)
    assert abs(heading_error_deg(350.0, 10.0) - 20.0) < 0.01


def test_heading_error_wraps_neg_180():
    # current 10, target 350 → -20°
    assert abs(heading_error_deg(10.0, 350.0) - (-20.0)) < 0.01


# ============== controller state machine ====================================
def test_idle_when_no_waypoints():
    nav = NavController()
    action, state, info = nav.tick((33.6, 73.0), 0.0)
    assert state == 'IDLE'
    assert action == STOP


def test_calibrating_when_waypoints_set():
    nav = NavController()
    nav.set_waypoints([(33.61, 73.0)])
    action, state, info = nav.tick((33.6, 73.0), None)
    assert state == 'CALIBRATING'
    assert action == SLOW_DOWN


def test_navigating_after_calibration_motion():
    nav = NavController()
    nav.set_waypoints([(33.61, 73.0)])    # roughly 1.1 km north
    # First tick — start calibration
    nav.tick((33.6, 73.0), None)
    # Second tick — moved 2 metres north → infer heading ~0
    nav.tick((33.6 + (2.0 / 111000), 73.0), None)
    assert nav.state == 'NAVIGATING'


def test_arrives_within_threshold():
    nav = NavController()
    target = (33.60005, 73.0)   # ~5.5m north
    nav.set_waypoints([target])
    # Pretend we're already at the target
    action, state, info = nav.tick(target, 0.0)
    assert state == 'ARRIVED'
    assert action == STOP


def test_stop_clears_state():
    nav = NavController()
    nav.set_waypoints([(33.61, 73.0)])
    nav.stop()
    assert nav.state == 'IDLE'
    assert nav.waypoints == []
