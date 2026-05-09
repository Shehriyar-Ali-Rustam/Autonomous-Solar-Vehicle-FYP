#!/usr/bin/env python3
"""GPS waypoint navigation controller.

State machine:
    IDLE     -- no destination, do nothing.
    CALIBRATING -- car is moving forward briefly to learn its heading
                   from successive GPS positions (because GPS heading
                   is unreliable when stopped).
    NAVIGATING -- driving toward the next waypoint along the path.
    ARRIVED   -- final waypoint reached. STOP.

Inputs (per tick):
    position: (lat, lon) — current GPS position, or None if no fix.
    heading:  float in [0, 360) — current compass heading, or None.

Outputs (per tick):
    nav_action_id: int — one of FORWARD, TURN_LEFT, TURN_RIGHT, STOP, etc.
    state: str — for the UI to display
    info:  dict — debug info (distance to target, bearing, etc.)

The action this controller picks is just an *intent*. The hybrid
sensor+YOLO controller can override it (e.g. STOP when person close).
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

from ml.actions import (FORWARD, SLOW_DOWN, TURN_LEFT, TURN_RIGHT, STOP)
from .geo import haversine_m, bearing_deg, heading_error_deg

# How close (in metres) to a waypoint counts as "reached".
WAYPOINT_REACHED_M = 5.0
# Distance below which we slow down for the FINAL waypoint.
FINAL_APPROACH_M = 8.0
# Heading error magnitudes that map to action choices.
TURN_THRESHOLD_DEG = 15.0     # within ±15° of target → drive forward
HARD_TURN_DEG = 45.0          # > 45° off → turn in place (slow forward + steer)
# Minimum distance the car must travel during calibration before we trust
# the inferred heading.
CALIB_MIN_TRAVEL_M = 1.5
CALIB_TIMEOUT_S = 8.0         # if we fail to move enough in 8s, give up


class NavController:
    """One-segment-at-a-time waypoint follower."""

    STATE_IDLE        = 'IDLE'
    STATE_CALIBRATING = 'CALIBRATING'
    STATE_NAVIGATING  = 'NAVIGATING'
    STATE_ARRIVED     = 'ARRIVED'

    def __init__(self):
        self._waypoints: List[Tuple[float, float]] = []
        self._index: int = 0
        self._state: str = self.STATE_IDLE
        # Heading we're using (either GPS reported or motion-inferred)
        self._inferred_heading: Optional[float] = None
        # For motion-inferred heading
        self._calib_start_pos: Optional[Tuple[float, float]] = None
        self._calib_start_t: float = 0.0

    # ---------------- Public API -------------------------
    def set_waypoints(self, waypoints: List[Tuple[float, float]]) -> None:
        """Set/replace path. Each waypoint is (lat, lon)."""
        self._waypoints = list(waypoints)
        self._index = 0
        self._inferred_heading = None
        if self._waypoints:
            self._state = self.STATE_CALIBRATING
            self._calib_start_pos = None
        else:
            self._state = self.STATE_IDLE

    def stop(self) -> None:
        """Abort navigation."""
        self._waypoints = []
        self._index = 0
        self._state = self.STATE_IDLE

    @property
    def state(self) -> str:
        return self._state

    @property
    def waypoints(self) -> List[Tuple[float, float]]:
        return list(self._waypoints)

    @property
    def current_target(self) -> Optional[Tuple[float, float]]:
        if self._index < len(self._waypoints):
            return self._waypoints[self._index]
        return None

    # ---------------- Tick -------------------------------
    def tick(self, position: Optional[Tuple[float, float]],
             heading_deg: Optional[float]) -> Tuple[int, str, dict]:
        """One control step. Returns (action_id, state, info)."""
        if self._state == self.STATE_IDLE:
            return STOP, self._state, {}

        target = self.current_target
        if target is None:
            self._state = self.STATE_ARRIVED
            return STOP, self._state, {'reason': 'all_waypoints_reached'}

        if position is None:
            return STOP, self._state, {'reason': 'no_gps_fix'}

        dist_m = haversine_m(position[0], position[1], target[0], target[1])
        target_bearing = bearing_deg(position[0], position[1], target[0], target[1])

        # Reached current waypoint?
        if dist_m < WAYPOINT_REACHED_M:
            self._index += 1
            if self._index >= len(self._waypoints):
                self._state = self.STATE_ARRIVED
                return STOP, self._state, {
                    'reason': 'final_arrived',
                    'index': self._index,
                    'dist_m': dist_m,
                }
            # Move to next waypoint, recalibrate heading
            self._state = self.STATE_CALIBRATING
            self._calib_start_pos = None
            return STOP, self._state, {
                'reason': 'waypoint_reached',
                'index': self._index,
            }

        # CALIBRATING: drive forward briefly to learn heading from motion
        if self._state == self.STATE_CALIBRATING:
            now = time.time()
            if self._calib_start_pos is None:
                self._calib_start_pos = position
                self._calib_start_t = now
                return SLOW_DOWN, self._state, {
                    'reason': 'calibrating_start',
                    'dist_m': dist_m,
                }
            # Have we moved enough?
            travelled = haversine_m(self._calib_start_pos[0],
                                    self._calib_start_pos[1],
                                    position[0], position[1])
            if travelled >= CALIB_MIN_TRAVEL_M:
                # Compute heading from motion
                self._inferred_heading = bearing_deg(
                    self._calib_start_pos[0], self._calib_start_pos[1],
                    position[0], position[1])
                self._state = self.STATE_NAVIGATING
                self._calib_start_pos = None
            elif now - self._calib_start_t > CALIB_TIMEOUT_S:
                # Couldn't calibrate. Give up and try with whatever heading
                # we get (or 0 if none).
                self._state = self.STATE_NAVIGATING
                self._calib_start_pos = None
            else:
                return SLOW_DOWN, self._state, {
                    'reason': 'calibrating_moving',
                    'travelled_m': travelled,
                    'dist_m': dist_m,
                }

        # NAVIGATING: pick action based on heading error
        # Prefer reported GPS heading if available and reasonable, else inferred.
        cur_heading = heading_deg if (heading_deg is not None) else self._inferred_heading
        if cur_heading is None:
            # Still nothing to go on. Slow forward — calibration will retry.
            return SLOW_DOWN, self._state, {
                'reason': 'no_heading_yet',
                'dist_m': dist_m,
                'target_bearing': target_bearing,
            }

        err = heading_error_deg(cur_heading, target_bearing)

        # Final approach: slow down even if heading is good
        if dist_m < FINAL_APPROACH_M and self._index == len(self._waypoints) - 1:
            base_action = SLOW_DOWN
        else:
            base_action = FORWARD

        info = {
            'dist_m': dist_m,
            'target_bearing': target_bearing,
            'cur_heading': cur_heading,
            'err_deg': err,
            'index': self._index,
            'total_waypoints': len(self._waypoints),
        }

        # Pick action by heading error magnitude
        if abs(err) < TURN_THRESHOLD_DEG:
            return base_action, self._state, info
        if abs(err) < HARD_TURN_DEG:
            # Gentle correction — keep moving forward but with steer
            return (TURN_LEFT if err < 0 else TURN_RIGHT), self._state, info
        # Hard turn needed — slow down and steer hard
        action = TURN_LEFT if err < 0 else TURN_RIGHT
        info['hard_turn'] = True
        return action, self._state, info
