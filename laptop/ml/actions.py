#!/usr/bin/env python3
"""Action space for the decision model (single source of truth)."""

FORWARD = 0
SLOW_DOWN = 1
TURN_LEFT = 2
TURN_RIGHT = 3
STOP = 4
REVERSE_LEFT = 5
REVERSE_RIGHT = 6
REVERSE = 7

ACTION_NAMES = [
    "FORWARD", "SLOW_DOWN", "TURN_LEFT", "TURN_RIGHT",
    "STOP", "REVERSE_LEFT", "REVERSE_RIGHT", "REVERSE",
]
NUM_ACTIONS = len(ACTION_NAMES)


def action_to_pi_command(action_id: int) -> dict:
    """Map action id -> Pi TCP command dict."""
    m = {
        FORWARD:       {'command': 'FORWARD',  'steer': 'STEER_STOP', 'speed': 60},
        SLOW_DOWN:     {'command': 'FORWARD',  'steer': 'STEER_STOP', 'speed': 30},
        TURN_LEFT:     {'command': 'FORWARD',  'steer': 'LEFT',       'speed': 40},
        TURN_RIGHT:    {'command': 'FORWARD',  'steer': 'RIGHT',      'speed': 40},
        STOP:          {'command': 'STOP',     'steer': 'STEER_STOP', 'speed': 0},
        REVERSE_LEFT:  {'command': 'BACKWARD', 'steer': 'LEFT',       'speed': 35},
        REVERSE_RIGHT: {'command': 'BACKWARD', 'steer': 'RIGHT',      'speed': 35},
        REVERSE:       {'command': 'BACKWARD', 'steer': 'STEER_STOP', 'speed': 35},
    }
    return m.get(action_id, m[STOP])


def manual_to_action(drive: str, steer: str) -> int:
    """Convert manual WASD drive+steer -> action id (used when labeling training data)."""
    if drive == 'STOP':
        return STOP
    if drive == 'FORWARD':
        if steer == 'LEFT':
            return TURN_LEFT
        if steer == 'RIGHT':
            return TURN_RIGHT
        return FORWARD
    if drive == 'BACKWARD':
        if steer == 'LEFT':
            return REVERSE_LEFT
        if steer == 'RIGHT':
            return REVERSE_RIGHT
        return REVERSE
    return STOP
