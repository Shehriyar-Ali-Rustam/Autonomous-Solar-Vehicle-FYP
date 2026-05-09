#!/usr/bin/env python3
"""Action space for the decision model (single source of truth).

After the Pi-side fix, FORWARD/BACKWARD are SEMANTIC commands referring to
the *physical* direction the car will move. The Pi's motor_controller honours
config.MOTOR_INVERTED to handle wiring inversion at the hardware boundary.
The laptop never compensates for wiring.
"""

# ===== Action IDs ============================================================
FORWARD       = 0
SLOW_DOWN     = 1
TURN_LEFT     = 2
TURN_RIGHT    = 3
STOP          = 4
REVERSE_LEFT  = 5
REVERSE_RIGHT = 6
REVERSE       = 7

ACTION_NAMES = [
    "FORWARD", "SLOW_DOWN", "TURN_LEFT", "TURN_RIGHT",
    "STOP", "REVERSE_LEFT", "REVERSE_RIGHT", "REVERSE",
]
NUM_ACTIONS = len(ACTION_NAMES)


# ===== Action → Pi command (intent) ==========================================
# Per-action speed defaults (only used by autonomous mode).
# NOTE: minimum motor PWM to actually move this car is ~40 (lower stalls).
_ACTION_SPEEDS = {
    FORWARD:       70,
    SLOW_DOWN:     45,
    TURN_LEFT:     50,
    TURN_RIGHT:    50,
    STOP:          0,
    REVERSE_LEFT:  45,
    REVERSE_RIGHT: 45,
    REVERSE:       45,
}

# Per-action drive + steer (semantic, no wiring compensation).
_ACTION_DRIVE_STEER = {
    FORWARD:       ('FORWARD',  'STEER_STOP'),
    SLOW_DOWN:     ('FORWARD',  'STEER_STOP'),
    TURN_LEFT:     ('FORWARD',  'LEFT'),
    TURN_RIGHT:    ('FORWARD',  'RIGHT'),
    STOP:          ('STOP',     'STEER_STOP'),
    REVERSE_LEFT:  ('BACKWARD', 'LEFT'),
    REVERSE_RIGHT: ('BACKWARD', 'RIGHT'),
    REVERSE:       ('BACKWARD', 'STEER_STOP'),
}


def action_to_pi_command(action_id: int) -> dict:
    """Map action id -> Pi TCP command dict."""
    if action_id not in _ACTION_DRIVE_STEER:
        action_id = STOP
    drive, steer = _ACTION_DRIVE_STEER[action_id]
    return {'command': drive, 'steer': steer, 'speed': _ACTION_SPEEDS[action_id]}


# ===== Manual driving → action label =========================================
# Speed below this threshold while moving forward → SLOW_DOWN label.
SLOW_DOWN_SPEED_THRESHOLD = 35


def _physical_direction(drive: str) -> str:
    """Returns 'forward' | 'backward' | 'stop' for a semantic drive command."""
    if drive == 'FORWARD':
        return 'forward'
    if drive == 'BACKWARD':
        return 'backward'
    return 'stop'


def manual_to_action(drive: str, steer: str, speed: int = 50) -> int:
    """Convert manual drive+steer+speed → action id for training labels.

    Intent-based:
      - STOP + steer LEFT/RIGHT → TURN_LEFT/TURN_RIGHT (user's turn intent)
      - Forward at low speed → SLOW_DOWN
    """
    direction = _physical_direction(drive)

    # User pressed STOP but turned the wheel: capture that turn intent.
    if direction == 'stop':
        if steer == 'LEFT':
            return TURN_LEFT
        if steer == 'RIGHT':
            return TURN_RIGHT
        return STOP

    if direction == 'forward':
        if steer == 'LEFT':
            return TURN_LEFT
        if steer == 'RIGHT':
            return TURN_RIGHT
        if 0 < speed < SLOW_DOWN_SPEED_THRESHOLD:
            return SLOW_DOWN
        return FORWARD

    # backward
    if steer == 'LEFT':
        return REVERSE_LEFT
    if steer == 'RIGHT':
        return REVERSE_RIGHT
    return REVERSE
