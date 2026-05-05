"""Unit tests for ml/actions.py — labels & Pi command mapping must be airtight."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.actions import (
    FORWARD, SLOW_DOWN, TURN_LEFT, TURN_RIGHT, STOP,
    REVERSE_LEFT, REVERSE_RIGHT, REVERSE,
    NUM_ACTIONS, ACTION_NAMES,
    SLOW_DOWN_SPEED_THRESHOLD,
    manual_to_action, action_to_pi_command,
)


# ---------- Constants ----------------------------------------------
def test_action_count_matches_names():
    assert NUM_ACTIONS == len(ACTION_NAMES) == 8


def test_action_ids_unique_and_consecutive():
    ids = sorted([FORWARD, SLOW_DOWN, TURN_LEFT, TURN_RIGHT, STOP,
                  REVERSE_LEFT, REVERSE_RIGHT, REVERSE])
    assert ids == list(range(NUM_ACTIONS))


# ---------- manual_to_action --------------------------------------
def test_manual_stop_no_steer():
    assert manual_to_action('STOP', 'STEER_STOP') == STOP


def test_manual_stop_with_left_is_turn_left():
    """Intent capture: stopped + steer LEFT → TURN_LEFT (not STOP)."""
    assert manual_to_action('STOP', 'LEFT') == TURN_LEFT


def test_manual_stop_with_right_is_turn_right():
    assert manual_to_action('STOP', 'RIGHT') == TURN_RIGHT


def test_manual_forward_full_speed_is_forward():
    assert manual_to_action('FORWARD', 'STEER_STOP', speed=60) == FORWARD


def test_manual_forward_low_speed_is_slow_down():
    assert manual_to_action('FORWARD', 'STEER_STOP',
                            speed=SLOW_DOWN_SPEED_THRESHOLD - 1) == SLOW_DOWN


def test_manual_forward_left_is_turn_left():
    assert manual_to_action('FORWARD', 'LEFT') == TURN_LEFT


def test_manual_forward_right_is_turn_right():
    assert manual_to_action('FORWARD', 'RIGHT') == TURN_RIGHT


def test_manual_backward_no_steer_is_reverse():
    assert manual_to_action('BACKWARD', 'STEER_STOP') == REVERSE


def test_manual_backward_left_is_reverse_left():
    assert manual_to_action('BACKWARD', 'LEFT') == REVERSE_LEFT


def test_manual_backward_right_is_reverse_right():
    assert manual_to_action('BACKWARD', 'RIGHT') == REVERSE_RIGHT


# ---------- action_to_pi_command ----------------------------------
def test_pi_command_stop_has_zero_speed():
    cmd = action_to_pi_command(STOP)
    assert cmd['command'] == 'STOP'
    assert cmd['speed'] == 0


def test_pi_command_forward_uses_semantic_forward():
    """After the inversion fix, FORWARD action sends 'FORWARD' (not 'BACKWARD')."""
    cmd = action_to_pi_command(FORWARD)
    assert cmd['command'] == 'FORWARD'
    assert cmd['steer'] == 'STEER_STOP'
    assert cmd['speed'] > 0


def test_pi_command_reverse_uses_semantic_backward():
    cmd = action_to_pi_command(REVERSE)
    assert cmd['command'] == 'BACKWARD'
    assert cmd['steer'] == 'STEER_STOP'


def test_pi_command_turn_left_is_forward_plus_left():
    cmd = action_to_pi_command(TURN_LEFT)
    assert cmd['command'] == 'FORWARD'
    assert cmd['steer'] == 'LEFT'


def test_pi_command_reverse_left_is_backward_plus_left():
    cmd = action_to_pi_command(REVERSE_LEFT)
    assert cmd['command'] == 'BACKWARD'
    assert cmd['steer'] == 'LEFT'


def test_pi_command_invalid_id_falls_back_to_stop():
    cmd = action_to_pi_command(999)
    assert cmd['command'] == 'STOP'
