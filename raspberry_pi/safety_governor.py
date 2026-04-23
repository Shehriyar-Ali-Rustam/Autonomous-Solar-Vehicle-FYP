#!/usr/bin/env python3
"""
Safety Governor - Hard Safety Layer (Layer 3)

This is the OUTERMOST safety layer running on the Pi.
It enforces hard safety rules that CANNOT be overridden by the laptop's ML.

Rules (in priority order):
1. Front < EMERGENCY threshold  -> STOP
2. Back < EMERGENCY threshold while reversing -> STOP
3. No command from laptop in WATCHDOG_TIMEOUT -> STOP
4. Sensor error / all sensors zero for > SENSOR_FAILURE_TIME -> STOP
5. Emergency command -> STOP

These rules are deterministic and fail-safe. The ML model only drives
when the governor says it is safe to do so.
"""

import time
from dataclasses import dataclass
from enum import Enum
import config


class SafetyViolation(Enum):
    NONE = "NONE"
    CRITICAL_FRONT = "CRITICAL_FRONT"
    CRITICAL_REAR = "CRITICAL_REAR"
    WATCHDOG_TIMEOUT = "WATCHDOG_TIMEOUT"
    SENSOR_FAILURE = "SENSOR_FAILURE"
    EMERGENCY = "EMERGENCY"


@dataclass
class SafetyDecision:
    is_safe: bool
    violation: SafetyViolation
    reason: str
    override_drive: str = config.CMD_STOP
    override_steer: str = config.CMD_STEER_STOP
    override_speed: int = 0


class SafetyGovernor:
    """
    Hard safety layer. Call check() every control loop cycle.
    If is_safe=False, ignore ML command and use override_drive/steer/speed.
    """

    CRITICAL_FRONT_DISTANCE = 25.0     # cm - immediate stop
    CRITICAL_REAR_DISTANCE = 30.0      # cm - immediate stop when reversing
    SENSOR_FAILURE_TIME = 3.0          # seconds of all-zeros before declaring failure

    def __init__(self):
        self.last_valid_sensor_time = time.time()
        self.violation_count = 0
        self.last_violation = SafetyViolation.NONE

    def check(self, distances: dict, requested_drive: str,
              last_command_timestamp: float) -> SafetyDecision:
        """
        Args:
            distances: dict of sensor distances (FL, FR, FW, BC, LS, RS)
            requested_drive: drive command being requested (FORWARD/BACKWARD/STOP)
            last_command_timestamp: Unix time of last laptop command

        Returns:
            SafetyDecision object
        """
        now = time.time()

        # Rule 5: Emergency command always wins
        if requested_drive == config.CMD_EMERGENCY:
            return SafetyDecision(
                is_safe=False,
                violation=SafetyViolation.EMERGENCY,
                reason="Emergency stop requested"
            )

        # Rule 3: Watchdog timeout
        if now - last_command_timestamp > config.WATCHDOG_TIMEOUT:
            return SafetyDecision(
                is_safe=False,
                violation=SafetyViolation.WATCHDOG_TIMEOUT,
                reason=f"No command for {now - last_command_timestamp:.1f}s"
            )

        # Rule 4: Sensor failure detection
        valid_readings = [d for d in distances.values()
                          if config.MIN_SENSOR_DISTANCE <= d <= config.MAX_SENSOR_DISTANCE]
        if valid_readings:
            self.last_valid_sensor_time = now
        elif now - self.last_valid_sensor_time > self.SENSOR_FAILURE_TIME:
            return SafetyDecision(
                is_safe=False,
                violation=SafetyViolation.SENSOR_FAILURE,
                reason="All sensors returning invalid readings"
            )

        # Rule 1: Critical front distance (only when driving forward)
        if requested_drive == config.CMD_FORWARD:
            front_min = self._min_valid(distances, ['FL', 'FR', 'FW'])
            if front_min is not None and front_min < self.CRITICAL_FRONT_DISTANCE:
                return SafetyDecision(
                    is_safe=False,
                    violation=SafetyViolation.CRITICAL_FRONT,
                    reason=f"Front obstacle at {front_min:.0f}cm"
                )

        # Rule 2: Critical rear distance (only when reversing)
        if requested_drive == config.CMD_BACKWARD:
            rear_min = self._min_valid(distances, ['BC'])
            if rear_min is not None and rear_min < self.CRITICAL_REAR_DISTANCE:
                return SafetyDecision(
                    is_safe=False,
                    violation=SafetyViolation.CRITICAL_REAR,
                    reason=f"Rear obstacle at {rear_min:.0f}cm"
                )

        # All checks passed
        return SafetyDecision(
            is_safe=True,
            violation=SafetyViolation.NONE,
            reason="OK"
        )

    def _min_valid(self, distances: dict, sensors: list):
        """Minimum distance among given sensor names, filtering invalid."""
        vals = []
        for s in sensors:
            d = distances.get(s, 0)
            if config.MIN_SENSOR_DISTANCE <= d <= config.MAX_SENSOR_DISTANCE:
                vals.append(d)
        return min(vals) if vals else None


if __name__ == "__main__":
    sg = SafetyGovernor()

    # Test 1: Safe cruise
    d1 = {'FL': 200, 'FR': 210, 'FW': 205, 'BC': 150, 'LS': 100, 'RS': 100}
    r = sg.check(d1, config.CMD_FORWARD, time.time())
    print(f"Test 1 (safe): safe={r.is_safe}, violation={r.violation.value}")

    # Test 2: Front obstacle
    d2 = {'FL': 20, 'FR': 22, 'FW': 18, 'BC': 150, 'LS': 100, 'RS': 100}
    r = sg.check(d2, config.CMD_FORWARD, time.time())
    print(f"Test 2 (front obstacle): safe={r.is_safe}, violation={r.violation.value}, reason={r.reason}")

    # Test 3: Watchdog timeout
    r = sg.check(d1, config.CMD_FORWARD, time.time() - 5)
    print(f"Test 3 (watchdog): safe={r.is_safe}, violation={r.violation.value}")

    # Test 4: All zeros
    d4 = {k: 0 for k in ['FL', 'FR', 'FW', 'BC', 'LS', 'RS']}
    r = sg.check(d4, config.CMD_FORWARD, time.time())
    print(f"Test 4 (zeros - first check OK): safe={r.is_safe}")
    time.sleep(0.1)
    sg.last_valid_sensor_time -= 5  # simulate 5s of zeros
    r = sg.check(d4, config.CMD_FORWARD, time.time())
    print(f"Test 4b (zeros - after timeout): safe={r.is_safe}, violation={r.violation.value}")
