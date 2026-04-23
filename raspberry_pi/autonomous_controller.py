#!/usr/bin/env python3
"""
Autonomous Controller
State machine for autonomous obstacle avoidance driving
"""

import time
import config


class AutonomousController:
    """
    Autonomous driving state machine.

    States:
        CRUISING     - Moving forward at cruise speed, path is clear
        SLOWING      - Obstacle ahead, gradually reducing speed
        AVOIDING     - Turning to avoid obstacle (uses side sensors)
        EMERGENCY_STOP - Sudden close obstacle, fully stopped
        REVERSING    - Going backward (after checking rear is clear)
        TURNING      - Turning after reversing to find new path
        WAITING      - Stuck (front and rear blocked), waiting for clearance
    """

    def __init__(self):
        self.state = config.AUTO_STATE_CRUISING
        self.active = False

        # Maneuver timing
        self.maneuver_start_time = 0
        self.wait_start_time = 0

        # Turn direction chosen during avoidance
        self.turn_direction = None  # 'LEFT' or 'RIGHT'

    def activate(self):
        """Enable autonomous mode"""
        self.active = True
        self.state = config.AUTO_STATE_CRUISING
        self.turn_direction = None
        print("[Autonomous] MODE ACTIVATED")

    def deactivate(self):
        """Disable autonomous mode"""
        self.active = False
        self.state = config.AUTO_STATE_CRUISING
        self.turn_direction = None
        print("[Autonomous] MODE DEACTIVATED")

    def is_active(self):
        return self.active

    def get_state(self):
        return self.state

    def _get_front_min(self, distances):
        """Get minimum front distance from FL, FR, FW sensors"""
        front_vals = []
        for s in ['FL', 'FR', 'FW']:
            d = distances.get(s, 0)
            if config.MIN_SENSOR_DISTANCE <= d <= config.MAX_SENSOR_DISTANCE:
                front_vals.append(d)
        return min(front_vals) if front_vals else 0

    def _get_rear_min(self, distances):
        """Get minimum rear distance from BC sensor"""
        d = distances.get('BC', 0)
        if config.MIN_SENSOR_DISTANCE <= d <= config.MAX_SENSOR_DISTANCE:
            return d
        return 0

    def _get_left_distance(self, distances):
        """Get left side distance"""
        d = distances.get('LS', 0)
        if config.MIN_SENSOR_DISTANCE <= d <= config.MAX_SENSOR_DISTANCE:
            return d
        return 0

    def _get_right_distance(self, distances):
        """Get right side distance"""
        d = distances.get('RS', 0)
        if config.MIN_SENSOR_DISTANCE <= d <= config.MAX_SENSOR_DISTANCE:
            return d
        return 0

    def _choose_turn_direction(self, distances):
        """Choose best turn direction based on side sensors"""
        left = self._get_left_distance(distances)
        right = self._get_right_distance(distances)

        # If both sides have readings, pick the clearer side
        if left > 0 and right > 0:
            if left >= right:
                return config.CMD_LEFT
            else:
                return config.CMD_RIGHT

        # If only one side has a reading, pick the one with more space
        if left > config.AUTO_DISTANCE_SIDE_MIN:
            return config.CMD_LEFT
        if right > config.AUTO_DISTANCE_SIDE_MIN:
            return config.CMD_RIGHT

        # Default: turn right
        return config.CMD_RIGHT

    def decide(self, distances):
        """
        Main decision function. Called every control loop cycle.

        Args:
            distances: dict with sensor readings (FL, FR, FW, BC, LS, RS)

        Returns:
            tuple: (drive_command, steer_command, drive_speed)
        """
        if not self.active:
            return config.CMD_STOP, config.CMD_STEER_STOP, 0

        front = self._get_front_min(distances)
        rear = self._get_rear_min(distances)
        now = time.time()

        # ---- STATE: CRUISING ----
        if self.state == config.AUTO_STATE_CRUISING:
            if front == 0:
                # No valid reading, stop for safety
                return config.CMD_FORWARD, config.CMD_STEER_STOP, config.AUTO_SLOW_SPEED

            if front < config.AUTO_DISTANCE_EMERGENCY:
                # Sudden close obstacle → emergency stop
                self.state = config.AUTO_STATE_EMERGENCY
                self.maneuver_start_time = now
                print(f"[Autonomous] EMERGENCY STOP! Front={front:.0f}cm")
                return config.CMD_STOP, config.CMD_STEER_STOP, 0

            if front < config.AUTO_DISTANCE_AVOID:
                # Close obstacle → start avoidance turn
                self.turn_direction = self._choose_turn_direction(distances)
                self.state = config.AUTO_STATE_AVOIDING
                self.maneuver_start_time = now
                print(f"[Autonomous] AVOIDING → turning {self.turn_direction}, Front={front:.0f}cm")
                return config.CMD_FORWARD, self.turn_direction, config.AUTO_SLOW_SPEED

            if front < config.AUTO_DISTANCE_SLOW:
                # Obstacle detected → slow down
                self.state = config.AUTO_STATE_SLOWING
                print(f"[Autonomous] SLOWING, Front={front:.0f}cm")
                return config.CMD_FORWARD, config.CMD_STEER_STOP, config.AUTO_SLOW_SPEED

            # Path clear → cruise
            return config.CMD_FORWARD, config.CMD_STEER_STOP, config.AUTO_CRUISE_SPEED

        # ---- STATE: SLOWING ----
        elif self.state == config.AUTO_STATE_SLOWING:
            if front == 0:
                return config.CMD_FORWARD, config.CMD_STEER_STOP, config.AUTO_SLOW_SPEED

            if front < config.AUTO_DISTANCE_EMERGENCY:
                self.state = config.AUTO_STATE_EMERGENCY
                self.maneuver_start_time = now
                print(f"[Autonomous] EMERGENCY STOP! Front={front:.0f}cm")
                return config.CMD_STOP, config.CMD_STEER_STOP, 0

            if front < config.AUTO_DISTANCE_AVOID:
                self.turn_direction = self._choose_turn_direction(distances)
                self.state = config.AUTO_STATE_AVOIDING
                self.maneuver_start_time = now
                print(f"[Autonomous] AVOIDING → turning {self.turn_direction}, Front={front:.0f}cm")
                return config.CMD_FORWARD, self.turn_direction, config.AUTO_SLOW_SPEED

            if front >= config.AUTO_DISTANCE_SLOW:
                # Obstacle cleared → back to cruising
                self.state = config.AUTO_STATE_CRUISING
                print(f"[Autonomous] CRUISING, Front={front:.0f}cm")
                return config.CMD_FORWARD, config.CMD_STEER_STOP, config.AUTO_CRUISE_SPEED

            # Still in slow zone
            # Gradual speed: map distance 70-100cm to speed 30-50%
            speed_range = config.AUTO_CRUISE_SPEED - config.AUTO_SLOW_SPEED
            dist_range = config.AUTO_DISTANCE_SLOW - config.AUTO_DISTANCE_AVOID
            if dist_range > 0:
                ratio = (front - config.AUTO_DISTANCE_AVOID) / dist_range
                speed = config.AUTO_SLOW_SPEED + int(ratio * speed_range)
            else:
                speed = config.AUTO_SLOW_SPEED
            return config.CMD_FORWARD, config.CMD_STEER_STOP, speed

        # ---- STATE: AVOIDING ----
        elif self.state == config.AUTO_STATE_AVOIDING:
            if front < config.AUTO_DISTANCE_EMERGENCY:
                self.state = config.AUTO_STATE_EMERGENCY
                self.maneuver_start_time = now
                print(f"[Autonomous] EMERGENCY STOP during avoidance! Front={front:.0f}cm")
                return config.CMD_STOP, config.CMD_STEER_STOP, 0

            if front >= config.AUTO_DISTANCE_SLOW:
                # Path is clear now → resume cruising
                self.state = config.AUTO_STATE_CRUISING
                self.turn_direction = None
                print(f"[Autonomous] Avoidance complete → CRUISING, Front={front:.0f}cm")
                return config.CMD_FORWARD, config.CMD_STEER_STOP, config.AUTO_CRUISE_SPEED

            # Keep turning and moving slowly
            return config.CMD_FORWARD, self.turn_direction, config.AUTO_SLOW_SPEED

        # ---- STATE: EMERGENCY STOP ----
        elif self.state == config.AUTO_STATE_EMERGENCY:
            # Wait a brief moment after stopping (0.5s)
            if now - self.maneuver_start_time < 0.5:
                return config.CMD_STOP, config.CMD_STEER_STOP, 0

            # Check if rear is clear to reverse
            if rear > config.AUTO_DISTANCE_REAR_SAFE:
                self.state = config.AUTO_STATE_REVERSING
                self.maneuver_start_time = now
                print(f"[Autonomous] REVERSING, Rear={rear:.0f}cm")
                return config.CMD_BACKWARD, config.CMD_STEER_STOP, config.AUTO_REVERSE_SPEED

            # Rear is blocked → wait
            self.state = config.AUTO_STATE_WAITING
            self.wait_start_time = now
            print(f"[Autonomous] WAITING (rear blocked), Rear={rear:.0f}cm")
            return config.CMD_STOP, config.CMD_STEER_STOP, 0

        # ---- STATE: REVERSING ----
        elif self.state == config.AUTO_STATE_REVERSING:
            # Check rear safety while reversing
            if rear > 0 and rear < config.AUTO_DISTANCE_REAR_SAFE:
                # Something appeared behind us, stop reversing
                self.state = config.AUTO_STATE_WAITING
                self.wait_start_time = now
                print(f"[Autonomous] Rear obstacle while reversing! Rear={rear:.0f}cm")
                return config.CMD_STOP, config.CMD_STEER_STOP, 0

            # Reverse for the configured time
            if now - self.maneuver_start_time >= config.AUTO_REVERSE_TIME:
                # Done reversing → turn
                self.turn_direction = self._choose_turn_direction(distances)
                self.state = config.AUTO_STATE_TURNING
                self.maneuver_start_time = now
                print(f"[Autonomous] TURNING {self.turn_direction}")
                return config.CMD_FORWARD, self.turn_direction, config.AUTO_SLOW_SPEED

            return config.CMD_BACKWARD, config.CMD_STEER_STOP, config.AUTO_REVERSE_SPEED

        # ---- STATE: TURNING ----
        elif self.state == config.AUTO_STATE_TURNING:
            if front < config.AUTO_DISTANCE_EMERGENCY:
                self.state = config.AUTO_STATE_EMERGENCY
                self.maneuver_start_time = now
                return config.CMD_STOP, config.CMD_STEER_STOP, 0

            # Turn for the configured time
            if now - self.maneuver_start_time >= config.AUTO_TURN_TIME:
                # Done turning → resume cruising
                self.state = config.AUTO_STATE_CRUISING
                self.turn_direction = None
                print(f"[Autonomous] Turn complete → CRUISING")
                return config.CMD_FORWARD, config.CMD_STEER_STOP, config.AUTO_CRUISE_SPEED

            return config.CMD_FORWARD, self.turn_direction, config.AUTO_SLOW_SPEED

        # ---- STATE: WAITING ----
        elif self.state == config.AUTO_STATE_WAITING:
            # Check if front has cleared
            if front >= config.AUTO_DISTANCE_AVOID:
                self.state = config.AUTO_STATE_CRUISING
                print(f"[Autonomous] Path cleared → CRUISING, Front={front:.0f}cm")
                return config.CMD_FORWARD, config.CMD_STEER_STOP, config.AUTO_CRUISE_SPEED

            # Check if rear cleared (can try reversing again)
            if rear > config.AUTO_DISTANCE_REAR_SAFE:
                elapsed = now - self.wait_start_time
                # Wait at least 2 seconds before trying reverse again
                if elapsed >= 2.0:
                    self.state = config.AUTO_STATE_REVERSING
                    self.maneuver_start_time = now
                    print(f"[Autonomous] Retrying REVERSE, Rear={rear:.0f}cm")
                    return config.CMD_BACKWARD, config.CMD_STEER_STOP, config.AUTO_REVERSE_SPEED

            # Timeout: try moving forward slowly after waiting too long
            if now - self.wait_start_time >= config.AUTO_WAIT_TIMEOUT:
                self.state = config.AUTO_STATE_CRUISING
                print(f"[Autonomous] Wait timeout → trying CRUISING")
                return config.CMD_FORWARD, config.CMD_STEER_STOP, config.AUTO_SLOW_SPEED

            # Keep waiting
            return config.CMD_STOP, config.CMD_STEER_STOP, 0

        # Fallback
        return config.CMD_STOP, config.CMD_STEER_STOP, 0


# Test code
if __name__ == "__main__":
    print("Autonomous Controller - State Machine Test")
    print("=" * 50)

    ac = AutonomousController()
    ac.activate()

    # Simulate different scenarios
    test_scenarios = [
        ("Clear path", {'FL': 350, 'FR': 340, 'FW': 360, 'BC': 200, 'LS': 150, 'RS': 180}),
        ("Obstacle ahead", {'FL': 150, 'FR': 140, 'FW': 130, 'BC': 200, 'LS': 150, 'RS': 180}),
        ("Close obstacle", {'FL': 60, 'FR': 55, 'FW': 50, 'BC': 200, 'LS': 150, 'RS': 80}),
        ("Very close!", {'FL': 30, 'FR': 25, 'FW': 20, 'BC': 200, 'LS': 150, 'RS': 180}),
        ("Rear blocked", {'FL': 30, 'FR': 25, 'FW': 20, 'BC': 50, 'LS': 150, 'RS': 180}),
        ("All clear again", {'FL': 350, 'FR': 340, 'FW': 360, 'BC': 200, 'LS': 150, 'RS': 180}),
    ]

    for name, distances in test_scenarios:
        drive, steer, speed = ac.decide(distances)
        print(f"\n{name}:")
        print(f"  State: {ac.get_state()}")
        print(f"  Drive: {drive}, Steer: {steer}, Speed: {speed}%")
        print(f"  Sensors: FL={distances['FL']} FR={distances['FR']} FW={distances['FW']} "
              f"BC={distances['BC']} LS={distances['LS']} RS={distances['RS']}")

    print("\nTest complete!")
