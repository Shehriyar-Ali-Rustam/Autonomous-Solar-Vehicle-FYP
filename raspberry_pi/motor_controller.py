#!/usr/bin/env python3
"""
Motor Controller for IBT_2 Motor Driver
Handles motor speed and direction control via PWM
"""

import RPi.GPIO as GPIO
import time
import config

class MotorController:
    """
    Controls IBT_2 motor driver via GPIO pins with PWM
    """

    def __init__(self):
        """Initialize motor controller (doesn't setup GPIO yet)"""
        self.pwm_right = None
        self.pwm_left = None
        self.is_setup = False
        self.current_direction = None
        self.current_speed = 0

    def setup(self):
        """
        Initialize GPIO pins and PWM channels
        Call this before using any motor control methods
        """
        if self.is_setup:
            return

        # Set GPIO mode to BCM numbering
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        # Setup pins as outputs
        GPIO.setup(config.MOTOR_RPWM, GPIO.OUT)
        GPIO.setup(config.MOTOR_LPWM, GPIO.OUT)
        GPIO.setup(config.MOTOR_R_EN, GPIO.OUT)
        GPIO.setup(config.MOTOR_L_EN, GPIO.OUT)

        # Enable the motor driver
        GPIO.output(config.MOTOR_R_EN, GPIO.HIGH)
        GPIO.output(config.MOTOR_L_EN, GPIO.HIGH)

        # Setup PWM on RPWM and LPWM pins
        self.pwm_right = GPIO.PWM(config.MOTOR_RPWM, config.PWM_FREQUENCY)
        self.pwm_left = GPIO.PWM(config.MOTOR_LPWM, config.PWM_FREQUENCY)

        # Start PWM with 0% duty cycle (motor stopped)
        self.pwm_right.start(0)
        self.pwm_left.start(0)

        self.is_setup = True
        print("[MotorController] GPIO setup complete")

    def set_speed(self, direction, speed_percent):
        """
        Set motor speed and direction

        Args:
            direction (str): Direction - 'FORWARD', 'BACKWARD', or 'STOP'
            speed_percent (int): Speed from 0-100%
        """
        if not self.is_setup:
            raise RuntimeError("Motor controller not setup. Call setup() first.")

        # Clamp speed to valid range
        speed_percent = max(0, min(100, speed_percent))

        # Store current state
        self.current_direction = direction
        self.current_speed = speed_percent

        if direction == config.CMD_FORWARD:
            self._motor_forward(speed_percent)
        elif direction == config.CMD_BACKWARD:
            self._motor_backward(speed_percent)
        elif direction == config.CMD_STOP:
            self._motor_stop()
        elif direction == config.CMD_EMERGENCY:
            self.emergency_stop()
        else:
            print(f"[MotorController] Warning: Unknown direction '{direction}', stopping motor")
            self._motor_stop()

    def _motor_forward(self, speed):
        """Run motor forward at specified speed"""
        if speed == 0:
            self._motor_stop()
            return

        self.pwm_left.ChangeDutyCycle(0)      # Ensure LPWM is off
        self.pwm_right.ChangeDutyCycle(speed)  # Set RPWM for forward

    def _motor_backward(self, speed):
        """Run motor backward at specified speed"""
        if speed == 0:
            self._motor_stop()
            return

        self.pwm_right.ChangeDutyCycle(0)     # Ensure RPWM is off
        self.pwm_left.ChangeDutyCycle(speed)   # Set LPWM for backward

    def _motor_stop(self):
        """Stop the motor (normal stop)"""
        if self.pwm_right and self.pwm_left:
            self.pwm_right.ChangeDutyCycle(0)
            self.pwm_left.ChangeDutyCycle(0)
        self.current_direction = config.CMD_STOP
        self.current_speed = 0

    def stop(self):
        """Public method to stop the motor"""
        self._motor_stop()

    def emergency_stop(self):
        """
        Emergency stop - immediately halt motor and disable driver
        Requires restart to resume operation
        """
        print("[MotorController] EMERGENCY STOP!")

        # Stop PWM immediately
        if self.pwm_right and self.pwm_left:
            self.pwm_right.ChangeDutyCycle(0)
            self.pwm_left.ChangeDutyCycle(0)

        # Disable motor driver
        GPIO.output(config.MOTOR_R_EN, GPIO.LOW)
        GPIO.output(config.MOTOR_L_EN, GPIO.LOW)

        self.current_direction = config.CMD_EMERGENCY
        self.current_speed = 0

    def get_status(self):
        """
        Get current motor status

        Returns:
            dict: Current direction and speed
        """
        return {
            'direction': self.current_direction,
            'speed': self.current_speed
        }

    def cleanup(self):
        """
        Clean up GPIO resources
        Call this before exiting program
        """
        if not self.is_setup:
            return

        print("[MotorController] Cleaning up GPIO...")

        # Stop motor first
        self._motor_stop()

        # Stop PWM
        if self.pwm_right:
            self.pwm_right.stop()
        if self.pwm_left:
            self.pwm_left.stop()

        # Disable motor driver
        GPIO.output(config.MOTOR_R_EN, GPIO.LOW)
        GPIO.output(config.MOTOR_L_EN, GPIO.LOW)

        # Cleanup GPIO
        GPIO.cleanup()

        self.is_setup = False
        print("[MotorController] Cleanup complete")

# Test code - run this file directly to test motor controller
if __name__ == "__main__":
    print("Testing Motor Controller...")
    print("Make sure IBT_2 is properly wired before running this test!")

    mc = MotorController()
    mc.setup()

    try:
        print("\n1. Forward at 50% speed for 2 seconds...")
        mc.set_speed(config.CMD_FORWARD, 50)
        time.sleep(2)

        print("2. Stopping...")
        mc.stop()
        time.sleep(1)

        print("3. Backward at 50% speed for 2 seconds...")
        mc.set_speed(config.CMD_BACKWARD, 50)
        time.sleep(2)

        print("4. Stopping...")
        mc.stop()
        time.sleep(1)

        print("5. Forward at 30% speed for 2 seconds...")
        mc.set_speed(config.CMD_FORWARD, 30)
        time.sleep(2)

        print("6. Final stop...")
        mc.stop()

        print("\nTest complete!")

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        mc.cleanup()
