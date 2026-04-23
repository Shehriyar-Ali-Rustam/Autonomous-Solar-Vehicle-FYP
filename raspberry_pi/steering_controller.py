#!/usr/bin/env python3
"""
Steering Controller for IBT_2 Motor Driver
Handles steering direction control via PWM
"""

import RPi.GPIO as GPIO
import time
import config


class SteeringController:
    """
    Controls IBT_2 steering motor driver via GPIO pins with PWM
    """

    def __init__(self):
        self.pwm_right = None
        self.pwm_left = None
        self.is_setup = False
        self.current_direction = None
        self.current_speed = 0

    def setup(self):
        """Initialize GPIO pins and PWM channels"""
        if self.is_setup:
            return

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        GPIO.setup(config.STEER_RPWM, GPIO.OUT)
        GPIO.setup(config.STEER_LPWM, GPIO.OUT)
        GPIO.setup(config.STEER_R_EN, GPIO.OUT)
        GPIO.setup(config.STEER_L_EN, GPIO.OUT)

        GPIO.output(config.STEER_R_EN, GPIO.HIGH)
        GPIO.output(config.STEER_L_EN, GPIO.HIGH)

        self.pwm_right = GPIO.PWM(config.STEER_RPWM, config.PWM_FREQUENCY)
        self.pwm_left = GPIO.PWM(config.STEER_LPWM, config.PWM_FREQUENCY)

        self.pwm_right.start(0)
        self.pwm_left.start(0)

        self.is_setup = True
        print("[SteeringController] GPIO setup complete")

    def set_direction(self, direction, speed_percent=None):
        """
        Set steering direction

        Args:
            direction (str): 'LEFT', 'RIGHT', or 'STEER_STOP'
            speed_percent (int): Speed 0-100% (default: config.STEER_SPEED)
        """
        if not self.is_setup:
            raise RuntimeError("Steering controller not setup. Call setup() first.")

        if speed_percent is None:
            speed_percent = config.STEER_SPEED

        speed_percent = max(0, min(100, speed_percent))

        self.current_direction = direction
        self.current_speed = speed_percent

        if direction == config.CMD_RIGHT:
            self.pwm_left.ChangeDutyCycle(0)
            self.pwm_right.ChangeDutyCycle(speed_percent)
        elif direction == config.CMD_LEFT:
            self.pwm_right.ChangeDutyCycle(0)
            self.pwm_left.ChangeDutyCycle(speed_percent)
        else:
            self._stop()

    def _stop(self):
        """Stop steering motor"""
        if self.pwm_right and self.pwm_left:
            self.pwm_right.ChangeDutyCycle(0)
            self.pwm_left.ChangeDutyCycle(0)
        self.current_direction = config.CMD_STEER_STOP
        self.current_speed = 0

    def stop(self):
        """Public stop method"""
        self._stop()

    def emergency_stop(self):
        """Emergency stop - disable driver"""
        print("[SteeringController] EMERGENCY STOP!")
        if self.pwm_right and self.pwm_left:
            self.pwm_right.ChangeDutyCycle(0)
            self.pwm_left.ChangeDutyCycle(0)
        GPIO.output(config.STEER_R_EN, GPIO.LOW)
        GPIO.output(config.STEER_L_EN, GPIO.LOW)
        self.current_direction = config.CMD_EMERGENCY
        self.current_speed = 0

    def get_status(self):
        """Get current steering status"""
        return {
            'steering_direction': self.current_direction,
            'steering_speed': self.current_speed
        }

    def cleanup(self):
        """Clean up GPIO resources"""
        if not self.is_setup:
            return

        print("[SteeringController] Cleaning up GPIO...")
        self._stop()

        if self.pwm_right:
            self.pwm_right.stop()
        if self.pwm_left:
            self.pwm_left.stop()

        GPIO.output(config.STEER_R_EN, GPIO.LOW)
        GPIO.output(config.STEER_L_EN, GPIO.LOW)

        self.is_setup = False
        print("[SteeringController] Cleanup complete")


if __name__ == "__main__":
    print("Testing Steering Controller...")

    sc = SteeringController()
    sc.setup()

    try:
        print("\n1. Steer RIGHT at 50% for 2 seconds...")
        sc.set_direction(config.CMD_RIGHT, 50)
        time.sleep(2)

        print("2. Stop...")
        sc.stop()
        time.sleep(1)

        print("3. Steer LEFT at 50% for 2 seconds...")
        sc.set_direction(config.CMD_LEFT, 50)
        time.sleep(2)

        print("4. Stop...")
        sc.stop()

        print("\nTest complete!")

    except KeyboardInterrupt:
        print("\nTest interrupted")
    finally:
        sc.cleanup()
