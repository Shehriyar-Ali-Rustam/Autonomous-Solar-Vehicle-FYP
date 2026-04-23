#!/usr/bin/env python3
"""Test script for steering IBT_2 motor driver"""

import RPi.GPIO as GPIO
import time

# Steering GPIO pins
STEER_RPWM = 22  # GPIO 22 (Pin 15) - Steer Right
STEER_LPWM = 23  # GPIO 23 (Pin 16) - Steer Left
STEER_R_EN = 24  # GPIO 24 (Pin 18) - Right Enable
STEER_L_EN = 25  # GPIO 25 (Pin 22) - Left Enable

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Setup pins
GPIO.setup(STEER_RPWM, GPIO.OUT)
GPIO.setup(STEER_LPWM, GPIO.OUT)
GPIO.setup(STEER_R_EN, GPIO.OUT)
GPIO.setup(STEER_L_EN, GPIO.OUT)

# Enable driver
GPIO.output(STEER_R_EN, GPIO.HIGH)
GPIO.output(STEER_L_EN, GPIO.HIGH)

# Setup PWM
pwm_right = GPIO.PWM(STEER_RPWM, 1000)
pwm_left = GPIO.PWM(STEER_LPWM, 1000)
pwm_right.start(0)
pwm_left.start(0)

try:
    print("Steering Motor Test")
    print("=" * 30)

    print("\n1. Steer RIGHT at 50% for 2 seconds...")
    pwm_left.ChangeDutyCycle(0)
    pwm_right.ChangeDutyCycle(50)
    time.sleep(2)

    print("2. Stop...")
    pwm_right.ChangeDutyCycle(0)
    time.sleep(1)

    print("3. Steer LEFT at 50% for 2 seconds...")
    pwm_right.ChangeDutyCycle(0)
    pwm_left.ChangeDutyCycle(50)
    time.sleep(2)

    print("4. Stop...")
    pwm_left.ChangeDutyCycle(0)
    time.sleep(1)

    print("5. Steer RIGHT at 30% for 2 seconds...")
    pwm_left.ChangeDutyCycle(0)
    pwm_right.ChangeDutyCycle(30)
    time.sleep(2)

    print("6. Stop...")
    pwm_right.ChangeDutyCycle(0)

    print("\nTest complete!")

except KeyboardInterrupt:
    print("\nTest interrupted")
finally:
    pwm_right.stop()
    pwm_left.stop()
    GPIO.output(STEER_R_EN, GPIO.LOW)
    GPIO.output(STEER_L_EN, GPIO.LOW)
    GPIO.cleanup()
    print("Cleanup done")
