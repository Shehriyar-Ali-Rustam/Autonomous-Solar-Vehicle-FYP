#!/usr/bin/env python3
"""Test script for both IBT_2 motor drivers (Drive + Steering)"""

import RPi.GPIO as GPIO
import time

# Drive motor pins
DRIVE_RPWM = 4   # Pin 7
DRIVE_LPWM = 17  # Pin 11
DRIVE_R_EN = 18  # Pin 12
DRIVE_L_EN = 27  # Pin 13

# Steering motor pins
STEER_RPWM = 22  # Pin 15
STEER_LPWM = 23  # Pin 16
STEER_R_EN = 24  # Pin 18
STEER_L_EN = 25  # Pin 22

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Setup all pins
for pin in [DRIVE_RPWM, DRIVE_LPWM, DRIVE_R_EN, DRIVE_L_EN,
            STEER_RPWM, STEER_LPWM, STEER_R_EN, STEER_L_EN]:
    GPIO.setup(pin, GPIO.OUT)

# Enable both drivers
GPIO.output(DRIVE_R_EN, GPIO.HIGH)
GPIO.output(DRIVE_L_EN, GPIO.HIGH)
GPIO.output(STEER_R_EN, GPIO.HIGH)
GPIO.output(STEER_L_EN, GPIO.HIGH)

# Setup PWM
drive_fwd = GPIO.PWM(DRIVE_RPWM, 1000)
drive_bwd = GPIO.PWM(DRIVE_LPWM, 1000)
steer_right = GPIO.PWM(STEER_RPWM, 1000)
steer_left = GPIO.PWM(STEER_LPWM, 1000)

drive_fwd.start(0)
drive_bwd.start(0)
steer_right.start(0)
steer_left.start(0)

def stop_all():
    drive_fwd.ChangeDutyCycle(0)
    drive_bwd.ChangeDutyCycle(0)
    steer_right.ChangeDutyCycle(0)
    steer_left.ChangeDutyCycle(0)

try:
    print("Both Motors Test")
    print("=" * 40)

    print("\n1. DRIVE FORWARD 50% - 2 sec")
    drive_fwd.ChangeDutyCycle(50)
    time.sleep(2)
    stop_all()
    time.sleep(1)

    print("2. DRIVE BACKWARD 50% - 2 sec")
    drive_bwd.ChangeDutyCycle(50)
    time.sleep(2)
    stop_all()
    time.sleep(1)

    print("3. STEER RIGHT 50% - 2 sec")
    steer_right.ChangeDutyCycle(50)
    time.sleep(2)
    stop_all()
    time.sleep(1)

    print("4. STEER LEFT 50% - 2 sec")
    steer_left.ChangeDutyCycle(50)
    time.sleep(2)
    stop_all()
    time.sleep(1)

    print("5. FORWARD + STEER RIGHT - 2 sec")
    drive_fwd.ChangeDutyCycle(50)
    steer_right.ChangeDutyCycle(50)
    time.sleep(2)
    stop_all()
    time.sleep(1)

    print("6. FORWARD + STEER LEFT - 2 sec")
    drive_fwd.ChangeDutyCycle(50)
    steer_left.ChangeDutyCycle(50)
    time.sleep(2)
    stop_all()

    print("\nAll tests complete!")

except KeyboardInterrupt:
    print("\nTest interrupted")
finally:
    stop_all()
    drive_fwd.stop()
    drive_bwd.stop()
    steer_right.stop()
    steer_left.stop()
    GPIO.output(DRIVE_R_EN, GPIO.LOW)
    GPIO.output(DRIVE_L_EN, GPIO.LOW)
    GPIO.output(STEER_R_EN, GPIO.LOW)
    GPIO.output(STEER_L_EN, GPIO.LOW)
    GPIO.cleanup()
    print("Cleanup done")
