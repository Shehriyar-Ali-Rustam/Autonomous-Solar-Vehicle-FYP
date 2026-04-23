#!/usr/bin/env python3
"""
Waterproof Ultrasonic Sensor Reader
Reads waterproof ultrasonic sensor connected directly to Raspberry Pi GPIO
"""

import RPi.GPIO as GPIO
import time
import threading
import config

class WaterproofSensor:
    """
    Reads waterproof ultrasonic sensor via GPIO
    """

    def __init__(self, trig_pin=None, echo_pin=None):
        """
        Initialize waterproof sensor

        Args:
            trig_pin (int): GPIO pin for trigger (default from config)
            echo_pin (int): GPIO pin for echo (default from config)
        """
        self.trig_pin = trig_pin if trig_pin else config.WATERPROOF_TRIG
        self.echo_pin = echo_pin if echo_pin else config.WATERPROOF_ECHO

        self.distance = 0.0
        self.is_setup = False
        self.is_reading = False
        self.lock = threading.Lock()
        self.read_thread = None

    def setup(self):
        """Initialize GPIO pins for sensor"""
        if self.is_setup:
            return

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        # Setup trigger as output
        GPIO.setup(self.trig_pin, GPIO.OUT)
        GPIO.output(self.trig_pin, GPIO.LOW)

        # Setup echo as input
        GPIO.setup(self.echo_pin, GPIO.IN)

        # Wait for sensor to settle
        time.sleep(0.1)

        self.is_setup = True
        print(f"[WaterproofSensor] Setup complete (TRIG: GPIO{self.trig_pin}, ECHO: GPIO{self.echo_pin})")

    def read_distance(self):
        """
        Read distance from sensor (single reading)

        Returns:
            float: Distance in centimeters (0 if error)
        """
        if not self.is_setup:
            return 0.0

        try:
            # Ensure trigger is low
            GPIO.output(self.trig_pin, GPIO.LOW)
            time.sleep(0.00002)  # 20 microseconds

            # Send 10us pulse to trigger
            GPIO.output(self.trig_pin, GPIO.HIGH)
            time.sleep(0.00001)  # 10 microseconds
            GPIO.output(self.trig_pin, GPIO.LOW)

            # Wait for echo to go HIGH (start of pulse)
            pulse_start = time.time()
            timeout_start = pulse_start
            while GPIO.input(self.echo_pin) == GPIO.LOW:
                pulse_start = time.time()
                if pulse_start - timeout_start > 0.03:  # 30ms timeout
                    return 0.0

            # Wait for echo to go LOW (end of pulse)
            pulse_end = time.time()
            timeout_end = pulse_end
            while GPIO.input(self.echo_pin) == GPIO.HIGH:
                pulse_end = time.time()
                if pulse_end - timeout_end > 0.03:  # 30ms timeout
                    return 0.0

            # Calculate distance
            pulse_duration = pulse_end - pulse_start
            # Speed of sound: 34300 cm/s
            # Distance = (Time × Speed) / 2 (round trip)
            distance = (pulse_duration * 34300) / 2

            # Validate distance (2cm to 400cm for HC-SR04)
            if config.MIN_SENSOR_DISTANCE <= distance <= config.MAX_SENSOR_DISTANCE:
                return round(distance, 1)
            else:
                return 0.0

        except Exception as e:
            # print(f"[WaterproofSensor] Error reading: {e}")
            return 0.0

    def start_continuous_reading(self):
        """Start reading sensor in background thread"""
        if self.is_reading:
            print("[WaterproofSensor] Already reading")
            return

        if not self.is_setup:
            self.setup()

        self.is_reading = True
        self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.read_thread.start()
        print("[WaterproofSensor] Started continuous reading")

    def _read_loop(self):
        """Background thread that continuously reads sensor"""
        while self.is_reading:
            try:
                distance = self.read_distance()

                with self.lock:
                    self.distance = distance

                # Wait 100ms between readings (10Hz)
                time.sleep(0.1)

            except Exception as e:
                # print(f"[WaterproofSensor] Read loop error: {e}")
                time.sleep(0.1)

    def get_distance(self):
        """
        Get latest distance reading

        Returns:
            float: Distance in centimeters
        """
        with self.lock:
            return self.distance

    def stop_reading(self):
        """Stop continuous reading"""
        if not self.is_reading:
            return

        print("[WaterproofSensor] Stopping reading...")
        self.is_reading = False

        if self.read_thread:
            self.read_thread.join(timeout=1.0)

        print("[WaterproofSensor] Reading stopped")

    def cleanup(self):
        """Cleanup GPIO resources"""
        if self.is_reading:
            self.stop_reading()

        if self.is_setup:
            GPIO.cleanup([self.trig_pin, self.echo_pin])
            self.is_setup = False
            print("[WaterproofSensor] Cleanup complete")


# Test code
if __name__ == "__main__":
    print("Testing Waterproof Ultrasonic Sensor...")
    print(f"TRIG: GPIO {config.WATERPROOF_TRIG}, ECHO: GPIO {config.WATERPROOF_ECHO}\n")

    sensor = WaterproofSensor()

    try:
        sensor.setup()

        # Test 1: Single readings
        print("Test 1: Taking 5 single readings...")
        for i in range(5):
            dist = sensor.read_distance()
            print(f"  Reading {i+1}: {dist:.1f} cm")
            time.sleep(0.5)

        print("\nTest 2: Continuous reading for 10 seconds...")
        sensor.start_continuous_reading()

        for i in range(20):
            dist = sensor.get_distance()
            print(f"  [{i+1}/20] Distance: {dist:.1f} cm")
            time.sleep(0.5)

        print("\nTest complete!")

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        sensor.cleanup()
