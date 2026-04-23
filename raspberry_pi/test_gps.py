#!/usr/bin/env python3
"""Test script for GY-GPS6MV2 GPS module"""

import serial
import pynmea2
import time

GPS_PORT = '/dev/serial0'
GPS_BAUD = 9600

print("GPS Module Test")
print("=" * 40)
print(f"Port: {GPS_PORT}")
print(f"Baud: {GPS_BAUD}")
print("Waiting for GPS data (may take 30-60 seconds for first fix)...")
print("Press Ctrl+C to stop\n")

try:
    ser = serial.Serial(GPS_PORT, GPS_BAUD, timeout=1)

    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()

        if not line:
            continue

        # Show raw data
        print(f"RAW: {line}")

        # Parse GPGGA sentences (position data)
        if line.startswith('$GPGGA') or line.startswith('$GNGGA'):
            try:
                msg = pynmea2.parse(line)
                if msg.latitude and msg.longitude:
                    print(f"\n*** GPS FIX ***")
                    print(f"  Latitude:   {msg.latitude} {msg.lat_dir}")
                    print(f"  Longitude:  {msg.longitude} {msg.lon_dir}")
                    print(f"  Altitude:   {msg.altitude} {msg.altitude_units}")
                    print(f"  Satellites: {msg.num_sats}")
                    print(f"  Quality:    {msg.gps_qual}\n")
                else:
                    print("  (No fix yet - move GPS near a window or outside)")
            except pynmea2.ParseError:
                pass

        # Parse GPRMC sentences (speed and heading)
        elif line.startswith('$GPRMC') or line.startswith('$GNRMC'):
            try:
                msg = pynmea2.parse(line)
                if msg.status == 'A':
                    speed_kmh = msg.spd_over_grnd * 1.852 if msg.spd_over_grnd else 0
                    print(f"  Speed: {speed_kmh:.1f} km/h  Heading: {msg.true_course}°")
            except pynmea2.ParseError:
                pass

except serial.SerialException as e:
    print(f"\nERROR: {e}")
    print("\nCheck:")
    print("1. GPS wired correctly (TX→Pin10, RX→Pin8)")
    print("2. Serial enabled: sudo raspi-config → Interface → Serial")
    print("3. Rebooted after enabling serial")
except KeyboardInterrupt:
    print("\n\nTest stopped.")
finally:
    try:
        ser.close()
    except:
        pass
    print("Done.")
