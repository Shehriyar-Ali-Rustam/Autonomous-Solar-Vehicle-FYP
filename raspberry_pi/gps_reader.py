#!/usr/bin/env python3
"""
GPS Reader for GY-GPS6MV2 (NEO-6M module)
Reads NMEA sentences from Pi UART and parses to lat/lon/speed/heading.

Wiring (Pi UART):
    GPS VCC -> Pi 5V (pin 2)
    GPS GND -> Pi GND (pin 6)
    GPS TX  -> Pi RX (GPIO 15, pin 10)
    GPS RX  -> Pi TX (GPIO 14, pin 8)

Enable UART on Pi:
    sudo raspi-config -> Interface Options -> Serial Port
    - Login shell over serial: NO
    - Serial port hardware: YES
    Reboot.

Default device: /dev/ttyAMA0 or /dev/serial0
"""

import threading
import time
import math
from dataclasses import dataclass
from typing import Optional

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    serial = None


@dataclass
class GPSFix:
    latitude: float        # decimal degrees
    longitude: float       # decimal degrees
    altitude: float        # meters
    speed_mps: float       # meters / second
    heading_deg: float     # 0-360, 0=North
    satellites: int
    hdop: float            # horizontal dilution of precision
    timestamp: float       # unix time of fix
    valid: bool


class GPSReader:
    def __init__(self, port: str = '/dev/serial0', baudrate: int = 9600):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.is_running = False
        self.thread = None
        self.lock = threading.Lock()
        self._last_fix: Optional[GPSFix] = None

    def start(self) -> bool:
        if not SERIAL_AVAILABLE:
            print("[GPSReader] pyserial not installed - GPS disabled")
            return False

        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=1.0)
            self.is_running = True
            self.thread = threading.Thread(target=self._read_loop, daemon=True)
            self.thread.start()
            print(f"[GPSReader] Started on {self.port} @ {self.baudrate}")
            return True
        except Exception as e:
            print(f"[GPSReader] Failed to open {self.port}: {e}")
            return False

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.serial:
            try:
                self.serial.close()
            except Exception:
                pass

    def get_fix(self) -> Optional[GPSFix]:
        with self.lock:
            return self._last_fix

    def _read_loop(self):
        while self.is_running:
            try:
                line = self.serial.readline().decode('ascii', errors='ignore').strip()
                if not line.startswith('$'):
                    continue
                fix = self._parse_nmea(line)
                if fix:
                    with self.lock:
                        self._last_fix = fix
            except Exception as e:
                time.sleep(0.1)

    def _parse_nmea(self, line: str) -> Optional[GPSFix]:
        """Parse $GPRMC and $GPGGA sentences."""
        parts = line.split(',')

        try:
            if parts[0] in ('$GPRMC', '$GNRMC'):
                # $GPRMC,time,status,lat,N/S,lon,E/W,speed_knots,course,date,...
                if len(parts) < 10 or parts[2] != 'A':  # A = active
                    return None
                lat = self._nmea_to_decimal(parts[3], parts[4])
                lon = self._nmea_to_decimal(parts[5], parts[6])
                speed_knots = float(parts[7]) if parts[7] else 0.0
                heading = float(parts[8]) if parts[8] else 0.0
                speed_mps = speed_knots * 0.514444
                prev = self._last_fix
                alt = prev.altitude if prev else 0.0
                sats = prev.satellites if prev else 0
                hdop = prev.hdop if prev else 99.0
                return GPSFix(lat, lon, alt, speed_mps, heading,
                              sats, hdop, time.time(), True)

            elif parts[0] in ('$GPGGA', '$GNGGA'):
                # $GPGGA,time,lat,N/S,lon,E/W,fix_quality,num_sats,hdop,altitude,M,...
                if len(parts) < 10 or parts[6] == '0':
                    return None
                lat = self._nmea_to_decimal(parts[2], parts[3])
                lon = self._nmea_to_decimal(parts[4], parts[5])
                sats = int(parts[7]) if parts[7] else 0
                hdop = float(parts[8]) if parts[8] else 99.0
                alt = float(parts[9]) if parts[9] else 0.0
                prev = self._last_fix
                speed = prev.speed_mps if prev else 0.0
                heading = prev.heading_deg if prev else 0.0
                return GPSFix(lat, lon, alt, speed, heading,
                              sats, hdop, time.time(), True)
        except (ValueError, IndexError):
            return None
        return None

    @staticmethod
    def _nmea_to_decimal(coord: str, direction: str) -> float:
        """Convert NMEA ddmm.mmmm format to decimal degrees."""
        if not coord:
            return 0.0
        dot = coord.find('.')
        deg_len = dot - 2
        degrees = float(coord[:deg_len])
        minutes = float(coord[deg_len:])
        decimal = degrees + minutes / 60.0
        if direction in ('S', 'W'):
            decimal = -decimal
        return decimal


def bearing_between(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute initial bearing from point 1 to point 2. Returns 0-360 deg."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360


def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two GPS points in meters."""
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam/2)**2
    return 2 * R * math.asin(math.sqrt(a))


if __name__ == "__main__":
    gps = GPSReader()
    if gps.start():
        print("Reading GPS for 30 seconds...")
        for _ in range(30):
            fix = gps.get_fix()
            if fix and fix.valid:
                print(f"lat={fix.latitude:.6f} lon={fix.longitude:.6f} "
                      f"sats={fix.satellites} hdop={fix.hdop:.1f} "
                      f"speed={fix.speed_mps:.2f}m/s heading={fix.heading_deg:.1f}")
            else:
                print("No fix yet (waiting for satellites)...")
            time.sleep(1)
        gps.stop()
