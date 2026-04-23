#!/usr/bin/env python3
"""
Main Integration Script (Pi side)

Architecture:
    - Laptop sends commands via TCP (manual or ML-generated)
    - Pi reads ultrasonic sensors (via Arduino) + GPS
    - Pi runs safety_governor as a hard-override layer
    - Pi sends sensor + GPS status back to laptop (for ML inference)
"""

import time
import signal
from datetime import datetime

import config
from motor_controller import MotorController
from steering_controller import SteeringController
from obstacle_monitor import ObstacleMonitor
from remote_server import RemoteServer
from safety_governor import SafetyGovernor, SafetyViolation

try:
    from gps_reader import GPSReader
    GPS_ENABLED = True
except ImportError:
    GPS_ENABLED = False


class VehicleController:
    def __init__(self):
        self.motor = MotorController()
        self.steering = SteeringController()
        self.sensors = ObstacleMonitor()
        self.server = RemoteServer()
        self.safety = SafetyGovernor()
        self.gps = GPSReader() if GPS_ENABLED else None

        self.running = False
        self.current_drive = config.CMD_STOP
        self.current_steer = config.CMD_STEER_STOP
        self.current_speed = 0
        self.last_violation = SafetyViolation.NONE

    def initialize(self):
        print("=" * 60)
        print("VEHICLE CONTROL SYSTEM - Pi Side")
        print("=" * 60)
        try:
            print("[1/5] Motor controller...")
            self.motor.setup()
            print("[2/5] Steering controller...")
            self.steering.setup()
            print("[3/5] Obstacle monitor (Arduino)...")
            self.sensors.start_monitoring()
            print("[4/5] Remote server...")
            self.server.start_server()
            print("[5/5] GPS reader...")
            if self.gps:
                self.gps.start()
            print("=" * 60)
            print(f"READY. Waiting for laptop on port {config.SERVER_PORT}")
            print("=" * 60)
            return True
        except Exception as e:
            print(f"Init failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def start(self):
        if not self.initialize():
            return
        self.running = True
        signal.signal(signal.SIGINT, self._sig_handler)
        signal.signal(signal.SIGTERM, self._sig_handler)
        self._control_loop()

    def _control_loop(self):
        loop_count = 0
        while self.running:
            loop_start = time.time()
            try:
                # 1. Read commands from laptop
                drive_cmd = self.server.get_latest_command()
                steer_cmd = self.server.get_latest_steer()
                requested_speed = self.server.get_latest_speed()
                cmd_timestamp = self.server.get_command_timestamp()

                # 2. Read sensors
                distances = self.sensors.get_all_distances()

                # 3. Safety governor (HARD override)
                decision = self.safety.check(distances, drive_cmd, cmd_timestamp)
                self.last_violation = decision.violation

                if decision.is_safe:
                    self.current_drive = drive_cmd
                    self.current_steer = steer_cmd
                    self.current_speed = requested_speed
                else:
                    self.current_drive = decision.override_drive
                    self.current_steer = decision.override_steer
                    self.current_speed = decision.override_speed

                # 4. Apply to motors
                self.motor.set_speed(self.current_drive, self.current_speed)
                self.steering.set_direction(self.current_steer)

                # 5. Build and send status (includes everything ML needs)
                status = self._build_status(distances)
                if self.server.is_connected():
                    self.server.send_status(status)

                # 6. Periodic console log
                loop_count += 1
                if loop_count % 10 == 0:
                    self._log(status)

                # 7. Maintain loop rate
                elapsed = time.time() - loop_start
                time.sleep(max(0, config.CONTROL_LOOP_INTERVAL - elapsed))

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[loop error] {e}")
                time.sleep(0.1)

    def _build_status(self, distances):
        status = {
            'current_command': self.current_drive,
            'current_steer': self.current_steer,
            'actual_speed': self.current_speed,
            'distances': distances,
            'min_distance_front': self._min(distances, ['FL', 'FR', 'FW']),
            'min_distance_back': self._min(distances, ['BC']),
            'min_distance_left': self._min(distances, ['LS']),
            'min_distance_right': self._min(distances, ['RS']),
            'safety_violation': self.last_violation.value,
            'connected': self.server.is_connected(),
        }
        if self.gps:
            fix = self.gps.get_fix()
            if fix and fix.valid:
                status['gps'] = {
                    'lat': fix.latitude, 'lon': fix.longitude,
                    'alt': fix.altitude, 'speed_mps': fix.speed_mps,
                    'heading_deg': fix.heading_deg,
                    'satellites': fix.satellites, 'hdop': fix.hdop,
                    'valid': True,
                }
            else:
                status['gps'] = {'valid': False}
        return status

    @staticmethod
    def _min(distances, sensors):
        vals = [distances.get(s, 0) for s in sensors
                if config.MIN_SENSOR_DISTANCE <= distances.get(s, 0) <= config.MAX_SENSOR_DISTANCE]
        return min(vals) if vals else 0.0

    def _log(self, status):
        ts = datetime.now().strftime("%H:%M:%S")
        conn = "C" if status['connected'] else "D"
        v = self.last_violation.value
        d = status.get('distances', {})
        print(f"[{ts}] {conn} | drive={self.current_drive:8s} "
              f"steer={self.current_steer:10s} spd={self.current_speed:3d}% | "
              f"FL={d.get('FL',0):5.1f} FR={d.get('FR',0):5.1f} FW={d.get('FW',0):5.1f} "
              f"BC={d.get('BC',0):5.1f} LS={d.get('LS',0):5.1f} RS={d.get('RS',0):5.1f} | "
              f"safety={v}")

    def _sig_handler(self, signum, frame):
        print("\nShutdown...")
        self.stop()

    def stop(self):
        if not self.running:
            return
        self.running = False
        print("Stopping motors...")
        try: self.motor.stop()
        except Exception: pass
        try: self.steering.stop()
        except Exception: pass
        try: self.server.stop_server()
        except Exception: pass
        try: self.sensors.stop_monitoring()
        except Exception: pass
        if self.gps:
            try: self.gps.stop()
            except Exception: pass
        try:
            self.motor.cleanup()
            self.steering.cleanup()
        except Exception: pass
        print("Shutdown complete.")


def main():
    c = VehicleController()
    try:
        c.start()
    finally:
        c.stop()


if __name__ == "__main__":
    main()
