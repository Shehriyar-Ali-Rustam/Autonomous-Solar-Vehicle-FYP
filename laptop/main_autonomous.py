#!/usr/bin/env python3
"""
Main autonomous loop (laptop side).

Loop:
    1. Capture webcam frame
    2. Receive sensor + GPS status from Pi
    3. Run Predictor (ONNX) -> action_id, confidence
    4. If confidence < threshold -> send STOP (Pi safety_governor will also check)
    5. Convert action_id -> drive/steer/speed command
    6. Send to Pi via TCP

Usage:
    python laptop/main_autonomous.py --pi 192.168.1.100 --model laptop/models/model.onnx
"""

import argparse
import json
import os
import select
import socket
import sys
import termios
import threading
import time
import tty

import cv2

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from vision.camera import Camera
from ml.actions import (ACTION_NAMES, STOP, action_to_pi_command)
from ml.inference import Predictor


class PiClient:
    """Same as in record.py but lighter (no manual command state)."""
    def __init__(self, ip: str, port: int = 5555):
        self.ip = ip; self.port = port
        self.sock = None
        self.connected = False
        self.running = True
        self.status = None
        self.status_lock = threading.Lock()

    def connect(self) -> bool:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5.0)
            s.connect((self.ip, self.port))
            s.settimeout(0.5)
            self.sock = s; self.connected = True
            return True
        except Exception as e:
            print(f"Connect failed: {e}")
            return False

    def send(self, cmd: dict):
        if not self.connected: return
        try:
            self.sock.sendall(json.dumps(cmd).encode('utf-8'))
        except Exception:
            self.connected = False

    def _receiver(self):
        buf = ""
        while self.running and self.connected:
            try:
                data = self.sock.recv(8192)
                if not data:
                    self.connected = False; return
                buf += data.decode('utf-8')
                while '\n' in buf:
                    line, buf = buf.split('\n', 1)
                    try:
                        with self.status_lock:
                            self.status = json.loads(line)
                    except json.JSONDecodeError:
                        pass
            except socket.timeout:
                continue
            except Exception:
                self.connected = False; return

    def start(self):
        threading.Thread(target=self._receiver, daemon=True).start()

    def get_status(self):
        with self.status_lock:
            return dict(self.status) if self.status else None

    def close(self):
        self.running = False
        if self.sock:
            try: self.sock.close()
            except Exception: pass


class AutonomousDriver:
    def __init__(self, pi_ip: str, model_path: str, camera_id: int,
                 conf_threshold: float = 0.55, loop_hz: float = 10.0):
        self.client = PiClient(pi_ip)
        self.camera = Camera(device=camera_id)
        self.predictor = Predictor(model_path)
        self.conf_threshold = conf_threshold
        self.loop_interval = 1.0 / loop_hz
        self.running = True
        self.paused = True   # start paused until user presses 'g'
        self.prev_action = STOP

    def run(self):
        print(f"Connecting to Pi @ {self.client.ip}...")
        if not self.client.connect():
            return
        self.client.start()
        if not self.camera.start():
            return

        print("=" * 55)
        print(" AUTONOMOUS ML DRIVER")
        print("=" * 55)
        print(" G      = GO (start autonomous driving)")
        print(" SPACE  = PAUSE / STOP")
        print(" Q/ESC  = Quit")
        print("=" * 55)

        old = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())
            last_pred_log = 0.0
            while self.running:
                loop_start = time.time()

                if select.select([sys.stdin], [], [], 0.0)[0]:
                    ch = sys.stdin.read(1)
                    if ch in ('q', 'Q', '\x03'):
                        break
                    if ch in ('g', 'G'):
                        self.paused = False
                    if ch == ' ':
                        self.paused = True
                        self.client.send({'command': 'STOP', 'steer': 'STEER_STOP', 'speed': 0})

                frame, _ = self.camera.read()
                status = self.client.get_status()

                if self.paused:
                    time.sleep(self.loop_interval)
                    continue

                if frame is None or status is None:
                    time.sleep(self.loop_interval)
                    continue

                dists = status.get('distances', {})
                sensors = {k: float(dists.get(k, 0)) for k in ['FL','FR','FW','BC','LS','RS']}
                gps = status.get('gps') or {}
                gps_valid = int(gps.get('valid', 0))
                gps_speed = float(gps.get('speed_mps', 0.0))
                gps_heading = float(gps.get('heading_deg', 0.0))

                # Inference
                pred = self.predictor.predict(
                    frame, sensors, gps_valid, gps_speed, gps_heading, self.prev_action)

                action_id = pred['action_id']
                conf = pred['confidence']

                # Low-confidence fallback: STOP
                if conf < self.conf_threshold:
                    action_id = STOP

                # Convert to Pi command
                cmd = action_to_pi_command(action_id)
                self.client.send(cmd)
                self.prev_action = action_id

                # Throttled log
                now = time.time()
                if now - last_pred_log > 0.5:
                    last_pred_log = now
                    print(f"\r[ML] {ACTION_NAMES[action_id]:15s} conf={conf*100:5.1f}% | "
                          f"F={min([v for v in [sensors['FL'],sensors['FR'],sensors['FW']] if v>2] or [0]):5.1f}cm  "
                          f"sat={status.get('gps',{}).get('satellites',0)} "
                          f"safety={status.get('safety_violation','')}", end='', flush=True)

                elapsed = time.time() - loop_start
                time.sleep(max(0, self.loop_interval - elapsed))

        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
            self.client.send({'command': 'STOP', 'steer': 'STEER_STOP', 'speed': 0})
            time.sleep(0.3)
            self.client.close()
            self.camera.stop()
            cv2.destroyAllWindows()
            print("\nStopped.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pi', required=True, help='Pi IP address')
    p.add_argument('--model', required=True, help='Path to .onnx or .pt model')
    p.add_argument('--camera', type=int, default=0)
    p.add_argument('--conf', type=float, default=0.55, help='Min confidence to act')
    p.add_argument('--hz', type=float, default=10.0)
    args = p.parse_args()

    AutonomousDriver(args.pi, args.model, args.camera,
                     conf_threshold=args.conf, loop_hz=args.hz).run()


if __name__ == "__main__":
    main()
