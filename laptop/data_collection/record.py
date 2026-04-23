#!/usr/bin/env python3
"""
Record training data while driving the car manually (WASD + arrows).

Every ~5 Hz:
    1. Capture webcam frame
    2. Read latest sensors + GPS from the Pi (received via TCP)
    3. Record current action (what the driver is doing)
    4. Save: image to data/images/, row to data/dataset.csv

Usage:
    python -m laptop.data_collection.record --pi 192.168.1.100 --out laptop/data
"""

import argparse
import csv
import json
import os
import select
import socket
import sys
import termios
import threading
import time
import tty
import uuid

import cv2

# Make `laptop.*` importable when running as a module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from laptop.vision.camera import Camera
from laptop.ml.actions import (FORWARD, SLOW_DOWN, TURN_LEFT, TURN_RIGHT, STOP,
                               REVERSE_LEFT, REVERSE_RIGHT, REVERSE, ACTION_NAMES,
                               manual_to_action)


class PiClient:
    """TCP client: sends drive/steer commands, receives status."""
    def __init__(self, ip: str, port: int = 5555):
        self.ip = ip
        self.port = port
        self.sock = None
        self.connected = False
        self.running = True
        self.status = None
        self.lock = threading.Lock()
        self.drive = 'STOP'
        self.steer = 'STEER_STOP'
        self.speed = 50
        self.cmd_lock = threading.Lock()

    def connect(self) -> bool:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5.0)
            s.connect((self.ip, self.port))
            s.settimeout(0.5)
            self.sock = s
            self.connected = True
            return True
        except Exception as e:
            print(f"Connect failed: {e}")
            return False

    def _sender(self):
        while self.running and self.connected:
            with self.cmd_lock:
                d, s, sp = self.drive, self.steer, self.speed
            try:
                msg = json.dumps({'command': d, 'steer': s, 'speed': sp})
                self.sock.sendall(msg.encode('utf-8'))
            except Exception:
                self.connected = False
                return
            # Auto-release steering so user must keep pressing
            with self.cmd_lock:
                if self.steer in ('LEFT', 'RIGHT'):
                    self.steer = 'STEER_STOP'
            time.sleep(0.2)

    def _receiver(self):
        buf = ""
        while self.running and self.connected:
            try:
                data = self.sock.recv(8192)
                if not data:
                    self.connected = False
                    return
                buf += data.decode('utf-8')
                while '\n' in buf:
                    line, buf = buf.split('\n', 1)
                    try:
                        with self.lock:
                            self.status = json.loads(line)
                    except json.JSONDecodeError:
                        pass
            except socket.timeout:
                continue
            except Exception:
                self.connected = False
                return

    def start(self):
        threading.Thread(target=self._sender, daemon=True).start()
        threading.Thread(target=self._receiver, daemon=True).start()

    def set_command(self, drive=None, steer=None, speed=None):
        with self.cmd_lock:
            if drive  is not None: self.drive = drive
            if steer  is not None: self.steer = steer
            if speed  is not None: self.speed = speed

    def get_status(self):
        with self.lock:
            return dict(self.status) if self.status else None

    def close(self):
        self.running = False
        if self.sock:
            try: self.sock.close()
            except Exception: pass


class Recorder:
    def __init__(self, pi_ip: str, out_dir: str, device: int = 0):
        self.out_dir = out_dir
        self.images_dir = os.path.join(out_dir, 'images')
        os.makedirs(self.images_dir, exist_ok=True)
        self.csv_path = os.path.join(out_dir, 'dataset.csv')
        self.csv_file = None
        self.csv_writer = None
        self.camera = Camera(device=device)
        self.client = PiClient(pi_ip)
        self.recording = False
        self.session_id = time.strftime('%Y%m%d_%H%M%S')
        self.samples_written = 0
        self.prev_action = STOP

    def _open_csv(self):
        is_new = not os.path.exists(self.csv_path)
        self.csv_file = open(self.csv_path, 'a', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        if is_new:
            self.csv_writer.writerow([
                'session', 'timestamp', 'frame_path',
                'FL', 'FR', 'FW', 'BC', 'LS', 'RS',
                'gps_valid', 'gps_speed', 'gps_heading',
                'drive', 'steer', 'speed',
                'prev_action', 'action_label', 'action_name',
            ])

    def _write_sample(self, frame, status, drive, steer, speed):
        dists = status.get('distances', {}) if status else {}
        gps = status.get('gps', {}) if status else {}

        action_id = manual_to_action(drive, steer)
        frame_name = f"{self.session_id}_{uuid.uuid4().hex[:8]}.jpg"
        frame_path = os.path.join(self.images_dir, frame_name)
        cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

        self.csv_writer.writerow([
            self.session_id, time.time(), frame_path,
            dists.get('FL', 0), dists.get('FR', 0), dists.get('FW', 0),
            dists.get('BC', 0), dists.get('LS', 0), dists.get('RS', 0),
            int(gps.get('valid', 0)), gps.get('speed_mps', 0.0), gps.get('heading_deg', 0.0),
            drive, steer, speed,
            self.prev_action, action_id, ACTION_NAMES[action_id],
        ])
        self.csv_file.flush()
        self.prev_action = action_id
        self.samples_written += 1

    def run(self):
        print(f"Connecting to Pi...")
        if not self.client.connect():
            return
        self.client.start()
        print("Connected.")

        if not self.camera.start():
            return

        self._open_csv()
        print("\n" + "=" * 55)
        print(" DATA RECORDER")
        print("=" * 55)
        print(" WASD / Arrows = drive, SPACE = stop")
        print(" R = toggle recording (must be ON to save samples)")
        print(" +/- = speed  |  Q/ESC = quit")
        print("=" * 55)

        old = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())
            last_sample = 0.0
            SAMPLE_INTERVAL = 0.2  # 5 Hz

            while True:
                # Non-blocking keypress
                if select.select([sys.stdin], [], [], 0.0)[0]:
                    ch = sys.stdin.read(1)
                    self._handle_key(ch)
                    if ch in ('q', 'Q', '\x03', '\x1b'):
                        break

                # Read status + frame, possibly record
                now = time.time()
                if self.recording and (now - last_sample) >= SAMPLE_INTERVAL:
                    frame, _ = self.camera.read()
                    status = self.client.get_status()
                    if frame is not None and status is not None:
                        with self.client.cmd_lock:
                            d, s, sp = self.client.drive, self.client.steer, self.client.speed
                        self._write_sample(frame, status, d, s, sp)
                        last_sample = now

                # Live preview (small)
                frame, _ = self.camera.read()
                if frame is not None:
                    small = cv2.resize(frame, (320, 240))
                    rec = "REC" if self.recording else "   "
                    status_txt = f"{rec} samples={self.samples_written}"
                    cv2.putText(small, status_txt, (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 255) if self.recording else (0, 255, 0), 2)
                    cv2.imshow("Recorder", small)
                    cv2.waitKey(1)

                time.sleep(0.03)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
            self.client.set_command(drive='STOP', steer='STEER_STOP', speed=0)
            time.sleep(0.3)
            self.client.close()
            self.camera.stop()
            if self.csv_file:
                self.csv_file.close()
            cv2.destroyAllWindows()
            print(f"\nSaved {self.samples_written} samples to {self.csv_path}")

    def _handle_key(self, ch: str):
        if ch in ('w', 'W'):
            self.client.set_command(drive='FORWARD')
        elif ch in ('s', 'S'):
            self.client.set_command(drive='BACKWARD')
        elif ch in ('a',):
            self.client.set_command(steer='LEFT')
        elif ch in ('d',):
            self.client.set_command(steer='RIGHT')
        elif ch in ('x', 'X'):
            self.client.set_command(steer='STEER_STOP')
        elif ch == ' ':
            self.client.set_command(drive='STOP', steer='STEER_STOP')
        elif ch in ('r', 'R'):
            self.recording = not self.recording
        elif ch == '+':
            self.client.speed = min(100, self.client.speed + 10)
        elif ch == '-':
            self.client.speed = max(0, self.client.speed - 10)
        elif ch == '\x1b':
            # Arrow key sequence
            if select.select([sys.stdin], [], [], 0.01)[0]:
                ch2 = sys.stdin.read(1)
                if ch2 == '[' and select.select([sys.stdin], [], [], 0.01)[0]:
                    ch3 = sys.stdin.read(1)
                    if ch3 == 'A': self.client.set_command(drive='FORWARD')
                    elif ch3 == 'B': self.client.set_command(drive='BACKWARD')
                    elif ch3 == 'C': self.client.set_command(steer='RIGHT')
                    elif ch3 == 'D': self.client.set_command(steer='LEFT')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pi', required=True, help='Pi IP address')
    p.add_argument('--out', default='laptop/data', help='Output directory')
    p.add_argument('--camera', type=int, default=0, help='USB camera device index')
    args = p.parse_args()
    Recorder(args.pi, args.out, args.camera).run()


if __name__ == "__main__":
    main()
