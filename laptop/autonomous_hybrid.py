#!/usr/bin/env python3
"""Hybrid sensor + YOLO autonomous controller (no ML model needed).

Decisions are made by pure rules over ultrasonic distances and YOLO output:

  * Person/object detected close (any position)        -> STOP
  * Front blocked (< FRONT_EMERGENCY_CM)               -> reverse toward clearer side
  * Front getting close (< FRONT_SLOW_CM):
        - clearer side has > SIDE_CLEAR_CM             -> TURN toward that side
        - otherwise                                    -> SLOW_DOWN
  * Front fairly clear (>= FRONT_CRUISE_CM)            -> FORWARD
  * In between                                         -> SLOW_DOWN

Safety stack still active:
  * Pi safety_governor blocks at 25cm front / 30cm rear
  * Direction-aware vision hard-stop (any side)
  * Graceful SIGINT/SIGTERM = STOP

Use this when you want a working autonomous demo TODAY without training data
quality issues. The ML model is still present in the project; this is a
parallel control mode.

Usage:
    python autonomous_hybrid.py --pi 192.168.100.30 --camera 1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import select
import signal
import socket
import sys
import termios
import threading
import time
import tty
from typing import Optional, Tuple

import cv2

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from utils.logger import get_logger
from utils.config import load_config
from vision.camera import Camera
from ml.actions import (ACTION_NAMES, FORWARD, SLOW_DOWN, TURN_LEFT, TURN_RIGHT,
                        STOP, REVERSE_LEFT, REVERSE_RIGHT, REVERSE,
                        action_to_pi_command)

try:
    from vision.object_detector import ObjectDetector
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

log = get_logger(__name__)
CFG = load_config()


# ===== Decision thresholds (cm) =========================================
# Override via config.yaml -> hybrid: section
HY_CFG = CFG.get('hybrid', {}) if isinstance(CFG.get('hybrid'), dict) else {}
FRONT_EMERGENCY_CM  = float(HY_CFG.get('front_emergency_cm', 35.0))
FRONT_SLOW_CM       = float(HY_CFG.get('front_slow_cm',      90.0))
FRONT_CRUISE_CM     = float(HY_CFG.get('front_cruise_cm',   150.0))
SIDE_CLEAR_CM       = float(HY_CFG.get('side_clear_cm',      40.0))
BACK_EMERGENCY_CM   = float(HY_CFG.get('back_emergency_cm',  30.0))
PERSON_AREA_THRESHOLD = float(CFG.get('pedestrian', {}).get('area_threshold', 0.06))
SENSOR_VALID_MIN_CM = float(CFG.get('sensors', {}).get('min_distance_cm', 2.0))
SENSOR_VALID_MAX_CM = float(CFG.get('sensors', {}).get('max_distance_cm', 400.0))


def _valid(d: float) -> bool:
    return SENSOR_VALID_MIN_CM <= d <= SENSOR_VALID_MAX_CM


def _min_valid(values, default: float = 9999.0) -> float:
    valid = [v for v in values if _valid(v)]
    return min(valid) if valid else default


# ===== TCP client ========================================================
class PiClient:
    def __init__(self, ip: str, port: int = 5555):
        self.ip = ip; self.port = port
        self.sock: Optional[socket.socket] = None
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
            log.error(f"Connect failed: {e}")
            return False

    def send(self, cmd: dict) -> None:
        if not self.connected or self.sock is None:
            return
        try:
            self.sock.sendall(json.dumps(cmd).encode('utf-8'))
        except Exception:
            self.connected = False

    def safe_stop(self) -> None:
        for _ in range(3):
            try:
                self.send({'command': 'STOP', 'steer': 'STEER_STOP', 'speed': 0})
                time.sleep(0.05)
            except Exception:
                break

    def _receiver(self) -> None:
        buf = ""
        while self.running and self.connected and self.sock is not None:
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

    def start(self) -> None:
        threading.Thread(target=self._receiver, daemon=True).start()

    def get_status(self) -> Optional[dict]:
        with self.status_lock:
            return dict(self.status) if self.status else None

    def close(self) -> None:
        self.running = False
        if self.sock:
            try: self.sock.close()
            except Exception: pass


# ===== Decision logic ====================================================
def decide(sensors: dict, yolo: dict) -> Tuple[int, str]:
    """Pure-rule decision. Returns (action_id, reason_string)."""
    # 1. Vision-based hard stop (any direction): person/obj close
    if yolo.get('person_detected', 0):
        if yolo.get('nearest_area_ratio', 0.0) >= PERSON_AREA_THRESHOLD:
            return STOP, f"PERSON_CLOSE area={yolo.get('nearest_area_ratio',0):.2f}"

    fl = sensors.get('FL', 0); fr = sensors.get('FR', 0); fw = sensors.get('FW', 0)
    bc = sensors.get('BC', 0); ls = sensors.get('LS', 0); rs = sensors.get('RS', 0)

    fmin = _min_valid([fl, fr, fw])
    bmin = bc if _valid(bc) else 9999.0
    lmin = ls if _valid(ls) else 9999.0
    rmin = rs if _valid(rs) else 9999.0

    # 2. Front emergency: too close to obstacle ahead
    if fmin < FRONT_EMERGENCY_CM:
        # Back also blocked? we're pinned -> STOP
        if bmin < BACK_EMERGENCY_CM:
            return STOP, f"PINNED front={fmin:.0f} back={bmin:.0f}"
        # Back clear: reverse, curving toward the clearer side so we end up
        # facing more open space.
        if lmin > rmin:
            return REVERSE_RIGHT, f"BACK_OUT_RIGHT front={fmin:.0f}"
        if rmin > lmin:
            return REVERSE_LEFT, f"BACK_OUT_LEFT front={fmin:.0f}"
        return REVERSE, f"BACK_OUT front={fmin:.0f}"

    # 3. Front getting close: try to avoid by turning toward clearer side
    if fmin < FRONT_SLOW_CM:
        # Pick the side with more clearance, only if it's actually open enough.
        if lmin >= SIDE_CLEAR_CM and lmin >= rmin:
            return TURN_LEFT, f"AVOID_LEFT front={fmin:.0f} L={lmin:.0f}"
        if rmin >= SIDE_CLEAR_CM and rmin >= lmin:
            return TURN_RIGHT, f"AVOID_RIGHT front={fmin:.0f} R={rmin:.0f}"
        # Neither side clear enough: just slow down and hope for the best
        return SLOW_DOWN, f"NARROW front={fmin:.0f} L={lmin:.0f} R={rmin:.0f}"

    # 4. Front fairly clear: cruise
    if fmin >= FRONT_CRUISE_CM:
        return FORWARD, f"CRUISE front={fmin:.0f}"

    # 5. In-between: slow forward
    return SLOW_DOWN, f"OPEN_BUT_CAREFUL front={fmin:.0f}"


# ===== Driver ============================================================
class HybridDriver:
    def __init__(self, pi_ip: str, camera_id: int, loop_hz: float = 10.0,
                 use_yolo: bool = True, log_dir: Optional[str] = None):
        self.client = PiClient(pi_ip)
        self.camera = Camera(device=camera_id)
        self.loop_interval = 1.0 / loop_hz
        self.running = True
        self.paused = True

        self.yolo: Optional[ObjectDetector] = None
        self.yolo_lock = threading.Lock()
        self.latest_yolo = self._empty_yolo()
        self.yolo_last_update = 0.0
        self.yolo_staleness_secs = float(CFG.get('yolo', {}).get('staleness_seconds', 1.0))
        if use_yolo and YOLO_AVAILABLE:
            try:
                log.info("[YOLO] Loading...")
                self.yolo = ObjectDetector(
                    conf_threshold=float(CFG.get('yolo', {}).get('conf_threshold', 0.4)),
                    device='cpu',
                )
                log.info("[YOLO] Ready")
            except Exception as e:
                log.error(f"[YOLO] Failed: {e}")

        self.log_file = None
        self.log_writer = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir,
                                    f"hybrid_{time.strftime('%Y%m%d_%H%M%S')}.csv")
            self.log_file = open(log_path, 'w', newline='')
            self.log_writer = csv.writer(self.log_file)
            self.log_writer.writerow([
                'timestamp', 'paused',
                'FL', 'FR', 'FW', 'BC', 'LS', 'RS',
                'fmin', 'bmin', 'lmin', 'rmin',
                'yolo_person', 'yolo_object', 'yolo_area', 'yolo_pos_x',
                'action', 'reason',
                'sent_drive', 'sent_steer', 'sent_speed',
                'safety_violation',
            ])
            log.info(f"[Log] {log_path}")

    @staticmethod
    def _empty_yolo() -> dict:
        return {'person_detected': 0, 'object_detected': 0,
                'nearest_area_ratio': 0.0, 'nearest_position_x': 0.0,
                'num_objects': 0}

    def _yolo_thread(self) -> None:
        while self.running:
            try:
                if self.yolo is None:
                    time.sleep(1.0); continue
                frame, _ = self.camera.read()
                if frame is None:
                    time.sleep(0.1); continue
                feats = self.yolo.extract_features(frame)
                pos_x = 0.0
                if feats.num_objects > 0:
                    dets = self.yolo.detect(frame)
                    if dets:
                        h, w = frame.shape[:2]
                        x1, _, x2, _ = dets[0].bbox
                        cx = (x1 + x2) / 2.0
                        pos_x = (cx / w) * 2.0 - 1.0
                with self.yolo_lock:
                    self.latest_yolo = {
                        'person_detected': feats.person_detected,
                        'object_detected': feats.object_detected,
                        'nearest_area_ratio': feats.nearest_area_ratio,
                        'nearest_position_x': pos_x,
                        'num_objects': feats.num_objects,
                    }
                    self.yolo_last_update = time.time()
            except Exception as e:
                log.warning(f"YOLO iteration failed: {e}")
                time.sleep(0.1)
            time.sleep(0.2)

    def _get_yolo(self) -> dict:
        with self.yolo_lock:
            snap = dict(self.latest_yolo)
            last = self.yolo_last_update
        if self.yolo is not None and (time.time() - last) > self.yolo_staleness_secs:
            # YOLO heartbeat lost: fail safe — pretend a person is in front.
            snap = self._empty_yolo()
            snap['_stale'] = True
            snap['person_detected'] = 1
            snap['nearest_area_ratio'] = 1.0
        else:
            snap['_stale'] = False
        return snap

    def _signal_handler(self, signum, _frame) -> None:
        log.warning(f"Signal {signum} received → graceful shutdown")
        self.running = False
        try: self.client.safe_stop()
        except Exception: pass

    def run(self) -> None:
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        log.info(f"Connecting to Pi @ {self.client.ip}...")
        if not self.client.connect():
            return
        self.client.start()
        if not self.camera.start():
            self.client.close(); return

        if self.yolo:
            threading.Thread(target=self._yolo_thread, daemon=True).start()

        log.info("=" * 55)
        log.info(" HYBRID SENSOR + YOLO AUTONOMOUS DRIVER")
        log.info("=" * 55)
        log.info(f" Front emergency : <{FRONT_EMERGENCY_CM:.0f}cm  (reverse)")
        log.info(f" Front slow      : <{FRONT_SLOW_CM:.0f}cm  (turn / slow)")
        log.info(f" Front cruise    : >={FRONT_CRUISE_CM:.0f}cm (full forward)")
        log.info(f" Side clear      : >={SIDE_CLEAR_CM:.0f}cm (need this to turn that way)")
        log.info(f" YOLO            : {'ON' if self.yolo else 'OFF'}")
        log.info(" G=GO  SPACE=PAUSE  Q/ESC=Quit")
        log.info("=" * 55)

        old = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())
            last_console = 0.0
            while self.running:
                loop_start = time.time()
                try:
                    self._loop_once(last_console)
                except Exception as e:
                    log.error(f"loop crashed: {e}", exc_info=True)
                    self.client.safe_stop()
                last_console = time.time()
                elapsed = time.time() - loop_start
                time.sleep(max(0, self.loop_interval - elapsed))
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
            self.client.safe_stop()
            self.client.close()
            self.camera.stop()
            if self.log_file:
                try: self.log_file.close()
                except Exception: pass
            cv2.destroyAllWindows()
            log.info("Stopped.")

    def _loop_once(self, _last_console) -> None:
        if select.select([sys.stdin], [], [], 0.0)[0]:
            ch = sys.stdin.read(1)
            if ch in ('q', 'Q', '\x03'):
                self.running = False; return
            if ch in ('g', 'G'):
                self.paused = False
            if ch == ' ':
                self.paused = True
                self.client.safe_stop()

        status = self.client.get_status()
        yolo = self._get_yolo()

        if self.paused:
            self._log_row(status, yolo, STOP, 'PAUSED', None)
            return

        if status is None:
            self.client.safe_stop()
            self._log_row(status, yolo, STOP, 'NO_PI_STATUS', None)
            return

        dists = status.get('distances', {})
        sensors = {k: float(dists.get(k, 0)) for k in ['FL','FR','FW','BC','LS','RS']}

        action_id, reason = decide(sensors, yolo)

        # If YOLO is stale, decide() already saw a fake "person close" → STOP.
        # Annotate the reason for clarity.
        if yolo.get('_stale') and action_id == STOP:
            reason = 'YOLO_STALE'

        cmd = action_to_pi_command(action_id)
        self.client.send(cmd)

        self._log_row(status, yolo, action_id, reason, cmd)

        # Throttled console
        now = time.time()
        if now - getattr(self, '_last_print', 0.0) > 0.5:
            self._last_print = now
            log.info(f"[HYB] {ACTION_NAMES[action_id]:13s} | {reason:50s} | "
                     f"yolo:p={yolo.get('person_detected',0)} a={yolo.get('nearest_area_ratio',0):.2f}")

    def _log_row(self, status, yolo, action_id, reason, cmd) -> None:
        if not self.log_writer:
            return
        if status is None:
            status = {}
        d = status.get('distances', {})
        cmd = cmd or {}
        try:
            fmin = _min_valid([d.get('FL', 0), d.get('FR', 0), d.get('FW', 0)])
            bmin = d.get('BC', 0) if _valid(d.get('BC', 0)) else 0
            lmin = d.get('LS', 0) if _valid(d.get('LS', 0)) else 0
            rmin = d.get('RS', 0) if _valid(d.get('RS', 0)) else 0
            self.log_writer.writerow([
                time.time(), self.paused,
                d.get('FL', 0), d.get('FR', 0), d.get('FW', 0),
                d.get('BC', 0), d.get('LS', 0), d.get('RS', 0),
                fmin, bmin, lmin, rmin,
                yolo.get('person_detected', 0), yolo.get('object_detected', 0),
                yolo.get('nearest_area_ratio', 0.0),
                yolo.get('nearest_position_x', 0.0),
                ACTION_NAMES[action_id] if action_id is not None else '',
                reason,
                cmd.get('command', ''), cmd.get('steer', ''), cmd.get('speed', 0),
                status.get('safety_violation', ''),
            ])
            self.log_file.flush()
        except Exception:
            pass


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--pi', required=True, help='Pi IP address')
    p.add_argument('--camera', type=int,
                   default=CFG.get('camera', {}).get('device', 0))
    p.add_argument('--hz', type=float,
                   default=CFG.get('model', {}).get('loop_hz', 10.0))
    p.add_argument('--no-yolo', action='store_true')
    p.add_argument('--log-dir',
                   default=os.path.join(os.path.dirname(__file__), 'autonomous_logs'))
    args = p.parse_args()

    HybridDriver(args.pi, args.camera, loop_hz=args.hz,
                 use_yolo=not args.no_yolo, log_dir=args.log_dir).run()


if __name__ == "__main__":
    main()
