#!/usr/bin/env python3
"""Main autonomous loop (laptop side).

Pipeline (10 Hz):
    1. Capture webcam frame (current + previous → frame stacking).
    2. Run YOLO in a background thread (heartbeat checked for staleness).
    3. Pull sensor + GPS status from Pi.
    4. Direction-aware vision hard-stop: if person/object close enough in the
       half of the frame the car is *moving toward*, force STOP. Works for
       forward AND reverse (when a back camera is connected, swap input).
    5. Predictor.predict(...) gives action_id + confidence.
    6. If confidence < threshold → STOP.
    7. Convert action → drive/steer/speed JSON, send to Pi via TCP.
    8. Log every decision (autonomous_logs/*.csv).

Safety: SIGINT/SIGTERM ALWAYS sends STOP to Pi before exit (graceful shutdown).
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
from typing import Optional

import cv2

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from utils.logger import get_logger
from utils.config import load_config
from vision.camera import Camera
from ml.actions import ACTION_NAMES, STOP, FORWARD, SLOW_DOWN, TURN_LEFT, TURN_RIGHT, REVERSE_LEFT, REVERSE_RIGHT, REVERSE
from ml.actions import action_to_pi_command
from ml.inference import Predictor

try:
    from vision.object_detector import ObjectDetector
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

log = get_logger(__name__)
CFG = load_config()


# ===== TCP client =========================================================
class PiClient:
    def __init__(self, ip: str, port: int = 5555):
        self.ip = ip
        self.port = port
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
            self.sock = s
            self.connected = True
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
        """Send STOP repeatedly to make sure Pi gets it (defensive shutdown)."""
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
                    self.connected = False
                    return
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
                self.connected = False
                return

    def start(self) -> None:
        threading.Thread(target=self._receiver, daemon=True).start()

    def get_status(self) -> Optional[dict]:
        with self.status_lock:
            return dict(self.status) if self.status else None

    def close(self) -> None:
        self.running = False
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass


# ===== Vision-based hard-stop logic =======================================
# How much of the frame area must a person/obstacle occupy to trigger a stop.
PEDESTRIAN_AREA_THRESHOLD = float(CFG.get('pedestrian', {}).get('area_threshold', 0.06))
# If False, only stop for centre-position. If True, any position counts.
PEDESTRIAN_ANY_POSITION  = bool(CFG.get('pedestrian', {}).get('any_position', True))
# Bbox-centre x in [-1..1]. Half-width threshold for "in this side":
HALF_WIDTH = 0.6   # ±0.6 covers most of the screen, gives slight peripheral pad


def _hardstop_reason(intent_action: int, yolo: dict) -> str:
    """Direction-aware vision hard-stop.

    Triggered when:
      * the model wants to move (not STOP)
      * AND YOLO detects a *close* person or object in the half of the frame
        we are heading toward (or any half if PEDESTRIAN_ANY_POSITION).

    Returns reason string if triggered, '' otherwise.
    """
    if intent_action == STOP:
        return ''
    person = bool(yolo.get('person_detected', 0))
    obj    = bool(yolo.get('object_detected', 0))
    if not (person or obj):
        return ''
    area = float(yolo.get('nearest_area_ratio', 0.0))
    if area < PEDESTRIAN_AREA_THRESHOLD:
        return ''

    # Direction the car is going to move (laterally):
    # forward straight & reverse straight: any-position is risky
    # forward+left/turn_left/reverse+left: prioritise left half
    # forward+right/turn_right/reverse+right: prioritise right half
    pos_x = float(yolo.get('nearest_position_x', 0.0))   # -1..1

    is_left_action  = intent_action in (TURN_LEFT, REVERSE_LEFT)
    is_right_action = intent_action in (TURN_RIGHT, REVERSE_RIGHT)

    if PEDESTRIAN_ANY_POSITION or (not is_left_action and not is_right_action):
        in_path = abs(pos_x) <= HALF_WIDTH
    elif is_left_action:
        in_path = pos_x <= 0.2     # anything in centre or left half
    else:  # is_right_action
        in_path = pos_x >= -0.2

    if not in_path:
        return ''
    label = 'PERSON' if person else 'OBSTACLE'
    return f"{label}_CLOSE area={area:.2f} pos_x={pos_x:+.2f}"


# ===== Autonomous driver =================================================
class AutonomousDriver:
    def __init__(self, pi_ip: str, model_path: str, camera_id: int,
                 conf_threshold: float = 0.55, loop_hz: float = 10.0,
                 use_yolo: bool = True, log_dir: Optional[str] = None):
        self.client = PiClient(pi_ip)
        self.camera = Camera(device=camera_id)
        self.predictor = Predictor(model_path)
        self.conf_threshold = conf_threshold
        self.loop_interval = 1.0 / loop_hz
        self.running = True
        self.paused = True
        self.prev_action = STOP
        self.prev_frame = None        # for frame stacking

        # YOLO + heartbeat
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
                self.yolo = None

        # Logging dir
        self.log_dir = log_dir
        self.log_file = None
        self.log_writer = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir,
                                    f"autonomous_{time.strftime('%Y%m%d_%H%M%S')}.csv")
            self.log_file = open(log_path, 'w', newline='')
            self.log_writer = csv.writer(self.log_file)
            self.log_writer.writerow([
                'timestamp', 'paused',
                'FL', 'FR', 'FW', 'BC', 'LS', 'RS',
                'min_front', 'min_back',
                'gps_valid', 'gps_speed', 'gps_heading',
                'yolo_person', 'yolo_object', 'yolo_area', 'yolo_pos_x', 'yolo_count',
                'yolo_stale',
                'ml_action', 'ml_confidence',
                'final_action', 'override_reason',
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
        """Background thread: ~5Hz YOLO. Updates heartbeat timestamp."""
        while self.running:
            try:
                if self.yolo is None:
                    time.sleep(1.0)
                    continue
                frame, _ = self.camera.read()
                if frame is None:
                    time.sleep(0.1)
                    continue
                feats = self.yolo.extract_features(frame)
                # Continuous bbox-centre x in -1..1
                pos_x = 0.0
                if feats.num_objects > 0:
                    # Use the area-largest detection's bbox centre.
                    h, w = frame.shape[:2]
                    dets = self.yolo.detect(frame)
                    if dets:
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
                log.warning(f"[YOLO] thread iteration failed: {e}")
                time.sleep(0.1)
            time.sleep(0.2)

    def _get_yolo(self) -> dict:
        """Returns YOLO snapshot; returns conservative all-zero with high
        person_detected==0 if YOLO is stale. The 'stale' flag is logged so
        we know when this happened.
        """
        with self.yolo_lock:
            snap = dict(self.latest_yolo)
            last = self.yolo_last_update
        if self.yolo is not None and (time.time() - last) > self.yolo_staleness_secs:
            # YOLO heartbeat lost — treat as conservative: assume hazard.
            snap = self._empty_yolo()
            snap['_stale'] = True
            snap['object_detected'] = 1     # force the model into caution
            snap['person_detected'] = 0
            snap['nearest_area_ratio'] = 1.0
        else:
            snap['_stale'] = False
        return snap

    def _signal_handler(self, signum, _frame) -> None:
        log.warning(f"Signal {signum} received → graceful shutdown")
        self.running = False
        try:
            self.client.safe_stop()
        except Exception:
            pass

    def run(self) -> None:
        # Install signal handlers FIRST so even early failures stop the car.
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        log.info(f"Connecting to Pi @ {self.client.ip}...")
        if not self.client.connect():
            return
        self.client.start()
        if not self.camera.start():
            self.client.close()
            return

        if self.yolo:
            threading.Thread(target=self._yolo_thread, daemon=True).start()

        log.info("=" * 55)
        log.info(" AUTONOMOUS ML DRIVER")
        log.info("=" * 55)
        log.info(f" YOLO: {'ON' if self.yolo else 'OFF'}")
        log.info(f" Vision hard-stop: ON (area>{PEDESTRIAN_AREA_THRESHOLD}, any_position={PEDESTRIAN_ANY_POSITION})")
        log.info(f" Logging: {'ON' if self.log_file else 'OFF'}")
        log.info(" G=GO  SPACE=PAUSE  Q/ESC=Quit")
        log.info("=" * 55)

        old = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())
            last_print = 0.0
            while self.running:
                loop_start = time.time()
                try:
                    self._loop_once(last_print, loop_start)
                except Exception as e:
                    # Critical loop guard: never let an exception kill the loop
                    # without first sending STOP. We log & continue.
                    log.error(f"loop iteration crashed: {e}", exc_info=True)
                    self.client.safe_stop()

                last_print = time.time() - loop_start  # placeholder; reset below
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

    def _loop_once(self, last_print, loop_start) -> None:
        # Keyboard
        if select.select([sys.stdin], [], [], 0.0)[0]:
            ch = sys.stdin.read(1)
            if ch in ('q', 'Q', '\x03'):
                self.running = False
                return
            if ch in ('g', 'G'):
                self.paused = False
            if ch == ' ':
                self.paused = True
                self.client.safe_stop()

        frame, _ = self.camera.read()
        status = self.client.get_status()
        yolo = self._get_yolo()

        if self.paused:
            self._log_row(status, yolo, None, 0.0, STOP, 'PAUSED', None)
            return

        if frame is None or status is None:
            # Don't try to predict without inputs. STOP defensively.
            self.client.safe_stop()
            self._log_row(status, yolo, None, 0.0, STOP, 'NO_INPUT', None)
            return

        dists = status.get('distances', {})
        sensors = {k: float(dists.get(k, 0)) for k in ['FL', 'FR', 'FW', 'BC', 'LS', 'RS']}
        gps = status.get('gps') or {}
        gps_valid = int(gps.get('valid', 0))
        gps_speed = float(gps.get('speed_mps', 0.0))
        gps_heading = float(gps.get('heading_deg', 0.0))

        # Inference
        try:
            pred = self.predictor.predict(
                frame, sensors, gps_valid, gps_speed, gps_heading,
                self.prev_action, yolo, prev_frame_bgr=self.prev_frame)
        except Exception as e:
            log.error(f"inference failed: {e}", exc_info=True)
            self.client.safe_stop()
            self._log_row(status, yolo, None, 0.0, STOP, f'INFERENCE_ERROR:{e}', None)
            self.prev_frame = frame
            return

        ml_action = pred['action_id']
        conf = pred['confidence']

        final_action = ml_action
        override_reason = ''

        # Vision hard-stop (direction-aware)
        v_reason = _hardstop_reason(ml_action, yolo)
        if v_reason:
            final_action = STOP
            override_reason = v_reason
        # YOLO staleness — be conservative
        elif yolo.get('_stale'):
            final_action = STOP
            override_reason = 'YOLO_STALE'
        # Confidence gate
        elif conf < self.conf_threshold:
            final_action = STOP
            override_reason = f'LOW_CONFIDENCE {conf:.2f}'

        cmd = action_to_pi_command(final_action)
        self.client.send(cmd)
        self.prev_action = final_action
        self.prev_frame = frame

        self._log_row(status, yolo, ml_action, conf, final_action, override_reason, cmd)

        # Console log
        now = time.time()
        if now - getattr(self, '_last_console_log', 0.0) > 0.5:
            self._last_console_log = now
            flag = f' ⚠{override_reason}' if override_reason else ''
            log.info(f"[ML] {ACTION_NAMES[ml_action]:13s} c={conf*100:5.1f}% "
                     f"→ {ACTION_NAMES[final_action]:13s} | "
                     f"yolo:p={yolo.get('person_detected',0)} o={yolo.get('num_objects',0)} "
                     f"a={yolo.get('nearest_area_ratio',0):.2f} x={yolo.get('nearest_position_x',0):+.2f}"
                     f"{flag}")

    def _log_row(self, status, yolo, ml_action, conf, final_action, reason, cmd) -> None:
        if not self.log_writer:
            return
        if status is None:
            status = {}
        d = status.get('distances', {})
        g = status.get('gps', {})
        cmd = cmd or {}
        try:
            self.log_writer.writerow([
                time.time(), self.paused,
                d.get('FL', 0), d.get('FR', 0), d.get('FW', 0),
                d.get('BC', 0), d.get('LS', 0), d.get('RS', 0),
                status.get('min_distance_front', 0), status.get('min_distance_back', 0),
                int(g.get('valid', 0)), g.get('speed_mps', 0.0), g.get('heading_deg', 0.0),
                yolo.get('person_detected', 0), yolo.get('object_detected', 0),
                yolo.get('nearest_area_ratio', 0.0),
                yolo.get('nearest_position_x', 0.0),
                yolo.get('num_objects', 0),
                int(bool(yolo.get('_stale', False))),
                ACTION_NAMES[ml_action] if ml_action is not None else '',
                conf,
                ACTION_NAMES[final_action] if final_action is not None else '',
                reason,
                cmd.get('command', ''), cmd.get('steer', ''), cmd.get('speed', 0),
                status.get('safety_violation', ''),
            ])
            self.log_file.flush()
        except Exception:
            pass


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--pi', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--camera', type=int,
                   default=CFG.get('camera', {}).get('device', 0))
    p.add_argument('--conf', type=float,
                   default=CFG.get('model', {}).get('conf_threshold', 0.55))
    p.add_argument('--hz', type=float,
                   default=CFG.get('model', {}).get('loop_hz', 10.0))
    p.add_argument('--no-yolo', action='store_true')
    p.add_argument('--log-dir',
                   default=os.path.join(os.path.dirname(__file__), 'autonomous_logs'))
    args = p.parse_args()

    AutonomousDriver(
        args.pi, args.model, args.camera,
        conf_threshold=args.conf, loop_hz=args.hz,
        use_yolo=not args.no_yolo, log_dir=args.log_dir,
    ).run()


if __name__ == "__main__":
    main()
