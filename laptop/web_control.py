#!/usr/bin/env python3
"""Web-based vehicle remote control + data recording.

Now uses semantic FORWARD/BACKWARD (Pi handles wiring inversion). Recorder
writes rows defensively (validated, flushed each row). YOLO computes
continuous position_x (-1..1), heartbeat-tracked for staleness.

Open: http://<laptop-ip>:8080
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import sys
import threading
import time
import uuid
from typing import Optional

import cv2
from flask import Flask, Response, jsonify, render_template_string, request

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from utils.logger import get_logger
from utils.config import load_config
from vision.camera import Camera
from ml.actions import (manual_to_action, ACTION_NAMES, STOP, FORWARD,
                         REVERSE, REVERSE_LEFT, REVERSE_RIGHT,
                         action_to_pi_command)
from nav.controller import NavController
# The hybrid decide() function chooses an action from sensors+YOLO.
# We use it as a SAFETY OVERRIDE for nav: if it returns STOP / REVERSE_* the
# nav action is replaced (a hazard was detected). Otherwise we keep the nav
# action so the car still steers toward its waypoint.
from autonomous_hybrid import decide as hybrid_decide

try:
    from vision.object_detector import ObjectDetector
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

log = get_logger(__name__)
CFG = load_config()

app = Flask(__name__)

# Globals
pi_client: Optional['PiClient'] = None
camera: Optional[Camera] = None
recorder: Optional['DataRecorder'] = None
yolo: Optional['ObjectDetector'] = None
yolo_lock = threading.Lock()
latest_yolo = {'person_detected': 0, 'object_detected': 0,
               'nearest_area_ratio': 0.0, 'nearest_position_x': 0.0,
               'num_objects': 0}
yolo_last_update = 0.0

# Navigation
nav = NavController()
nav_lock = threading.Lock()
nav_active = False               # True after user taps GO
nav_last_info: dict = {}


# ===== YOLO ===============================================================
def yolo_loop() -> None:
    global latest_yolo, yolo_last_update
    while True:
        try:
            if camera is None or yolo is None:
                time.sleep(0.5)
                continue
            frame, _ = camera.read()
            if frame is None:
                time.sleep(0.1)
                continue
            feats = yolo.extract_features(frame)
            pos_x = 0.0
            if feats.num_objects > 0:
                dets = yolo.detect(frame)
                if dets:
                    h, w = frame.shape[:2]
                    x1, _, x2, _ = dets[0].bbox
                    cx = (x1 + x2) / 2.0
                    pos_x = (cx / w) * 2.0 - 1.0
            with yolo_lock:
                latest_yolo = {
                    'person_detected': feats.person_detected,
                    'object_detected': feats.object_detected,
                    'nearest_area_ratio': feats.nearest_area_ratio,
                    'nearest_position_x': pos_x,
                    'num_objects': feats.num_objects,
                }
                yolo_last_update = time.time()
        except Exception as e:
            log.warning(f"yolo_loop iteration failed: {e}")
            time.sleep(0.2)
            continue
        time.sleep(0.2)


def get_yolo_snapshot() -> dict:
    with yolo_lock:
        snap = dict(latest_yolo)
        last = yolo_last_update
    stale_thresh = float(CFG.get('yolo', {}).get('staleness_seconds', 1.0))
    snap['_stale'] = (yolo is not None) and (time.time() - last > stale_thresh)
    return snap


# ===== Pi client ==========================================================
class PiClient:
    def __init__(self, ip: str, port: int = 5555):
        self.ip = ip
        self.port = port
        self.sock: Optional[socket.socket] = None
        self.connected = False
        self.running = True
        self.status = None
        self.status_lock = threading.Lock()
        self.cmd_lock = threading.Lock()
        # State sent to Pi every 100ms
        self.drive = 'STOP'
        self.steer = 'STEER_STOP'
        self.speed = 50

    def connect(self) -> bool:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5.0)
            s.connect((self.ip, self.port))
            s.settimeout(0.5)
            self.sock = s
            self.connected = True
            log.info(f"[PiClient] Connected to {self.ip}:{self.port}")
            return True
        except Exception as e:
            log.error(f"[PiClient] Connect failed: {e}")
            return False

    def _sender(self) -> None:
        while self.running and self.connected:
            try:
                with self.cmd_lock:
                    d, s, sp = self.drive, self.steer, self.speed
                msg = json.dumps({'command': d, 'steer': s, 'speed': sp})
                if self.sock is not None:
                    self.sock.sendall(msg.encode('utf-8'))
            except Exception:
                self.connected = False
                return
            time.sleep(0.1)   # 10 Hz keep-alive (matches Pi watchdog 0.5s)

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
        threading.Thread(target=self._sender, daemon=True).start()
        threading.Thread(target=self._receiver, daemon=True).start()

    def set_command(self, drive: Optional[str] = None,
                    steer: Optional[str] = None,
                    speed: Optional[int] = None) -> None:
        with self.cmd_lock:
            if drive is not None: self.drive = drive
            if steer is not None: self.steer = steer
            if speed is not None: self.speed = max(0, min(100, int(speed)))

    def get_state_snapshot(self) -> tuple:
        with self.cmd_lock:
            return self.drive, self.steer, self.speed

    def get_status(self) -> Optional[dict]:
        with self.status_lock:
            return dict(self.status) if self.status else None

    def safe_stop(self) -> None:
        self.set_command(drive='STOP', steer='STEER_STOP', speed=0)

    def close(self) -> None:
        self.running = False
        self.safe_stop()
        time.sleep(0.2)
        if self.sock:
            try: self.sock.close()
            except Exception: pass


# ===== Recorder ===========================================================
CSV_HEADER = [
    'session', 'timestamp', 'frame_path',
    'FL', 'FR', 'FW', 'BC', 'LS', 'RS',
    'gps_valid', 'gps_speed', 'gps_heading',
    'drive', 'steer', 'speed',
    'prev_action', 'action_label', 'action_name',
    'yolo_person', 'yolo_object', 'yolo_area', 'yolo_pos_x', 'yolo_count',
]


def _validate_row(row: list) -> bool:
    """Cheap sanity check before flushing a row to CSV."""
    if len(row) != len(CSV_HEADER):
        return False
    try:
        # All sensor + numeric fields must be cast-able
        for idx in (1, 3, 4, 5, 6, 7, 8, 10, 11, 16, 20, 21, 22):
            float(row[idx])
        # Drive must be one of known values
        if row[12] not in ('FORWARD', 'BACKWARD', 'STOP'):
            return False
        if row[13] not in ('LEFT', 'RIGHT', 'STEER_STOP'):
            return False
    except (ValueError, TypeError):
        return False
    return True


class DataRecorder:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self.images_dir = os.path.join(out_dir, 'images')
        os.makedirs(self.images_dir, exist_ok=True)
        self.csv_path = os.path.join(out_dir, 'dataset.csv')
        self.recording = False
        self.running = True
        self.samples_written = 0
        self.dropped_rows = 0
        self.session_id = time.strftime('%Y%m%d_%H%M%S')
        self.prev_action = STOP
        self.lock = threading.Lock()
        self.sample_hz = float(CFG.get('recorder', {}).get('sample_hz', 10))
        self.jpg_quality = int(CFG.get('recorder', {}).get('jpg_quality', 85))
        self._init_csv()

    def _init_csv(self) -> None:
        is_new = not os.path.exists(self.csv_path)
        # Open in line-buffered + binary-flush mode — each writerow is flushed.
        self.csv_file = open(self.csv_path, 'a', newline='', buffering=1)
        self.csv_writer = csv.writer(self.csv_file)
        if is_new:
            self.csv_writer.writerow(CSV_HEADER)
        # Count existing samples
        try:
            import pandas as pd
            df = pd.read_csv(self.csv_path)
            self.samples_written = len(df)
        except Exception:
            pass

    def toggle(self) -> bool:
        with self.lock:
            self.recording = not self.recording
            state = self.recording
        log.info(f"[Recorder] {'RECORDING' if state else 'PAUSED'} "
                 f"(samples: {self.samples_written}, dropped: {self.dropped_rows})")
        return state

    def is_recording(self) -> bool:
        with self.lock:
            return self.recording

    def record_loop(self) -> None:
        interval = 1.0 / self.sample_hz
        while self.running:
            try:
                self._record_one()
            except Exception as e:
                log.warning(f"recorder iteration failed: {e}")
            time.sleep(interval if not self.is_recording() else interval)

    def _record_one(self) -> None:
        if not self.is_recording():
            return
        if camera is None or pi_client is None:
            return
        frame, _ = camera.read()
        status = pi_client.get_status()
        if frame is None or status is None:
            return

        drive, steer, speed = pi_client.get_state_snapshot()
        dists = status.get('distances', {})
        gps = status.get('gps', {})
        action_id = manual_to_action(drive, steer, speed)

        # Write image to a temp file then rename — atomic-ish on POSIX
        frame_name = f"{self.session_id}_{uuid.uuid4().hex[:8]}.jpg"
        frame_path = os.path.join(self.images_dir, frame_name)
        tmp_path = frame_path + '.tmp'
        if not cv2.imwrite(tmp_path, frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpg_quality]):
            self.dropped_rows += 1
            return
        os.rename(tmp_path, frame_path)

        yolo_snap = get_yolo_snapshot()
        row = [
            self.session_id, time.time(), frame_path,
            dists.get('FL', 0), dists.get('FR', 0), dists.get('FW', 0),
            dists.get('BC', 0), dists.get('LS', 0), dists.get('RS', 0),
            int(gps.get('valid', 0)), gps.get('speed_mps', 0.0), gps.get('heading_deg', 0.0),
            drive, steer, speed,
            self.prev_action, action_id, ACTION_NAMES[action_id],
            yolo_snap['person_detected'], yolo_snap['object_detected'],
            yolo_snap['nearest_area_ratio'],
            yolo_snap['nearest_position_x'],
            yolo_snap['num_objects'],
        ]

        if not _validate_row(row):
            self.dropped_rows += 1
            log.warning(f"recorder dropped invalid row (n_dropped={self.dropped_rows})")
            try: os.remove(frame_path)
            except OSError: pass
            return

        try:
            self.csv_writer.writerow(row)
            self.csv_file.flush()
            os.fsync(self.csv_file.fileno())
            self.prev_action = action_id
            self.samples_written += 1
        except Exception as e:
            log.error(f"CSV write failed: {e}")
            self.dropped_rows += 1

    def close(self) -> None:
        self.running = False
        self.recording = False
        try:
            if self.csv_file:
                self.csv_file.close()
        except Exception:
            pass
        log.info(f"[Recorder] Saved {self.samples_written} total samples "
                 f"(dropped {self.dropped_rows}) to {self.csv_path}")


# ===== Web UI =============================================================
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
<title>Vehicle Control</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body {
    background:#1a1a2e; color:#eee; font-family:system-ui,-apple-system,sans-serif;
    touch-action:manipulation; user-select:none; -webkit-user-select:none; overflow-x:hidden;
}
.header { background:#16213e; padding:10px 16px; display:flex; justify-content:space-between; align-items:center; }
.header h1 { font-size:18px; color:#0f0; }
.status-dot { width:12px; height:12px; border-radius:50%; display:inline-block; }
.status-dot.on { background:#0f0; } .status-dot.off { background:#f00; }
.camera-container { width:100%; max-width:640px; margin:8px auto; position:relative; background:#000; border-radius:8px; overflow:hidden; }
.camera-container img { width:100%; display:block; }
.rec-overlay { position:absolute; top:10px; left:10px; padding:4px 10px; border-radius:4px; font-weight:bold; font-size:14px; }
.rec-overlay.on { background:rgba(255,0,0,0.8); color:#fff; } .rec-overlay.off { background:rgba(0,0,0,0.5); color:#888; }
.sensor-bar { display:grid; grid-template-columns:repeat(3,1fr); gap:6px; padding:8px 12px; max-width:640px; margin:0 auto; }
.sensor { background:#16213e; border-radius:6px; padding:8px; text-align:center; }
.sensor .label { font-size:11px; color:#888; }
.sensor .value { font-size:18px; font-weight:bold; }
.sensor .value.danger { color:#f44; } .sensor .value.warn { color:#fa0; } .sensor .value.safe { color:#0f0; }
.sensor .unit { font-size:10px; color:#666; }
.controls { max-width:400px; margin:12px auto; padding:0 12px; }
.dpad { display:grid; grid-template-columns:1fr 1fr 1fr; grid-template-rows:1fr 1fr 1fr; gap:8px; width:260px; margin:0 auto; }
.btn { background:#0a3d62; border:2px solid #1e90ff; border-radius:12px; color:#fff; font-size:18px; font-weight:bold; cursor:pointer; display:flex; align-items:center; justify-content:center; min-height:70px; transition:background 0.1s; }
.btn:active, .btn.active { background:#1e90ff; }
.btn.stop-btn { background:#8b0000; border-color:#f44; font-size:16px; }
.btn.stop-btn:active, .btn.stop-btn.active { background:#f44; }
.rec-btn { display:block; max-width:260px; margin:12px auto; padding:14px; border:2px solid #f44; border-radius:12px; background:#3a0000; color:#fff; font-size:18px; font-weight:bold; text-align:center; cursor:pointer; }
.rec-btn.recording { background:#f44; border-color:#fff; }
.speed-section { max-width:400px; margin:12px auto; padding:0 24px; text-align:center; }
.speed-section label { font-size:14px; color:#888; }
.speed-section input[type=range] { width:100%; margin:6px 0; }
.speed-val { font-size:24px; font-weight:bold; color:#1e90ff; }
.info-bar { max-width:640px; margin:8px auto; padding:4px 12px; display:flex; justify-content:space-between; font-size:12px; color:#666; }
.cmd-display { max-width:640px; margin:4px auto; padding:6px 12px; text-align:center; font-size:16px; color:#1e90ff; }
.samples-count { text-align:center; font-size:14px; color:#888; margin:4px; }
</style>
</head>
<body>

<div class="header">
    <h1>VEHICLE CONTROL</h1>
    <div><span class="status-dot" id="connDot"></span> <span id="connText">---</span></div>
</div>

<div class="camera-container">
    <img id="camFeed" src="/video_feed" alt="Camera">
    <div class="rec-overlay off" id="recOverlay">REC OFF</div>
</div>

<div class="cmd-display">
    <span id="cmdDrive">STOP</span> | <span id="cmdSteer">STRAIGHT</span>
</div>
<div class="cmd-display" id="yoloInfo" style="font-size:14px;color:#888;">
    YOLO: <span id="yoloStatus">--</span>
</div>

<!-- ===== GPS NAVIGATION MAP ===== -->
<div id="navPanel" style="max-width:640px; margin:8px auto; padding:8px; background:#16213e; border-radius:8px;">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
        <strong style="color:#0f0;">NAVIGATION</strong>
        <span id="navState" style="font-size:13px;color:#888;">--</span>
    </div>
    <!-- Search input (Places Autocomplete) -->
    <div style="display:flex; gap:6px; margin-bottom:6px;">
        <input id="searchInput" type="text" placeholder="Search a place / type address" style="
            flex:1; padding:10px; border-radius:6px; border:1px solid #1e90ff;
            background:#0a1d2e; color:#fff; font-size:14px;">
        <button id="btnLocate" title="Center on my phone's location" style="
            min-width:44px; padding:10px; border-radius:6px; border:2px solid #1e90ff;
            background:#0a3d62; color:#fff; font-size:18px; cursor:pointer;">📍</button>
    </div>

    <!-- A / B selector — which point are you setting? -->
    <div style="display:flex; gap:6px; margin-bottom:6px;">
        <button id="btnPickA" class="ab-btn active" data-ab="A" style="
            flex:1; padding:8px; border-radius:6px; border:2px solid #0f0;
            background:#103a10; color:#fff; font-weight:bold; cursor:pointer;">
            A: Pickup
        </button>
        <button id="btnPickB" class="ab-btn" data-ab="B" style="
            flex:1; padding:8px; border-radius:6px; border:2px solid #444;
            background:#222; color:#888; font-weight:bold; cursor:pointer;">
            B: Destination
        </button>
        <button id="btnUseCar" title="Use car's GPS as A" style="
            min-width:44px; padding:8px; border-radius:6px; border:2px solid #1e90ff;
            background:#0a3d62; color:#fff; font-size:13px; cursor:pointer;">🚗→A</button>
    </div>
    <div id="abState" style="font-size:11px; color:#888; margin-bottom:6px; text-align:center;">
        Setting <b style="color:#0f0;">A (pickup)</b> — tap on map
    </div>

    <div id="map" style="width:100%; height:300px; background:#000; border-radius:6px;"></div>
    <div style="display:flex; gap:6px; margin-top:8px;">
        <button id="btnNavGo" class="btn" style="flex:1; min-height:44px; font-size:14px;
                background:#2d5016; border-color:#0f0;">GO</button>
        <button id="btnNavStop" class="btn stop-btn" style="flex:1; min-height:44px; font-size:14px;">STOP NAV</button>
        <button id="btnNavClear" class="btn" style="flex:1; min-height:44px; font-size:13px;">CLEAR</button>
    </div>
    <div id="navInfo" style="font-size:12px; color:#888; text-align:center; margin-top:6px;">
        Tap "A: Pickup" or "B: Destination", then tap map
    </div>
</div>

<div class="sensor-bar">
    <div class="sensor"><div class="label">Front-L</div><div class="value" id="sFL">--</div><div class="unit">cm</div></div>
    <div class="sensor"><div class="label">Front-W</div><div class="value" id="sFW">--</div><div class="unit">cm</div></div>
    <div class="sensor"><div class="label">Front-R</div><div class="value" id="sFR">--</div><div class="unit">cm</div></div>
    <div class="sensor"><div class="label">Left</div><div class="value" id="sLS">--</div><div class="unit">cm</div></div>
    <div class="sensor"><div class="label">Back</div><div class="value" id="sBC">--</div><div class="unit">cm</div></div>
    <div class="sensor"><div class="label">Right</div><div class="value" id="sRS">--</div><div class="unit">cm</div></div>
</div>
<div class="sensor-bar" style="grid-template-columns:1fr 1fr; max-width:320px;">
    <div class="sensor"><div class="label">MIN FRONT</div><div class="value" id="sMinF">--</div><div class="unit">cm</div></div>
    <div class="sensor"><div class="label">MIN BACK</div><div class="value" id="sMinB">--</div><div class="unit">cm</div></div>
</div>

<div class="controls">
    <div class="dpad">
        <div></div>
        <div class="btn" id="btnW" data-drive="FORWARD">FWD</div>
        <div class="btn" id="btnX" data-steer="STEER_STOP">STRAIGHT</div>
        <div class="btn" id="btnA" data-steer="LEFT">LEFT</div>
        <div class="btn stop-btn" id="btnStop" data-drive="STOP" data-steer="STEER_STOP">STOP</div>
        <div class="btn" id="btnD" data-steer="RIGHT">RIGHT</div>
        <div></div>
        <div class="btn" id="btnS" data-drive="BACKWARD">REV</div>
        <div></div>
    </div>
</div>

<div class="speed-section">
    <label>SPEED</label>
    <div class="speed-val" id="speedVal">50%</div>
    <input type="range" id="speedSlider" min="0" max="100" value="50" step="5">
</div>

<div class="rec-btn" id="recBtn" onclick="toggleRec()">START RECORDING</div>
<div class="samples-count">Samples: <span id="sampleCount">0</span> · Dropped: <span id="droppedCount">0</span></div>

<div class="info-bar">
    <span>Speed: <span id="actualSpeed">0</span>%</span>
    <span>Alert: <span id="alertLevel">--</span></span>
    <span id="autoState"></span>
</div>

<script>
function sendCmd(drive, steer, speed) {
    const params = new URLSearchParams();
    if (drive) params.set('drive', drive);
    if (steer) params.set('steer', steer);
    if (speed !== undefined) params.set('speed', speed);
    fetch('/cmd?' + params.toString()).catch(() => {});
}

const STEER_PULSE_MS = """ + str(int(CFG.get('model', {}).get('pulse_steer_ms', 200))) + """;
let steerTimer = null;

document.querySelectorAll('.btn').forEach(btn => {
    const drive = btn.dataset.drive || null;
    const steer = btn.dataset.steer || null;
    function tap(e) {
        e.preventDefault();
        document.querySelectorAll('.btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        sendCmd(drive, steer);
        if (steerTimer) { clearTimeout(steerTimer); steerTimer = null; }
        if (steer === 'LEFT' || steer === 'RIGHT') {
            steerTimer = setTimeout(() => {
                sendCmd(null, 'STEER_STOP');
                btn.classList.remove('active');
                steerTimer = null;
            }, STEER_PULSE_MS);
        }
    }
    btn.addEventListener('click', tap);
    btn.addEventListener('touchstart', tap, {passive:false});
});

const keyMap = {w:'FORWARD', s:'BACKWARD'};
const steerMap = {a:'LEFT', d:'RIGHT'};
document.addEventListener('keydown', e => {
    const k = e.key.toLowerCase();
    if (keyMap[k]) sendCmd(keyMap[k], null);
    else if (steerMap[k]) {
        sendCmd(null, steerMap[k]);
        if (steerTimer) clearTimeout(steerTimer);
        steerTimer = setTimeout(() => { sendCmd(null, 'STEER_STOP'); steerTimer = null; }, STEER_PULSE_MS);
    }
    else if (k === ' ') { e.preventDefault(); sendCmd('STOP', 'STEER_STOP'); }
    else if (k === 'x') sendCmd(null, 'STEER_STOP');
    else if (k === 'r') toggleRec();
});

const slider = document.getElementById('speedSlider');
const speedVal = document.getElementById('speedVal');
slider.addEventListener('input', () => {
    speedVal.textContent = slider.value + '%';
    sendCmd(null, null, slider.value);
});

function toggleRec() {
    fetch('/record/toggle').then(r => r.json()).then(d => {
        updateRecUI(d.recording, d.samples, d.dropped);
    }).catch(() => {});
}
function updateRecUI(recording, samples, dropped) {
    const btn = document.getElementById('recBtn');
    const overlay = document.getElementById('recOverlay');
    if (recording) {
        btn.textContent = 'STOP RECORDING';
        btn.classList.add('recording');
        overlay.textContent = 'REC';
        overlay.className = 'rec-overlay on';
    } else {
        btn.textContent = 'START RECORDING';
        btn.classList.remove('recording');
        overlay.textContent = 'REC OFF';
        overlay.className = 'rec-overlay off';
    }
    document.getElementById('sampleCount').textContent = samples || 0;
    document.getElementById('droppedCount').textContent = dropped || 0;
}

function colorForDist(v) {
    if (v < 50) return 'danger';
    if (v < 150) return 'warn';
    return 'safe';
}
function pollStatus() {
    fetch('/status').then(r => r.json()).then(s => {
        const dot = document.getElementById('connDot');
        const txt = document.getElementById('connText');
        if (s.connected) { dot.className = 'status-dot on'; txt.textContent = 'Connected'; }
        else { dot.className = 'status-dot off'; txt.textContent = 'Disconnected'; }

        if (s.distances) {
            ['FL','FR','FW','BC','LS','RS'].forEach(k => {
                const el = document.getElementById('s' + k);
                const v = s.distances[k] || 0;
                el.textContent = v.toFixed(0);
                el.className = 'value ' + colorForDist(v);
            });
        }

        document.getElementById('cmdDrive').textContent = s.drive || 'STOP';
        document.getElementById('cmdSteer').textContent =
            (s.steer === 'STEER_STOP' ? 'STRAIGHT' : s.steer) || 'STRAIGHT';
        document.getElementById('actualSpeed').textContent = s.actual_speed || 0;
        document.getElementById('alertLevel').textContent = s.alert_level || '--';
        document.getElementById('autoState').textContent = s.auto_state || '';

        const mf = s.min_front || 0;
        const mb = s.min_back || 0;
        const elMF = document.getElementById('sMinF');
        const elMB = document.getElementById('sMinB');
        elMF.textContent = mf.toFixed(0); elMB.textContent = mb.toFixed(0);
        elMF.className = 'value ' + colorForDist(mf);
        elMB.className = 'value ' + colorForDist(mb);

        updateRecUI(s.recording, s.samples, s.dropped);

        if (s.yolo) {
            const y = s.yolo;
            const sideTxt = (y.nearest_position_x < -0.33) ? 'LEFT' :
                            (y.nearest_position_x > 0.33) ? 'RIGHT' : 'CENTER';
            let txt = `objects=${y.num_objects}`;
            if (y._stale) txt = '⚠ YOLO STALE';
            else if (y.person_detected) txt = '⚠ PERSON ' + sideTxt + ' | ' + txt;
            else if (y.object_detected) txt = `obj at ${sideTxt} | ` + txt;
            document.getElementById('yoloStatus').textContent = txt;
            const yEl = document.getElementById('yoloInfo');
            yEl.style.color = (y._stale || y.person_detected) ? '#f44' : '#888';
        }
    }).catch(() => {});
}
setInterval(pollStatus, 300);


// ===== Google Maps + Navigation =====
let gmap = null;
let carMarker = null;
let userMarker = null;       // phone holder's location (HTML5 geolocation)
let userAccCircle = null;    // accuracy circle around user marker
let aMarker = null;          // A = pickup
let bMarker = null;          // B = destination
let routeLine = null;
let trailLine = null;
let trail = [];
let pointA = null;           // {lat, lng}
let pointB = null;           // {lat, lng}
let activeAB = 'A';          // which one the next tap sets
let placesAutocomplete = null;

const MAX_TRAIL_POINTS = 200;
const POLL_NAV_MS = 500;

function setActiveAB(which) {
    activeAB = which;
    document.querySelectorAll('.ab-btn').forEach(b => {
        const isActive = b.dataset.ab === which;
        b.style.borderColor = isActive ? '#0f0' : '#444';
        b.style.background = isActive ? '#103a10' : '#222';
        b.style.color = isActive ? '#fff' : '#888';
    });
    const label = (which === 'A') ? 'A (pickup)' : 'B (destination)';
    const colour = (which === 'A') ? '#0f0' : '#f44';
    document.getElementById('abState').innerHTML =
        `Setting <b style="color:${colour};">${label}</b> — tap on map`;
}
document.getElementById('btnPickA').addEventListener('click', () => setActiveAB('A'));
document.getElementById('btnPickB').addEventListener('click', () => setActiveAB('B'));

function initMap() {
    // Default center: CUST Islamabad campus (replace with your area). The map
    // will recenter on the car's first valid GPS fix.
    const cust = {lat: 33.6520, lng: 73.1613};
    gmap = new google.maps.Map(document.getElementById('map'), {
        center: cust,
        zoom: 18,
        mapTypeId: 'satellite',
        disableDefaultUI: false,
        gestureHandling: 'greedy',  // single-finger pan/zoom on mobile
    });

    // Tap → set whichever point is currently active (A or B)
    gmap.addListener('click', e => {
        const lat = e.latLng.lat();
        const lng = e.latLng.lng();
        setPointAB(activeAB, lat, lng);
    });

    // Places Autocomplete on the search box (only works if Places API is
    // enabled in Google Cloud Console + libraries=places in the Maps URL).
    if (google.maps.places && google.maps.places.Autocomplete) {
        const input = document.getElementById('searchInput');
        placesAutocomplete = new google.maps.places.Autocomplete(input, {
            types: ['geocode', 'establishment'],
            fields: ['geometry', 'name', 'formatted_address'],
        });
        placesAutocomplete.bindTo('bounds', gmap);
        placesAutocomplete.addListener('place_changed', () => {
            const place = placesAutocomplete.getPlace();
            if (!place || !place.geometry || !place.geometry.location) return;
            const lat = place.geometry.location.lat();
            const lng = place.geometry.location.lng();
            gmap.panTo({lat, lng});
            gmap.setZoom(19);
            setPointAB(activeAB, lat, lng);
            input.value = place.name || place.formatted_address || '';
        });
    } else {
        // Places library not loaded (Places API not enabled).
        // Fallback: pressing Enter geocodes via the Maps Geocoder.
        const input = document.getElementById('searchInput');
        input.placeholder = 'Type address + Enter (Places API not enabled)';
        input.addEventListener('keypress', e => {
            if (e.key !== 'Enter') return;
            e.preventDefault();
            const q = input.value.trim();
            if (!q) return;
            const geocoder = new google.maps.Geocoder();
            geocoder.geocode({address: q}, (results, status) => {
                if (status !== 'OK' || !results[0]) {
                    alert('Could not find: ' + q);
                    return;
                }
                const loc = results[0].geometry.location;
                gmap.panTo(loc);
                gmap.setZoom(19);
                setPointAB(activeAB, loc.lat(), loc.lng());
            });
        });
    }
}

// HTML5 geolocation — phone's GPS / WiFi location of the *user* (not the car).
function locateMe() {
    if (!navigator.geolocation) {
        alert('Geolocation not supported by browser.');
        return;
    }
    document.getElementById('navInfo').textContent = 'Locating you…';
    navigator.geolocation.getCurrentPosition(
        pos => {
            const lat = pos.coords.latitude;
            const lng = pos.coords.longitude;
            const acc = pos.coords.accuracy || 50;   // metres
            if (gmap) {
                gmap.panTo({lat, lng});
                gmap.setZoom(19);
                if (userMarker) userMarker.setMap(null);
                if (userAccCircle) userAccCircle.setMap(null);
                userMarker = new google.maps.Marker({
                    position: {lat, lng}, map: gmap,
                    icon: {
                        path: google.maps.SymbolPath.CIRCLE,
                        scale: 7, fillColor: '#4285F4', fillOpacity: 1,
                        strokeColor: '#fff', strokeWeight: 2,
                    },
                    title: 'You',
                });
                userAccCircle = new google.maps.Circle({
                    center: {lat, lng}, radius: acc,
                    fillColor: '#4285F4', fillOpacity: 0.12,
                    strokeColor: '#4285F4', strokeOpacity: 0.4, strokeWeight: 1,
                    map: gmap,
                });
            }
            document.getElementById('navInfo').textContent =
                `Your location: ${lat.toFixed(5)}, ${lng.toFixed(5)} (±${acc.toFixed(0)}m)`;
        },
        err => {
            alert('Could not locate you: ' + err.message);
        },
        {enableHighAccuracy: true, timeout: 10000, maximumAge: 5000}
    );
}
document.getElementById('btnLocate').addEventListener('click', locateMe);

function setPointAB(which, lat, lng) {
    const pos = {lat: lat, lng: lng};
    if (which === 'A') {
        pointA = pos;
        if (aMarker) aMarker.setMap(null);
        aMarker = new google.maps.Marker({
            position: pos, map: gmap,
            label: {text: 'A', color: '#fff', fontWeight: 'bold'},
            icon: {
                path: google.maps.SymbolPath.CIRCLE,
                scale: 16, fillColor: '#0a0', fillOpacity: 1,
                strokeColor: '#fff', strokeWeight: 2,
            },
            title: 'Pickup (A)',
        });
        // After picking A, auto-switch to B
        setActiveAB('B');
    } else {
        pointB = pos;
        if (bMarker) bMarker.setMap(null);
        bMarker = new google.maps.Marker({
            position: pos, map: gmap,
            label: {text: 'B', color: '#fff', fontWeight: 'bold'},
            icon: {
                path: google.maps.SymbolPath.CIRCLE,
                scale: 16, fillColor: '#d00', fillOpacity: 1,
                strokeColor: '#fff', strokeWeight: 2,
            },
            title: 'Destination (B)',
        });
    }
    updateRouteAndStage();
}

let directionsService = null;
function updateRouteAndStage() {
    // Stage waypoints with backend (A then B). The car drives them in order.
    const wps = [];
    if (pointA) wps.push({lat: pointA.lat, lon: pointA.lng});
    if (pointB) wps.push({lat: pointB.lat, lon: pointB.lng});
    if (wps.length > 0) {
        fetch('/api/destination', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({waypoints: wps}),
        });
    }

    // Visualise route (only if both A and B are set)
    if (!pointA || !pointB) {
        if (routeLine) { routeLine.setMap(null); routeLine = null; }
        document.getElementById('navInfo').textContent =
            (pointA && !pointB) ? 'A set. Now tap B (destination).' :
            (pointB && !pointA) ? 'B set. Now tap A (pickup) or use 🚗→A.' :
            'Tap "A: Pickup" or "B: Destination", then tap map';
        return;
    }
    document.getElementById('navInfo').textContent =
        `A → B set. Tap GO to drive.`;

    if (!directionsService) directionsService = new google.maps.DirectionsService();
    directionsService.route({
        origin: pointA, destination: pointB, travelMode: 'WALKING',
    }, (res, status) => {
        if (routeLine) routeLine.setMap(null);
        if (status !== 'OK' || !res.routes || !res.routes[0]) {
            // Fall back to a straight line
            routeLine = new google.maps.Polyline({
                path: [pointA, pointB],
                geodesic: true, strokeColor: '#FF8800',
                strokeWeight: 3, strokeOpacity: 0.7, map: gmap,
            });
            return;
        }
        routeLine = new google.maps.Polyline({
            path: res.routes[0].overview_path,
            strokeColor: '#FF8800', strokeWeight: 4, strokeOpacity: 0.8,
            map: gmap,
        });
    });
}

// "Use car's GPS as A" button
document.getElementById('btnUseCar').addEventListener('click', () => {
    if (!carMarker) {
        alert('No car GPS yet. Connect Pi and wait for fix.');
        return;
    }
    const p = carMarker.getPosition();
    setPointAB('A', p.lat(), p.lng());
});

document.getElementById('btnNavGo').addEventListener('click', () => {
    if (!pointB) {
        alert('Set destination (B) first.');
        return;
    }
    fetch('/api/go', {method: 'POST'}).then(r => r.json()).then(d => {
        document.getElementById('navInfo').textContent =
            d.ok ? 'Navigation started.' : ('Failed: ' + (d.error || 'unknown'));
    });
});
document.getElementById('btnNavStop').addEventListener('click', () => {
    fetch('/api/stop', {method: 'POST'});
    document.getElementById('navInfo').textContent = 'Navigation stopped.';
});
document.getElementById('btnNavClear').addEventListener('click', () => {
    if (aMarker) { aMarker.setMap(null); aMarker = null; }
    if (bMarker) { bMarker.setMap(null); bMarker = null; }
    if (routeLine) { routeLine.setMap(null); routeLine = null; }
    pointA = null; pointB = null;
    setActiveAB('A');
    fetch('/api/stop', {method: 'POST'});
    document.getElementById('navInfo').textContent = 'Tap "A: Pickup" or "B: Destination", then tap map';
});

function pollNav() {
    fetch('/api/status').then(r => r.json()).then(s => {
        const navState = (s.nav && s.nav.state) || '--';
        document.getElementById('navState').textContent = navState;
        if (s.nav && s.nav.info && s.nav.info.dist_m !== undefined) {
            document.getElementById('navInfo').textContent =
                `${navState}: ${s.nav.info.dist_m.toFixed(1)}m to target  err=${(s.nav.info.err_deg || 0).toFixed(0)}°`;
        }
        if (gmap && s.gps && s.gps.valid && s.gps.lat != null) {
            const pos = {lat: s.gps.lat, lng: s.gps.lon};
            if (!carMarker) {
                carMarker = new google.maps.Marker({
                    position: pos, map: gmap,
                    icon: {
                        path: google.maps.SymbolPath.CIRCLE,
                        scale: 8, fillColor: '#1e90ff', fillOpacity: 1,
                        strokeColor: '#fff', strokeWeight: 2,
                    },
                    title: 'Car',
                });
                gmap.setCenter(pos);
            } else {
                carMarker.setPosition(pos);
            }
            // Append to trail
            trail.push(pos);
            if (trail.length > MAX_TRAIL_POINTS) trail.shift();
            if (trailLine) {
                trailLine.setPath(trail);
            } else {
                trailLine = new google.maps.Polyline({
                    path: trail, strokeColor: '#00ff00',
                    strokeWeight: 3, strokeOpacity: 0.8, map: gmap,
                });
            }
        }
    }).catch(() => {});
}
setInterval(pollNav, POLL_NAV_MS);

// Load Google Maps after fetching the API key
fetch('/api/maps_key').then(r => r.json()).then(d => {
    if (!d.key) {
        document.getElementById('navInfo').textContent =
            'No Google Maps key configured (laptop/secrets.yaml).';
        return;
    }
    const s = document.createElement('script');
    s.src = 'https://maps.googleapis.com/maps/api/js?key=' + encodeURIComponent(d.key) + '&libraries=places&callback=initMap';
    s.async = true; s.defer = true;
    document.head.appendChild(s);
});
</script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_PAGE)


@app.route('/cmd')
def cmd():
    drive = request.args.get('drive')
    steer = request.args.get('steer')
    speed = request.args.get('speed')
    if pi_client is not None:
        pi_client.set_command(
            drive=drive,
            steer=steer,
            speed=int(speed) if speed else None,
        )
    return jsonify(ok=True)


@app.route('/status')
def status():
    if pi_client is None:
        return jsonify(connected=False)
    st = pi_client.get_status() or {}
    drive, steer, speed = pi_client.get_state_snapshot()
    return jsonify(
        connected=pi_client.connected,
        drive=drive, steer=steer, speed=speed,
        actual_speed=st.get('actual_speed', 0),
        distances=st.get('distances', {}),
        alert_level=st.get('alert_level', ''),
        auto_state=st.get('auto_state', ''),
        min_front=st.get('min_distance_front', 0),
        min_back=st.get('min_distance_back', 0),
        recording=recorder.is_recording() if recorder else False,
        samples=recorder.samples_written if recorder else 0,
        dropped=recorder.dropped_rows if recorder else 0,
        yolo=get_yolo_snapshot(),
    )


# ===== Navigation API ======================================================
def _gps_position() -> Optional[tuple]:
    """Returns (lat, lon) from latest Pi status, or None if no fix."""
    if pi_client is None:
        return None
    st = pi_client.get_status() or {}
    g = st.get('gps') or {}
    if int(g.get('valid', 0)) != 1:
        return None
    lat = g.get('lat')
    lon = g.get('lon')
    if lat is None or lon is None:
        return None
    return (float(lat), float(lon))


def _gps_heading() -> Optional[float]:
    if pi_client is None:
        return None
    st = pi_client.get_status() or {}
    g = st.get('gps') or {}
    if int(g.get('valid', 0)) != 1:
        return None
    h = g.get('heading_deg')
    speed = float(g.get('speed_mps', 0))
    # GPS heading is unreliable below ~1 m/s — discard
    if h is None or speed < 1.0:
        return None
    return float(h)


def _apply_obstacle_override(nav_action: int, sensors: dict, yolo: dict
                             ) -> tuple:
    """Run the hybrid sensor+YOLO check on top of a nav-chosen action.

    Returns (final_action_id, override_reason).
    Policy:
      * If hybrid says STOP (person close, pinned, etc) → STOP
      * If hybrid says REVERSE_* (front emergency, need to back out) → REVERSE_*
      * Otherwise → keep the nav action (the navigation steers toward the
        target; the hybrid would only "slow down or turn" which would
        contradict the chosen heading)
    """
    if not sensors:
        return nav_action, ''
    h_action, h_reason = hybrid_decide(sensors, yolo)
    if h_action == STOP:
        return STOP, f"OBST_STOP {h_reason}"
    if h_action in (REVERSE, REVERSE_LEFT, REVERSE_RIGHT):
        return h_action, f"OBST_REVERSE {h_reason}"
    return nav_action, ''


def nav_loop() -> None:
    """Background thread: ticks the nav controller, applies obstacle override,
    sends final action to Pi.
    """
    global nav_last_info
    while True:
        try:
            if not nav_active or pi_client is None:
                time.sleep(0.2)
                continue

            pos = _gps_position()
            head = _gps_heading()
            with nav_lock:
                action_id, state, info = nav.tick(pos, head)

            if state == 'ARRIVED':
                pi_client.safe_stop()
                _set_nav_active(False)
                nav_last_info = {'state': state, **info}
                log.info(f"[NAV] ARRIVED  info={info}")
                time.sleep(0.5)
                continue

            # Pull current sensors + YOLO snapshot for the obstacle override.
            pi_status = pi_client.get_status() or {}
            sensors = pi_status.get('distances', {})
            yolo_snap = get_yolo_snapshot()

            final_action, reason = _apply_obstacle_override(
                action_id, sensors, yolo_snap)

            cmd = action_to_pi_command(final_action)
            pi_client.set_command(
                drive=cmd['command'], steer=cmd['steer'], speed=cmd['speed'])

            # Expose both the nav-chosen and the overridden action for the UI
            nav_last_info = {
                'state': state,
                **info,
                'nav_action': ACTION_NAMES[action_id],
                'final_action': ACTION_NAMES[final_action],
                'override_reason': reason,
            }
        except Exception as e:
            log.warning(f"nav_loop iteration failed: {e}")
        time.sleep(0.2)


def _set_nav_active(value: bool) -> None:
    global nav_active
    nav_active = value


@app.route('/api/status')
def api_status():
    """Status payload for the mobile app + web map UI."""
    with nav_lock:
        nav_state = nav.state
        wps = nav.waypoints

    if pi_client is None:
        return jsonify(
            connected=False,
            gps={'valid': False},
            sent={'drive': 'STOP', 'steer': 'STEER_STOP', 'speed': 0},
            sensors={},
            nav={
                'active': nav_active,
                'state': nav_state,
                'waypoints': [{'lat': lat, 'lon': lon} for lat, lon in wps],
                'info': nav_last_info,
            },
            yolo=get_yolo_snapshot(),
        )
    st = pi_client.get_status() or {}
    g = st.get('gps') or {}
    drive, steer, speed = pi_client.get_state_snapshot()
    return jsonify(
        connected=pi_client.connected,
        gps={
            'valid': int(g.get('valid', 0)) == 1,
            'lat': g.get('lat'),
            'lon': g.get('lon'),
            'heading_deg': g.get('heading_deg'),
            'speed_mps': g.get('speed_mps'),
            'satellites': g.get('satellites'),
        },
        sent={'drive': drive, 'steer': steer, 'speed': speed},
        sensors=st.get('distances', {}),
        nav={
            'active': nav_active,
            'state': nav_state,
            'waypoints': [{'lat': lat, 'lon': lon} for lat, lon in wps],
            'info': nav_last_info,
        },
        yolo=get_yolo_snapshot(),
        recording=recorder.is_recording() if recorder else False,
        samples=recorder.samples_written if recorder else 0,
    )


@app.route('/api/destination', methods=['POST'])
def api_destination():
    """Set a single destination. Body JSON: {lat, lon} OR {waypoints: [{lat,lon}, ...]}.
    Calling this only stages the destination; you must also call /api/go to start moving."""
    data = request.get_json(silent=True) or request.args
    waypoints: list = []
    if 'waypoints' in data and isinstance(data['waypoints'], list):
        for w in data['waypoints']:
            try:
                waypoints.append((float(w['lat']), float(w['lon'])))
            except (KeyError, TypeError, ValueError):
                continue
    elif 'lat' in data and 'lon' in data:
        waypoints = [(float(data['lat']), float(data['lon']))]
    if not waypoints:
        return jsonify(ok=False, error='no valid waypoints'), 400
    with nav_lock:
        nav.set_waypoints(waypoints)
    log.info(f"[NAV] {len(waypoints)} waypoint(s) staged. Call /api/go to start.")
    return jsonify(ok=True, waypoints=len(waypoints))


@app.route('/api/go', methods=['POST'])
def api_go():
    """Start navigation toward the staged destination."""
    with nav_lock:
        if not nav.waypoints:
            return jsonify(ok=False, error='no destination set'), 400
    _set_nav_active(True)
    log.info("[NAV] GO")
    return jsonify(ok=True)


@app.route('/api/stop', methods=['POST'])
def api_stop():
    """Stop navigation + safe_stop motors."""
    _set_nav_active(False)
    with nav_lock:
        nav.stop()
    if pi_client is not None:
        pi_client.safe_stop()
    log.info("[NAV] STOP")
    return jsonify(ok=True)


@app.route('/api/maps_key')
def api_maps_key():
    """Returns the Google Maps API key for the front-end. Restrict the key
    in Google Cloud Console to your laptop IP — this isn't a secret, it's
    OK to expose to the page that uses it. NEVER commit the key to git
    (it lives in laptop/secrets.yaml which is gitignored)."""
    key = CFG.get('google', {}).get('maps_api_key', '')
    return jsonify(key=key)


@app.route('/record/toggle')
def record_toggle():
    if recorder is not None:
        is_rec = recorder.toggle()
        return jsonify(recording=is_rec,
                       samples=recorder.samples_written,
                       dropped=recorder.dropped_rows)
    return jsonify(recording=False, samples=0, dropped=0)


def gen_frames():
    while True:
        try:
            if camera is None:
                time.sleep(0.1); continue
            frame, _ = camera.read()
            if frame is None:
                time.sleep(0.05); continue
            small = cv2.resize(frame, (480, 360))
            ok, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 60])
            if not ok:
                time.sleep(0.05); continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            time.sleep(0.066)  # ~15 fps
        except Exception:
            time.sleep(0.1)


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def _signal_handler(signum, _frame):
    log.warning(f"Signal {signum} received → graceful shutdown")
    try:
        if recorder: recorder.close()
    finally:
        try:
            if pi_client: pi_client.close()
        finally:
            os._exit(0)


def main() -> None:
    global pi_client, camera, recorder, yolo

    p = argparse.ArgumentParser()
    p.add_argument('--pi', default=None,
                   help='Pi IP address. Omit (or use --no-pi) to run map-UI-only.')
    p.add_argument('--port', type=int, default=CFG.get('web', {}).get('port', 8080))
    p.add_argument('--camera', type=int, default=CFG.get('camera', {}).get('device', 0))
    p.add_argument('--data-dir',
                   default=os.path.join(os.path.dirname(__file__),
                                        CFG.get('recorder', {}).get('out_dir', 'data')))
    p.add_argument('--no-yolo', action='store_true')
    p.add_argument('--no-pi', action='store_true',
                   help='Skip Pi connection (map-UI-only / dry-run mode).')
    p.add_argument('--no-camera', action='store_true',
                   help='Skip camera (useful if no webcam plugged in).')
    p.add_argument('--https', action='store_true',
                   help='Serve over HTTPS with a self-signed cert (required '
                        'for the 📍 location button on phones — browsers '
                        'block geolocation over plain HTTP).')
    args = p.parse_args()

    import signal
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    if args.no_camera:
        log.info("[main] camera disabled by flag")
    else:
        camera = Camera(device=args.camera)
        if not camera.start():
            log.warning("Camera failed to start — continuing without camera")
            camera = None

    if args.no_pi or not args.pi:
        log.info("[main] Pi connection disabled — map UI / API only mode")
        pi_client = None
    else:
        pi_client = PiClient(args.pi)
        if not pi_client.connect():
            log.warning(f"Cannot connect to Pi @ {args.pi} — continuing without Pi")
            pi_client = None
        else:
            pi_client.start()

    if YOLO_AVAILABLE and not args.no_yolo:
        try:
            log.info("[YOLO] Loading...")
            yolo = ObjectDetector(
                conf_threshold=float(CFG.get('yolo', {}).get('conf_threshold', 0.4)),
                device='cpu',
            )
            threading.Thread(target=yolo_loop, daemon=True).start()
            log.info("[YOLO] Started")
        except Exception as e:
            log.error(f"[YOLO] Failed to start: {e}")
            yolo = None

    recorder = DataRecorder(args.data_dir)
    threading.Thread(target=recorder.record_loop, daemon=True).start()

    # Navigation control loop
    threading.Thread(target=nav_loop, daemon=True).start()

    log.info("=" * 55)
    log.info("  WEB VEHICLE CONTROL + RECORDER + YOLO")
    log.info("=" * 55)
    log.info(f"  Open in browser: http://0.0.0.0:{args.port}")
    log.info(f"  Pi: {args.pi}")
    log.info(f"  Camera: device {args.camera}")
    log.info(f"  YOLO: {'ON' if yolo else 'OFF'}")
    log.info(f"  Data: {args.data_dir}")
    log.info(f"  Existing samples: {recorder.samples_written}")
    log.info("=" * 55)

    run_kwargs = dict(
        host=CFG.get('web', {}).get('bind', '0.0.0.0'),
        port=args.port, threaded=True, use_reloader=False,
    )
    if args.https:
        # Self-signed cert generated on the fly. Browser will warn — accept.
        # Required for the geolocation 📍 button on phones (HTTPS-only API).
        run_kwargs['ssl_context'] = 'adhoc'
        log.info(f"  HTTPS: ON (self-signed). Open: https://<laptop-ip>:{args.port}")
        log.info("  Browser will warn about cert — accept it once.")

    try:
        app.run(**run_kwargs)
    except KeyboardInterrupt:
        pass
    finally:
        if recorder: recorder.close()
        if pi_client: pi_client.close()
        if camera: camera.stop()
        log.info("Stopped.")


if __name__ == '__main__':
    main()
