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
from ml.actions import manual_to_action, ACTION_NAMES, STOP

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
    p.add_argument('--pi', required=True)
    p.add_argument('--port', type=int, default=CFG.get('web', {}).get('port', 8080))
    p.add_argument('--camera', type=int, default=CFG.get('camera', {}).get('device', 0))
    p.add_argument('--data-dir',
                   default=os.path.join(os.path.dirname(__file__),
                                        CFG.get('recorder', {}).get('out_dir', 'data')))
    p.add_argument('--no-yolo', action='store_true')
    args = p.parse_args()

    import signal
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    camera = Camera(device=args.camera)
    if not camera.start():
        log.error("Camera failed to start"); return

    pi_client = PiClient(args.pi)
    if not pi_client.connect():
        log.error("Cannot connect to Pi")
        camera.stop()
        return
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

    try:
        app.run(host=CFG.get('web', {}).get('bind', '0.0.0.0'),
                port=args.port, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        recorder.close()
        pi_client.close()
        camera.stop()
        log.info("Stopped.")


if __name__ == '__main__':
    main()
