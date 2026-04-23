#!/usr/bin/env python3
"""
Web-based vehicle remote control + data recording.
Access from any device on the same network: http://<laptop-ip>:8080

Features:
    - Live camera feed (MJPEG)
    - Touch-friendly WASD controls (works on phone)
    - Real-time sensor display
    - Speed slider
    - Data recording (toggle from web UI)
"""

import argparse
import csv
import json
import os
import socket
import sys
import threading
import time
import uuid

import cv2
from flask import Flask, Response, jsonify, render_template_string, request

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from vision.camera import Camera
from ml.actions import manual_to_action, ACTION_NAMES, STOP

app = Flask(__name__)

# Global state
pi_client = None
camera = None
recorder = None


class PiClient:
    def __init__(self, ip: str, port: int = 5555):
        self.ip = ip
        self.port = port
        self.sock = None
        self.connected = False
        self.running = True
        self.status = None
        self.status_lock = threading.Lock()
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
            print(f"[PiClient] Connected to {self.ip}:{self.port}")
            return True
        except Exception as e:
            print(f"[PiClient] Connect failed: {e}")
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
                        with self.status_lock:
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
            if drive is not None:
                self.drive = drive
            if steer is not None:
                self.steer = steer
            if speed is not None:
                self.speed = max(0, min(100, int(speed)))

    def get_status(self):
        with self.status_lock:
            return dict(self.status) if self.status else None

    def close(self):
        self.running = False
        self.set_command(drive='STOP', steer='STEER_STOP', speed=0)
        time.sleep(0.3)
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass


class DataRecorder:
    """Records frames + sensor data to CSV at ~5Hz while recording is on."""

    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self.images_dir = os.path.join(out_dir, 'images')
        os.makedirs(self.images_dir, exist_ok=True)
        self.csv_path = os.path.join(out_dir, 'dataset.csv')
        self.recording = False
        self.running = True
        self.samples_written = 0
        self.session_id = time.strftime('%Y%m%d_%H%M%S')
        self.prev_action = STOP
        self.lock = threading.Lock()
        self._init_csv()

    def _init_csv(self):
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
        # Count existing samples
        try:
            import pandas as pd
            df = pd.read_csv(self.csv_path)
            self.samples_written = len(df)
        except Exception:
            pass

    def toggle(self):
        with self.lock:
            self.recording = not self.recording
            state = self.recording
        print(f"[Recorder] {'RECORDING' if state else 'PAUSED'} (samples: {self.samples_written})")
        return state

    def is_recording(self):
        with self.lock:
            return self.recording

    def record_loop(self):
        """Background thread: saves samples at ~5Hz when recording."""
        while self.running:
            if not self.is_recording():
                time.sleep(0.1)
                continue

            frame, _ = camera.read() if camera else (None, 0)
            status = pi_client.get_status() if pi_client else None

            if frame is None or status is None:
                time.sleep(0.1)
                continue

            with pi_client.cmd_lock:
                drive = pi_client.drive
                steer = pi_client.steer
                speed = pi_client.speed

            dists = status.get('distances', {})
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

            time.sleep(0.2)  # 5 Hz

    def close(self):
        self.running = False
        self.recording = False
        if self.csv_file:
            self.csv_file.close()
        print(f"[Recorder] Saved {self.samples_written} total samples to {self.csv_path}")


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
    touch-action:manipulation; user-select:none; -webkit-user-select:none;
    overflow-x:hidden;
}
.header {
    background:#16213e; padding:10px 16px; display:flex;
    justify-content:space-between; align-items:center;
}
.header h1 { font-size:18px; color:#0f0; }
.status-dot { width:12px; height:12px; border-radius:50%; display:inline-block; }
.status-dot.on { background:#0f0; }
.status-dot.off { background:#f00; }
.camera-container {
    width:100%; max-width:640px; margin:8px auto; position:relative;
    background:#000; border-radius:8px; overflow:hidden;
}
.camera-container img { width:100%; display:block; }
.rec-overlay {
    position:absolute; top:10px; left:10px; padding:4px 10px;
    border-radius:4px; font-weight:bold; font-size:14px;
}
.rec-overlay.on { background:rgba(255,0,0,0.8); color:#fff; }
.rec-overlay.off { background:rgba(0,0,0,0.5); color:#888; }
.sensor-bar {
    display:grid; grid-template-columns:repeat(3,1fr); gap:6px;
    padding:8px 12px; max-width:640px; margin:0 auto;
}
.sensor {
    background:#16213e; border-radius:6px; padding:8px; text-align:center;
}
.sensor .label { font-size:11px; color:#888; }
.sensor .value { font-size:18px; font-weight:bold; }
.sensor .value.danger { color:#f44; }
.sensor .value.warn { color:#fa0; }
.sensor .value.safe { color:#0f0; }
.sensor .unit { font-size:10px; color:#666; }
.controls {
    max-width:400px; margin:12px auto; padding:0 12px;
}
.dpad {
    display:grid; grid-template-columns:1fr 1fr 1fr;
    grid-template-rows:1fr 1fr 1fr; gap:8px; width:260px; margin:0 auto;
}
.btn {
    background:#0a3d62; border:2px solid #1e90ff; border-radius:12px;
    color:#fff; font-size:20px; font-weight:bold; cursor:pointer;
    display:flex; align-items:center; justify-content:center;
    min-height:70px; transition:background 0.1s;
}
.btn:active, .btn.active { background:#1e90ff; }
.btn.stop-btn { background:#8b0000; border-color:#f44; font-size:16px; }
.btn.stop-btn:active, .btn.stop-btn.active { background:#f44; }
.rec-btn {
    display:block; max-width:260px; margin:12px auto; padding:14px;
    border:2px solid #f44; border-radius:12px; background:#3a0000;
    color:#fff; font-size:18px; font-weight:bold; text-align:center; cursor:pointer;
}
.rec-btn.recording { background:#f44; border-color:#fff; }
.speed-section {
    max-width:400px; margin:12px auto; padding:0 24px; text-align:center;
}
.speed-section label { font-size:14px; color:#888; }
.speed-section input[type=range] { width:100%; margin:6px 0; }
.speed-val { font-size:24px; font-weight:bold; color:#1e90ff; }
.info-bar {
    max-width:640px; margin:8px auto; padding:4px 12px;
    display:flex; justify-content:space-between; font-size:12px; color:#666;
}
.cmd-display {
    max-width:640px; margin:4px auto; padding:6px 12px; text-align:center;
    font-size:16px; color:#1e90ff;
}
.samples-count {
    text-align:center; font-size:14px; color:#888; margin:4px;
}
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
        <div class="btn" id="btnW" data-drive="BACKWARD">FWD</div>
        <div></div>
        <div class="btn" id="btnA" data-steer="LEFT">LEFT</div>
        <div class="btn stop-btn" id="btnStop" data-drive="STOP" data-steer="STEER_STOP">STOP</div>
        <div class="btn" id="btnD" data-steer="RIGHT">RIGHT</div>
        <div></div>
        <div class="btn" id="btnS" data-drive="FORWARD">REV</div>
        <div></div>
    </div>
</div>

<div class="speed-section">
    <label>SPEED</label>
    <div class="speed-val" id="speedVal">50%</div>
    <input type="range" id="speedSlider" min="0" max="100" value="50" step="5">
</div>

<div class="rec-btn" id="recBtn" onclick="toggleRec()">START RECORDING</div>
<div class="samples-count">Samples: <span id="sampleCount">0</span></div>

<div class="info-bar">
    <span>Speed: <span id="actualSpeed">0</span>%</span>
    <span>Alert: <span id="alertLevel">--</span></span>
    <span id="autoState"></span>
</div>

<script>
// Command sending
function sendCmd(drive, steer, speed) {
    const params = new URLSearchParams();
    if (drive) params.set('drive', drive);
    if (steer) params.set('steer', steer);
    if (speed !== undefined) params.set('speed', speed);
    fetch('/cmd?' + params.toString()).catch(() => {});
}

// D-pad buttons - ONE TAP = keep moving, STOP to stop
document.querySelectorAll('.btn').forEach(btn => {
    const drive = btn.dataset.drive || null;
    const steer = btn.dataset.steer || null;

    function tap(e) {
        e.preventDefault();
        document.querySelectorAll('.btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        sendCmd(drive, steer);
    }

    btn.addEventListener('click', tap);
    btn.addEventListener('touchstart', tap, {passive:false});
});

// Keyboard controls
const keyMap = {w:'BACKWARD', s:'FORWARD'};
const steerMap = {a:'LEFT', d:'RIGHT'};
document.addEventListener('keydown', e => {
    const k = e.key.toLowerCase();
    if (keyMap[k]) sendCmd(keyMap[k], null);
    else if (steerMap[k]) sendCmd(null, steerMap[k]);
    else if (k === ' ') { e.preventDefault(); sendCmd('STOP', 'STEER_STOP'); }
    else if (k === 'x') sendCmd(null, 'STEER_STOP');
    else if (k === 'r') toggleRec();
});

// Speed slider
const slider = document.getElementById('speedSlider');
const speedVal = document.getElementById('speedVal');
slider.addEventListener('input', () => {
    speedVal.textContent = slider.value + '%';
    sendCmd(null, null, slider.value);
});

// Recording toggle
function toggleRec() {
    fetch('/record/toggle').then(r => r.json()).then(d => {
        updateRecUI(d.recording, d.samples);
    }).catch(() => {});
}

function updateRecUI(recording, samples) {
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
}

// Status polling
function colorForDist(v) {
    if (v < 50) return 'danger';
    if (v < 150) return 'warn';
    return 'safe';
}
function pollStatus() {
    fetch('/status').then(r => r.json()).then(s => {
        const dot = document.getElementById('connDot');
        const txt = document.getElementById('connText');
        if (s.connected) {
            dot.className = 'status-dot on'; txt.textContent = 'Connected';
        } else {
            dot.className = 'status-dot off'; txt.textContent = 'Disconnected';
        }

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
        elMF.textContent = mf.toFixed(0);
        elMB.textContent = mb.toFixed(0);
        elMF.className = 'value ' + colorForDist(mf);
        elMB.className = 'value ' + colorForDist(mb);

        // Update recording status
        updateRecUI(s.recording, s.samples);
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
    if pi_client:
        pi_client.set_command(
            drive=drive,
            steer=steer,
            speed=int(speed) if speed else None,
        )
    return jsonify(ok=True)


@app.route('/status')
def status():
    if not pi_client:
        return jsonify(connected=False)
    st = pi_client.get_status() or {}
    with pi_client.cmd_lock:
        drive = pi_client.drive
        steer = pi_client.steer
        speed = pi_client.speed
    return jsonify(
        connected=pi_client.connected,
        drive=drive,
        steer=steer,
        speed=speed,
        actual_speed=st.get('actual_speed', 0),
        distances=st.get('distances', {}),
        alert_level=st.get('alert_level', ''),
        auto_state=st.get('auto_state', ''),
        min_front=st.get('min_distance_front', 0),
        min_back=st.get('min_distance_back', 0),
        recording=recorder.is_recording() if recorder else False,
        samples=recorder.samples_written if recorder else 0,
    )


@app.route('/record/toggle')
def record_toggle():
    if recorder:
        is_rec = recorder.toggle()
        return jsonify(recording=is_rec, samples=recorder.samples_written)
    return jsonify(recording=False, samples=0)


def gen_frames():
    while True:
        if camera is None:
            time.sleep(0.1)
            continue
        frame, ts = camera.read()
        if frame is None:
            time.sleep(0.05)
            continue
        small = cv2.resize(frame, (480, 360))
        _, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 60])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.066)  # ~15 fps


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def main():
    global pi_client, camera, recorder

    p = argparse.ArgumentParser()
    p.add_argument('--pi', required=True, help='Pi IP address')
    p.add_argument('--port', type=int, default=8080, help='Web server port')
    p.add_argument('--camera', type=int, default=0, help='Camera device index')
    p.add_argument('--data-dir', default=os.path.join(os.path.dirname(__file__), 'data'),
                   help='Data output directory')
    args = p.parse_args()

    # Start camera
    camera = Camera(device=args.camera)
    if not camera.start():
        print("Camera failed to start")
        return

    # Connect to Pi
    pi_client = PiClient(args.pi)
    if not pi_client.connect():
        print("Cannot connect to Pi")
        camera.stop()
        return
    pi_client.start()

    # Start recorder
    recorder = DataRecorder(args.data_dir)
    threading.Thread(target=recorder.record_loop, daemon=True).start()

    print("=" * 55)
    print("  WEB VEHICLE CONTROL + RECORDER")
    print("=" * 55)
    print(f"  Open in browser: http://0.0.0.0:{args.port}")
    print(f"  Pi: {args.pi}")
    print(f"  Camera: device {args.camera}")
    print(f"  Data: {args.data_dir}")
    print(f"  Existing samples: {recorder.samples_written}")
    print("=" * 55)

    try:
        app.run(host='0.0.0.0', port=args.port, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        recorder.close()
        pi_client.close()
        camera.stop()
        print("\nStopped.")


if __name__ == '__main__':
    main()
