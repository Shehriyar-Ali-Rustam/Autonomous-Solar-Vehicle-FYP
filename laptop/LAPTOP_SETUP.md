# LAPTOP SIDE — The ML Brain

This is the **laptop half** of the autonomous car.
The laptop is the "brain": it has the camera, the object detector, and the
trained ML decision model. It talks to the Raspberry Pi over TCP.

---

## WHAT IS HERE (files explained)

```
laptop/
├── LAPTOP_SETUP.md                <-- you are here
├── requirements.txt               <-- Python libraries needed
│
├── vision/                        <-- everything that deals with the camera
│   ├── camera.py                  <-- opens your USB webcam, grabs frames in a thread
│   └── object_detector.py         <-- runs YOLOv8n on each frame (person, car, chair, dog, etc.)
│
├── ml/                            <-- the actual Machine Learning code
│   ├── actions.py                 <-- 8 possible actions the car can do
│   ├── model.py                   <-- neural network (MobileNetV3 + MLP head)
│   ├── dataset.py                 <-- loads your recorded data, augments it
│   ├── train.py                   <-- TRAINS the model (proper ML: train/val/test)
│   ├── evaluate.py                <-- accuracy, per-class F1, confusion matrix
│   ├── export.py                  <-- converts trained model to ONNX (fast inference)
│   └── inference.py               <-- loads the trained model at drive time
│
├── data_collection/
│   └── record.py                  <-- YOU drive the car with WASD, this records everything
│
├── main_autonomous.py             <-- THE MAIN FILE: camera -> ML -> Pi commands
├── remote_control_fixed.py        <-- old manual control (still works)
│
├── data/                          <-- (auto-created) recorded frames + CSV
└── models/                        <-- (auto-created) trained weights go here
```

---

## THE 8 ACTIONS THE ML MODEL PREDICTS

The model looks at camera + sensors and outputs ONE of these:

| ID | Action        | Meaning                                   |
|----|---------------|-------------------------------------------|
| 0  | FORWARD       | Go straight                               |
| 1  | SLOW_DOWN     | Go straight but slowly                    |
| 2  | TURN_LEFT     | Go forward while steering left            |
| 3  | TURN_RIGHT    | Go forward while steering right           |
| 4  | STOP          | Stop                                      |
| 5  | REVERSE_LEFT  | Back up while steering left               |
| 6  | REVERSE_RIGHT | Back up while steering right              |
| 7  | REVERSE       | Back up straight                          |

---

## WHAT THE ML MODEL ACTUALLY LEARNS

The neural network takes TWO inputs:

1. **Image** (from your USB camera, resized to 224x224)
2. **State vector** (19 numbers):
   - 6 ultrasonic distances (FL, FR, FW, BC, LS, RS)
   - 2 derived values (closest front, closest back)
   - 3 GPS values (valid flag, speed, heading)
   - 8 one-hot for the previous action

It outputs: **probabilities over the 8 actions**.

We pick the action with highest probability. If confidence < 55%, we send STOP
(the Pi's safety_governor is still the final say).

---

## HOW TO SET IT UP (first time only)

### 1. Install Python dependencies
Open a terminal on your laptop, go to the cloned repo folder, then:

```bash
cd laptop
python -m venv venv
source venv/bin/activate          # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

This installs: PyTorch, OpenCV, Ultralytics (YOLO), ONNXRuntime, TensorBoard, etc.
The first run of `object_detector.py` or `main_autonomous.py` will auto-download
the YOLOv8n weights (~6 MB).

### 2. Verify your webcam works
```bash
python -m laptop.vision.camera
```
A window should appear showing your webcam feed. Press `q` to quit.

### 3. Verify YOLO works (optional)
```bash
python -m laptop.vision.object_detector
```
A window should appear with green/red boxes around detected objects.

---

## THE FULL WORKFLOW (do this in order)

### STEP A — Start the Pi
On your Pi (via SSH):
```bash
cd Pi-4/raspberry_pi
python3 main.py
```
Leave it running. Note the Pi's IP address (e.g. 192.168.1.100).

### STEP B — Collect training data
On your LAPTOP:

```bash
cd /path/to/Pi-4
python -m laptop.data_collection.record --pi 192.168.1.100 --out laptop/data
```

Controls:
- `W` / Up arrow   — drive forward
- `S` / Down arrow — drive backward
- `A` / Left arrow — steer left (hold)
- `D` / Right arrow— steer right (hold)
- `X`              — steering straight
- `SPACE`          — STOP
- `+` / `-`        — adjust speed
- **`R`  — TOGGLE RECORDING** (must be ON to save data)
- `Q`/ESC          — quit

Every time you press `R` to enable recording, the script saves one sample
every 0.2 seconds: an image + sensor reading + your action.

**How much data?** Aim for **3,000-5,000 samples total** (~15-25 min of driving).
Drive in different rooms / outdoors, do lots of turns, stop in front of obstacles,
include people and furniture. Balanced data = better model.

Output:
- `laptop/data/images/*.jpg`    (each frame)
- `laptop/data/dataset.csv`     (one row per sample)

### STEP C — Train the model
```bash
python -m laptop.ml.train --csv laptop/data/dataset.csv --out laptop/models
```

This will:
1. Split data 70% train / 15% val / 15% test (fixed seed = reproducible)
2. Compute inverse-frequency class weights (so rare actions are not ignored)
3. Train in TWO stages:
   - Stage 1 (10 epochs): only the MLP head trains, MobileNetV3 frozen
   - Stage 2 (20 epochs): full fine-tune, cosine LR schedule
4. Save the best checkpoint (lowest val loss) to `laptop/models/best.pt`
5. Report test accuracy at the end

Track progress live:
```bash
tensorboard --logdir laptop/models
```
Open http://localhost:6006 in a browser.

**Target: test accuracy >= 85%.** If lower, collect more data.

### STEP D — Evaluate (look at per-class metrics)
```bash
python -m laptop.ml.evaluate \
    --ckpt laptop/models/best.pt \
    --csv  laptop/models/splits/test.csv
```

Prints: overall accuracy, precision / recall / F1 per action, inference latency.
Saves: `laptop/models/confusion_matrix.png` and `eval_metrics.json`.

**Red flag:** if one class (usually STOP or REVERSE*) has very low recall,
collect more examples of that action and retrain.

### STEP E — Export to ONNX (fast inference)
```bash
python -m laptop.ml.export \
    --ckpt laptop/models/best.pt \
    --out  laptop/models/model.onnx
```

ONNX runs ~2-3x faster than PyTorch on laptop CPU.

### STEP F — Drive autonomously!
```bash
python laptop/main_autonomous.py \
    --pi 192.168.1.100 \
    --model laptop/models/model.onnx
```

Controls:
- `G`     — GO (start autonomous driving)
- `SPACE` — pause / stop
- `Q`/ESC — quit

The script:
1. Grabs a camera frame (every 100 ms)
2. Reads sensor + GPS data from Pi
3. Runs the model → gets action + confidence
4. If confidence too low → sends STOP
5. Else → translates action to `FORWARD/STEER/speed` and sends to Pi

---

## IF IT MISBEHAVES

| Symptom                                   | Likely cause / fix                                       |
|-------------------------------------------|-----------------------------------------------------------|
| Car always STOPs                          | Low confidence. Try `--conf 0.4`. Collect more data.     |
| Car only goes forward                     | Dataset imbalanced — too few turn/stop examples          |
| Car swerves toward obstacles              | Camera angle wrong during recording vs driving           |
| "cannot connect to Pi"                    | Check Pi IP, firewall. Run `ping <PI_IP>` first          |
| Webcam shows wrong device                 | Pass `--camera 1` or `--camera 2`                        |
| Training crashes "CUDA out of memory"     | `--batch-size 16`. CPU works too (slower).               |

---

## THE SAFETY STORY (important)

Three layers protect you:

1. **ML confidence gate** (laptop) — if the model is < 55% sure, send STOP.
2. **Safety governor** (Pi, see `raspberry_pi/safety_governor.py`) — HARD stops:
   - Front sensor < 25 cm → STOP, no matter what the ML says
   - No laptop command for 2 s → STOP (watchdog)
   - All sensors reading zero for 3 s → STOP (sensor failure)
   - EMERGENCY command → STOP
3. **The big red button** — press SPACE on the laptop, or Ctrl+C on the Pi.

**You can never fully trust an ML model on real hardware.** That is why Layer 2
exists and why Layer 2 is HARDCODED rules that the ML cannot reach.

---

## HONEST SCOPE

This is a real ML pipeline, but:
- It is a **final-year-project level** autonomous vehicle.
- Expect ~85-90% action accuracy on held-out data, which is fine for slow
  indoor/outdoor driving but is NOT self-driving-car quality.
- The model imitates YOUR driving style. Drive carefully and consistently
  during data collection — the model will learn whatever you do.
- Calling this "Level 5 autonomy" would be a lie. Call it:
  *"ML-based autonomous obstacle avoidance with safety overrides"*.
  That is what professors/examiners want to hear, and it is the truth.
