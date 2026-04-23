# ML-Based Autonomous Vehicle - Pipeline

Proper supervised-learning pipeline. Not hard-coded rules.

## Architecture

```
┌────────────────────────────────┐               ┌────────────────────────────┐
│ LAPTOP (brain, this repo)      │      TCP      │ RASPBERRY PI (body)        │
│ ───────────────────────────    │ ◄──────────► │ ────────────────────────   │
│ USB webcam                     │  commands     │ 6 ultrasonic (Arduino)     │
│ YOLOv8n object detection       │               │ IBT_2 drive motor          │
│ PyTorch decision model         │  status       │ IBT_2 steering motor       │
│ ONNX runtime @ 30 FPS          │               │ GY-GPS6MV2 GPS             │
│ main_autonomous.py             │               │ safety_governor.py (hard)  │
└────────────────────────────────┘               └────────────────────────────┘
```

## One-time setup

On laptop:
```bash
cd laptop
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On Pi:
```bash
cd raspberry_pi
pip install pyserial RPi.GPIO pyserial
```

## Workflow

### 1. Start Pi
```bash
ssh pi@<PI_IP>
cd Pi-4/raspberry_pi
python3 main.py
```

### 2. Collect training data (on laptop)
Drive the car manually for 20-30 min using WASD. Aim for 3,000-5,000 samples,
distributed across actions (don't only drive forward).

```bash
python -m laptop.data_collection.record --pi <PI_IP> --out laptop/data
```
Press **R** to toggle recording. Press **W/A/S/D** / arrows to drive.

Tips for good data:
- Record in multiple rooms / outdoor scenes
- Include obstacles, walls, people, turning scenarios
- Roughly balance FORWARD / TURN_LEFT / TURN_RIGHT / STOP

### 3. Train the model
```bash
python -m laptop.ml.train --csv laptop/data/dataset.csv --out laptop/models
```

Produces:
- `laptop/models/best.pt`  (best validation checkpoint)
- `laptop/models/splits/{train,val,test}.csv`
- TensorBoard logs at `laptop/models/tb_<timestamp>/`

Watch training:
```bash
tensorboard --logdir laptop/models
```

### 4. Evaluate
```bash
python -m laptop.ml.evaluate --ckpt laptop/models/best.pt \
                             --csv  laptop/models/splits/test.csv
```
Reports: accuracy, per-class precision/recall/F1, confusion matrix,
inference latency. Target: **>= 85% test accuracy** before deployment.

### 5. Export to ONNX (fast inference)
```bash
python -m laptop.ml.export --ckpt laptop/models/best.pt \
                           --out  laptop/models/model.onnx
```

### 6. Run autonomous
```bash
python laptop/main_autonomous.py --pi <PI_IP> --model laptop/models/model.onnx
```
Press **G** to start driving. **SPACE** to pause. **Q** to quit.

## ML Specification

### Inputs
- **Image** (3, 224, 224) RGB, ImageNet-normalized
- **State vector** (19-dim):
  - 6 ultrasonic distances normalized [0, 1]
  - 2 derived (front_min, back_min) normalized [0, 1]
  - 3 GPS: valid flag, speed_mps normalized, heading_rad normalized
  - 8 one-hot previous action

### Output
8-class classification over actions:
`FORWARD, SLOW_DOWN, TURN_LEFT, TURN_RIGHT, STOP, REVERSE_LEFT, REVERSE_RIGHT, REVERSE`

### Model
Two-stream CNN + MLP:
- Vision: MobileNetV3-Small (pretrained on ImageNet, 576-dim features)
- Fusion head: Linear(595 -> 256) + Dropout + Linear(256 -> 128) + Linear(128 -> 8)

### Training (industry-standard)
- Loss: `CrossEntropyLoss` with inverse-frequency class weights + label smoothing 0.05
- Optimizer: `AdamW`, weight_decay 1e-4
- LR schedule: `CosineAnnealingLR`
- Two-stage: 10 epochs head-only, then 20 epochs full fine-tune with 10x lower LR on backbone
- Augmentation: horizontal flip with action re-mapping, color jitter, sensor gaussian noise
- Early stopping on val_loss, patience 6
- Train/val/test: 70/15/15 random split, seeded

### Deployment
- Model exported to ONNX (opset 17)
- Runs on laptop CPU at 30+ FPS (MobileNetV3 is tiny)
- Pi's `safety_governor.py` provides hard overrides (ML never overrides hardware safety)

## Safety Architecture

```
┌─ Laptop ML (Layer 1) ───────────┐
│ Decision model predicts action  │
│ If confidence < 0.55 -> STOP    │
└─────────┬───────────────────────┘
          │ command via TCP
          ▼
┌─ Pi safety_governor (Layer 2) ──┐
│ HARD checks that ML CANNOT      │
│ override:                       │
│  - front < 25cm -> STOP         │
│  - back  < 30cm -> STOP (rev)   │
│  - watchdog timeout -> STOP     │
│  - sensor failure -> STOP       │
│  - EMERGENCY cmd -> STOP        │
└─────────┬───────────────────────┘
          │ applied command
          ▼
      motors + steering
```
