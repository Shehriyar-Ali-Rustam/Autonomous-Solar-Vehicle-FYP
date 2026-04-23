# RASPBERRY PI SIDE — The Car's Body

This is the **Pi half** of the autonomous car.
The Pi is the "body": it reads sensors, drives motors, and talks to the laptop
(the brain) over TCP. The Pi does NOT do ML. The Pi's job is to:

1. Read sensors (ultrasonic via Arduino + GPS)
2. Send sensor data to the laptop
3. Receive drive commands from the laptop
4. Check HARD safety rules before applying anything
5. Apply commands to motors and steering

---

## WHAT IS HERE (files explained)

```
raspberry_pi/
├── PI_SETUP.md                    <-- you are here
├── config.py                      <-- all GPIO pin numbers, ports, timings
│
├── main.py                        <-- THE MAIN FILE. Starts everything and runs the loop
│
├── motor_controller.py            <-- controls the IBT_2 drive motor (rear wheels)
├── steering_controller.py         <-- controls the IBT_2 steering motor
│
├── obstacle_monitor.py            <-- reads the Arduino (6 ultrasonic sensors)
├── gps_reader.py                  <-- reads the GY-GPS6MV2 GPS over UART
│
├── remote_server.py               <-- TCP server on port 5555, talks to laptop
│
├── safety_governor.py             <-- HARD safety rules - ML cannot override these
│
└── (test scripts: test_gps.py, test_steering.py, test_both_motors.py)
```

---

## HARDWARE LAYOUT (as wired)

### Drive motor (IBT_2 #1) — controls REAR WHEELS
| Signal  | GPIO (BCM) | Pi Physical Pin |
|---------|------------|-----------------|
| RPWM    | GPIO 4     | Pin 7           |
| LPWM    | GPIO 17    | Pin 11          |
| R_EN    | GPIO 18    | Pin 12          |
| L_EN    | GPIO 27    | Pin 13          |

### Steering motor (IBT_2 #2) — controls FRONT WHEEL STEERING
| Signal  | GPIO (BCM) | Pi Physical Pin |
|---------|------------|-----------------|
| RPWM    | GPIO 22    | Pin 15          |
| LPWM    | GPIO 23    | Pin 16          |
| R_EN    | GPIO 24    | Pin 18          |
| L_EN    | GPIO 25    | Pin 22          |

### Arduino (6 ultrasonic sensors)
- Connected via USB → `/dev/ttyUSB0` (auto-detected) at 115200 baud
- Sensors: `FL`, `FR`, `FW` (front), `BC` (back), `LS`, `RS` (sides)

### GPS (GY-GPS6MV2 / NEO-6M)
| GPS Pin | Pi Pin                  |
|---------|-------------------------|
| VCC     | 5V  (Physical pin 2)    |
| GND     | GND (Physical pin 6)    |
| TX      | GPIO 15 / RX (Pin 10)   |
| RX      | GPIO 14 / TX (Pin 8)    |

Device path: `/dev/serial0` at 9600 baud.

To enable the serial port:
```bash
sudo raspi-config
# Interface Options → Serial Port
#   Login shell over serial? → NO
#   Serial port hardware enabled? → YES
sudo reboot
```

---

## WHAT EACH FILE ACTUALLY DOES

### `config.py`
All pin numbers, serial ports, TCP port (5555), distance thresholds, command
constants. Change pins here, nowhere else.

### `motor_controller.py`
Sets up PWM on the drive IBT_2 pins. Exposes `set_speed(direction, percent)`.
`direction` is `FORWARD` / `BACKWARD` / `STOP`.

### `steering_controller.py`
Same idea for the steering motor. `set_direction(LEFT / RIGHT / STEER_STOP)`.
Steering motor runs at fixed 70% speed (configured in `config.STEER_SPEED`).

### `obstacle_monitor.py`
Starts a background thread that keeps reading JSON from the Arduino.
Exposes `get_all_distances()` → dict of 6 sensor readings in cm.

### `gps_reader.py`  (NEW)
Opens `/dev/serial0`, parses `$GPRMC` and `$GPGGA` NMEA sentences, keeps the
latest fix (lat/lon/alt/speed/heading/#satellites/HDOP). Also has helper
functions `haversine_meters()` and `bearing_between()` for point-A-to-B math.

### `remote_server.py`
TCP server on port 5555. Accepts JSON messages from laptop:
```json
{"command": "FORWARD", "steer": "LEFT", "speed": 50}
```
- `command`: FORWARD / BACKWARD / STOP / EMERGENCY / AUTONOMOUS
- `steer`:   LEFT / RIGHT / STEER_STOP
- `speed`:   0-100 (optional; default 50)

Sends status back to laptop every loop (~20 Hz) as newline-terminated JSON
with `distances`, `gps`, `safety_violation`, etc.

### `safety_governor.py`  (NEW)
Runs every control loop. Given current sensors + requested command, decides
whether to allow the command or override with STOP. HARD rules:

| Violation              | Trigger                                    |
|------------------------|--------------------------------------------|
| CRITICAL_FRONT         | Front sensor < 25 cm while going forward  |
| CRITICAL_REAR          | Back sensor < 30 cm while reversing       |
| WATCHDOG_TIMEOUT       | No command from laptop for 2 s            |
| SENSOR_FAILURE         | All sensors invalid for > 3 s             |
| EMERGENCY              | Laptop sent CMD_EMERGENCY                  |

**These rules cannot be overridden by the laptop ML model.** They exist
specifically so that the ML cannot drive the car into a wall.

### `main.py`  (UPDATED)
The glue. Every 50 ms (20 Hz):
```
1. Read latest drive/steer/speed from laptop    (remote_server)
2. Read 6 ultrasonic distances                  (obstacle_monitor)
3. Check safety rules                           (safety_governor)
    if UNSAFE -> override with STOP
4. Apply to motors                              (motor / steering controller)
5. Send status (sensors + GPS) back to laptop   (remote_server.send_status)
```

---

## HOW TO RUN (on the Pi)

### One-time setup
```bash
# Install Python libs
pip3 install pyserial

# RPi.GPIO is usually already installed on Raspberry Pi OS
# If not: sudo apt install python3-rpi.gpio

# Make sure UART is enabled (see GPS section above)
```

### Every time you want to use the car
```bash
cd Pi-4/raspberry_pi
python3 main.py
```

You should see something like:
```
============================================================
VEHICLE CONTROL SYSTEM - Pi Side
============================================================
[1/5] Motor controller...
[2/5] Steering controller...
[3/5] Obstacle monitor (Arduino)...
[RemoteServer] Server started on 0.0.0.0:5555
[4/5] Remote server...
[5/5] GPS reader...
[GPSReader] Started on /dev/serial0 @ 9600
============================================================
READY. Waiting for laptop on port 5555
============================================================
```

Leave this running. Then start the laptop side.

### How to read the status log
Every 10 loops the Pi prints a line like:
```
[14:23:55] C | drive=FORWARD  steer=STEER_STOP spd= 60% | F= 87.0 B=150.0 | safety=NONE
```
- `C` = laptop connected, `D` = disconnected
- `drive / steer / spd` = what is actually being applied to the motors
  (after safety check — not necessarily what the laptop requested!)
- `F` = closest front sensor in cm, `B` = closest back sensor in cm
- `safety=NONE` means no override. If you see `CRITICAL_FRONT` the ML asked
  to drive forward but the Pi refused because the front sensor is too close.

---

## IF IT MISBEHAVES

| Symptom                                | Fix                                                      |
|----------------------------------------|----------------------------------------------------------|
| "Failed to connect to Arduino"         | Check USB cable, `ls /dev/ttyUSB*`, maybe need `sudo chmod 666 /dev/ttyUSB0` |
| "Failed to open /dev/serial0"          | Enable UART with `raspi-config` (see GPS section)        |
| GPS says "No fix yet"                  | GPS needs clear sky view, 30-120 s to get first fix      |
| All sensor readings 0                  | Arduino not running the right sketch, or serial garbled  |
| Motor not moving but no errors         | Usually safety_governor is triggering — check log for `safety=`... |
| "Port 5555 already in use"             | `sudo fuser -k 5555/tcp` then restart                    |

---

## TESTING INDIVIDUAL COMPONENTS

Each component has a test script you can run standalone:

```bash
python3 test_both_motors.py        # drives both IBT_2s briefly
python3 test_steering.py           # just steering
python3 gps_reader.py              # shows raw GPS readings for 30 s
python3 obstacle_monitor.py        # shows all 6 ultrasonic distances
python3 safety_governor.py         # runs test scenarios of safety rules
```

---

## SAFETY REMINDER

The Pi's `safety_governor.py` is your HARD safety layer. It is the last line
of defence between buggy ML and your vehicle hitting something.

**Never edit `safety_governor.py` to make it looser just because the car
"won't move" — first figure out why the sensors are reading what they are.**

If you genuinely need to disable safety (for debugging only):
```python
# In main.py, comment out the safety_governor.check() call temporarily
# BUT put it back before you run ML autonomously.
```
