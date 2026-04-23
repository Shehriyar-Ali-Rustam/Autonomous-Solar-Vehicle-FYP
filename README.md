# Laptop-Controlled Vehicle System

Remote control system for autonomous vehicle with obstacle avoidance and distance-based speed control.

## System Overview

```
Laptop ──(Ethernet)──> Raspberry Pi ──(Serial)──> Arduino ──> 6 Ultrasonic Sensors
                            │
                            └──(GPIO)──> IBT_2 Motor Driver ──> DC Motor
```

### Features

✅ **Remote Control** - Control vehicle from laptop via ethernet cable
✅ **Obstacle Avoidance** - Automatic speed reduction based on distance
✅ **Safety First** - Cannot override 30cm emergency stop distance
✅ **Real-time Status** - Live display of speeds, distances, and alerts
✅ **Watchdog Timer** - Auto-stop on connection loss
✅ **Emergency Stop** - ESC key for immediate halt
✅ **Dual Direction** - Obstacle detection for both forward and backward
✅ **Visual Alerts** - Color-coded warnings in CLI interface

---

## Hardware Requirements

### Raspberry Pi Side:
- Raspberry Pi (any model with GPIO)
- IBT_2 Motor Driver
- DC Motor
- Arduino Uno (running ultrasonic sensor code)
- 6x HC-SR04 Ultrasonic Sensors (connected to Arduino)
- Motor power supply (12V or 24V depending on motor)
- LM2596 buck converter (for 5V logic power)
- Jumper wires

### Laptop Side:
- Laptop with Python 3
- Ethernet cable (to connect to Raspberry Pi)

---

## Software Requirements

### Raspberry Pi:
```bash
python3              # Python 3.7+
python3-rpi.gpio     # Already installed
python3-serial       # Already installed (pyserial)
```

### Laptop:
```bash
python3              # Python 3.7+
pip install pynput   # Keyboard input handling
```

---

## Hardware Wiring

### IBT_2 Motor Driver to Raspberry Pi:

| IBT_2 Pin | Raspberry Pi GPIO | Physical Pin |
|-----------|-------------------|--------------|
| RPWM | GPIO 12 | Pin 32 |
| LPWM | GPIO 13 | Pin 33 |
| R_EN | GPIO 23 | Pin 16 |
| L_EN | GPIO 24 | Pin 18 |
| GND (control) | GND | Pin 6 |
| Vcc (control) | 5V from LM2596 | - |

### IBT_2 Motor Driver Power:

| IBT_2 Pin | Connection |
|-----------|------------|
| Vcc (motor power) | Motor Power Supply + (12V/24V) |
| GND (motor power) | Motor Power Supply - AND Common GND |
| B+ | Motor + terminal |
| B- | Motor - terminal |

### Arduino to Raspberry Pi:

| Arduino | Raspberry Pi |
|---------|--------------|
| USB | USB Port (appears as /dev/ttyUSB0 or /dev/ttyACM0) |

**Note:** The Arduino should already be running the ultrasonic sensor code from your autonomous vehicle project.

### Common Ground:
**CRITICAL:** All grounds must be connected together:
- Raspberry Pi GND
- IBT_2 GND (both power and control)
- Motor Power Supply GND
- LM2596 GND

---

## Installation & Setup

### 1. Raspberry Pi Setup

#### Copy files to Raspberry Pi:
```bash
# Option 1: If editing on Pi directly, files are already in place

# Option 2: If copying from laptop
scp -r laptop_controlled_vehicle/ pi@<PI_IP>:/home/pi/
```

#### Verify Arduino connection:
```bash
# Find Arduino serial port
ls /dev/ttyUSB* /dev/ttyACM*

# Should see /dev/ttyUSB0 or /dev/ttyACM0
# Update config.py if needed
```

#### Make scripts executable:
```bash
cd /home/sherry007/laptop_controlled_vehicle/raspberry_pi
chmod +x *.py
```

### 2. Laptop Setup

#### Install dependencies:
```bash
cd laptop_controlled_vehicle/laptop
pip install pynput
```

#### Make script executable:
```bash
chmod +x remote_control.py
```

### 3. Network Setup

#### Option A: Direct Ethernet Connection
1. Connect laptop to Pi with ethernet cable
2. Configure static IP on Pi (e.g., 192.168.1.100)
3. Configure static IP on laptop (e.g., 192.168.1.101, same subnet)

#### Option B: Same WiFi Network
1. Connect both Pi and laptop to same WiFi
2. Find Pi's IP address:
   ```bash
   hostname -I
   ```

---

## Usage

### Starting the System

#### 1. Start on Raspberry Pi:
```bash
cd /home/sherry007/laptop_controlled_vehicle/raspberry_pi
sudo python3 main.py
```

You should see:
```
╔════════════════════════════════════════════════════════════╗
║       LAPTOP-CONTROLLED VEHICLE CONTROL SYSTEM             ║
╚════════════════════════════════════════════════════════════╝

INITIALIZATION
Motor controller ready
Obstacle monitor ready
Remote server ready

Waiting for laptop connection...
Laptop should connect to: <Pi IP Address>:5555
```

#### 2. Start on Laptop:
```bash
cd laptop_controlled_vehicle/laptop
python3 remote_control.py
```

Enter Pi's IP address when prompted.

You should see:
```
╔════════════════════════════════════════════════════════════╗
║           VEHICLE REMOTE CONTROL                           ║
╠════════════════════════════════════════════════════════════╣
║  Status: FORWARD          Speed: 60%  (100% requested)     ║
║  Alert: WARNING - Obstacle detected!                       ║
╠════════════════════════════════════════════════════════════╣
...
```

### Controls

| Key | Action |
|-----|--------|
| ↑ (Up Arrow) | Move Forward |
| ↓ (Down Arrow) | Move Backward |
| SPACE | Stop |
| ESC | Emergency Stop & Quit |

---

## Safety Features

### 1. Distance-Based Speed Control

| Distance Range | Speed | Alert Level |
|----------------|-------|-------------|
| >= 300cm (3m) | 100% (Full Speed) | CLEAR (Green) |
| 100-299cm | 60% (Moderate) | WARNING (Yellow) |
| 30-99cm | 30% (Slow) | CRITICAL (Magenta) |
| < 30cm | 0% (STOP) | EMERGENCY (Red) |

### 2. Automatic Safety Features

- **Emergency Auto-Stop:** Vehicle stops automatically at 30cm regardless of user command
- **Watchdog Timer:** Stops motor if no command received for 2 seconds
- **Connection Loss:** Stops motor immediately if laptop disconnects
- **Sensor Validation:** Ignores invalid readings, uses minimum of valid sensors
- **Cannot Override:** User cannot override safety stop at 30cm

### 3. Direction-Specific Detection

- **Forward Movement:** Only front sensors (FL, FC, FR) control speed
- **Backward Movement:** Only back sensors (BL, BC, BR) control speed
- **Minimum Distance:** System uses closest sensor in direction of travel

---

## Testing

### Test 1: Motor Controller Only
```bash
cd /home/sherry007/laptop_controlled_vehicle/raspberry_pi
sudo python3 motor_controller.py
```
Motor should run forward 2 sec, stop, backward 2 sec, stop.

### Test 2: Obstacle Monitor Only
```bash
sudo python3 obstacle_monitor.py
```
Should display real-time sensor distances and calculated speeds.

### Test 3: Remote Server Only
```bash
sudo python3 remote_server.py
```
Server should start and wait for connections.

### Test 4: Full System
Follow "Usage" section above.

---

## Troubleshooting

### Pi can't find Arduino
```bash
# Check USB connection
ls /dev/ttyUSB* /dev/ttyACM*

# Try different port in config.py
nano config.py
# Change SERIAL_PORT = '/dev/ttyACM0'
```

### Motor doesn't move
1. Check all GPIO connections to IBT_2
2. Verify motor power supply is on
3. Check LM2596 output is 5V
4. Verify all grounds are connected together
5. Test with motor_controller.py standalone

### Laptop can't connect
1. Verify Pi's IP address: `hostname -I`
2. Check ethernet cable connection
3. Verify Pi and laptop are on same network/subnet
4. Check firewall isn't blocking port 5555
5. Verify main.py is running on Pi

### Sensors read 0 or invalid
1. Check Arduino USB connection
2. Verify Arduino is running ultrasonic_sensors.ino
3. Check sensor wiring to Arduino
4. Test sensors separately with Arduino Serial Monitor

### Speed is always reduced
1. Check for obstacles near sensors
2. Verify sensor readings with obstacle_monitor.py test
3. Adjust distance thresholds in config.py if needed

---

## Configuration

Edit `raspberry_pi/config.py` to customize:

```python
# Distance Thresholds (centimeters)
DISTANCE_SAFE = 300      # >= 300cm: Full speed
DISTANCE_WARNING = 100   # 100-299cm: Reduced speed
DISTANCE_CRITICAL = 30   # 30-99cm: Very slow
DISTANCE_EMERGENCY = 30  # < 30cm: STOP

# Speed Settings (percentage)
SPEED_FULL = 100         # Full speed
SPEED_MODERATE = 60      # Moderate speed
SPEED_SLOW = 30          # Slow speed

# Safety Settings
WATCHDOG_TIMEOUT = 2.0   # Seconds - stop if no command
```

---

## File Structure

```
laptop_controlled_vehicle/
├── raspberry_pi/
│   ├── config.py              # Configuration constants
│   ├── motor_controller.py    # IBT_2 motor control
│   ├── obstacle_monitor.py    # Distance monitoring
│   ├── remote_server.py       # Socket server
│   └── main.py                # Main integration
├── laptop/
│   └── remote_control.py      # CLI remote control
└── README.md                  # This file
```

---

## Related Projects

This system integrates with your existing autonomous vehicle project:
- Arduino sensor code: `/home/sherry007/AUTONOMOUS SOLAR VEHICLE/arduino_sensor_code/`
- Sensor reader library: `/home/sherry007/AUTONOMOUS SOLAR VEHICLE/raspberry_pi/sensor_reader.py`

---

## Support

For issues or questions:
1. Check troubleshooting section above
2. Verify all hardware connections
3. Test each component individually
4. Check system logs for error messages

---

## Safety Warning

⚠️ **IMPORTANT SAFETY NOTES:**

1. **Always supervise** the vehicle during operation
2. **Test in safe area** away from people, pets, and obstacles
3. **Emergency stop** is always available (ESC key)
4. **Never disable** safety features
5. **Check connections** before each use
6. **Start slowly** - test at low speeds first

---

## License

This project is for educational purposes.

---

**Created:** February 2026
**Last Updated:** February 2026
**Version:** 1.0
