# Arduino Sensor Configuration Guide

## Your Sensor Setup

**Total: 6 sensors**
- **5 HC-SR04 sensors** connected to Arduino
- **1 Waterproof ultrasonic sensor** connected to Raspberry Pi GPIO

---

## Sensor Naming and Positions

### Arduino Sensors (5 sensors):

| Sensor Name | Position | Description |
|-------------|----------|-------------|
| **FL** | Front Left | HC-SR04 on front left |
| **FR** | Front Right | HC-SR04 on front right |
| **BC** | Back Center | HC-SR04 on back center |
| **LS** | Left Side | HC-SR04 on left side |
| **RS** | Right Side | HC-SR04 on right side |

### Raspberry Pi Sensor (1 sensor):

| Sensor Name | Position | GPIO Pins | Description |
|-------------|----------|-----------|-------------|
| **FW** | Front Waterproof (Center) | TRIG: GPIO 17 (Pin 11)<br>ECHO: GPIO 27 (Pin 13) | Waterproof sensor |

---

## Visual Layout

```
         FRONT OF VEHICLE

    FL          FW          FR
    👁️          👁️          👁️
   (Arduino)    (Pi)     (Arduino)

LS 👁️  [====== VEHICLE ======]  👁️ RS
 (Arduino)                    (Arduino)

              BC
              👁️
           (Arduino)

         BACK OF VEHICLE
```

---

## Arduino Code Modification

Your Arduino code needs to send data in this JSON format:

```json
{"FL":123.4,"FR":234.5,"BC":345.6,"LS":456.7,"RS":567.8}
```

### Example Arduino Code Structure:

```cpp
// Pin definitions
#define TRIG_FL 2   // Front Left trigger
#define ECHO_FL 3   // Front Left echo

#define TRIG_FR 4   // Front Right trigger
#define ECHO_FR 5   // Front Right echo

#define TRIG_BC 6   // Back Center trigger
#define ECHO_BC 7   // Back Center echo

#define TRIG_LS 8   // Left Side trigger
#define ECHO_LS 9   // Left Side echo

#define TRIG_RS 10  // Right Side trigger
#define ECHO_RS 11  // Right Side echo

void setup() {
  Serial.begin(115200);

  // Setup pins
  pinMode(TRIG_FL, OUTPUT);
  pinMode(ECHO_FL, INPUT);
  pinMode(TRIG_FR, OUTPUT);
  pinMode(ECHO_FR, INPUT);
  pinMode(TRIG_BC, OUTPUT);
  pinMode(ECHO_BC, INPUT);
  pinMode(TRIG_LS, OUTPUT);
  pinMode(ECHO_LS, INPUT);
  pinMode(TRIG_RS, OUTPUT);
  pinMode(ECHO_RS, INPUT);
}

void loop() {
  // Read all sensors
  float dist_FL = readUltrasonic(TRIG_FL, ECHO_FL);
  float dist_FR = readUltrasonic(TRIG_FR, ECHO_FR);
  float dist_BC = readUltrasonic(TRIG_BC, ECHO_BC);
  float dist_LS = readUltrasonic(TRIG_LS, ECHO_LS);
  float dist_RS = readUltrasonic(TRIG_RS, ECHO_RS);

  // Send JSON (IMPORTANT: Must match these exact names!)
  Serial.print("{");
  Serial.print("\"FL\":");
  Serial.print(dist_FL);
  Serial.print(",\"FR\":");
  Serial.print(dist_FR);
  Serial.print(",\"BC\":");
  Serial.print(dist_BC);
  Serial.print(",\"LS\":");
  Serial.print(dist_LS);
  Serial.print(",\"RS\":");
  Serial.print(dist_RS);
  Serial.println("}");

  delay(100);  // 10Hz update rate
}

float readUltrasonic(int trigPin, int echoPin) {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  long duration = pulseIn(echoPin, HIGH, 30000);  // 30ms timeout

  if (duration == 0) return 0;  // Timeout

  float distance = duration * 0.01715;  // Convert to cm
  return distance;
}
```

---

## Pin Assignment Table for Arduino

| Sensor | Arduino Pins | Physical Position |
|--------|--------------|-------------------|
| FL (Front Left) | D2 (TRIG), D3 (ECHO) | Front left corner |
| FR (Front Right) | D4 (TRIG), D5 (ECHO) | Front right corner |
| BC (Back Center) | D6 (TRIG), D7 (ECHO) | Back center |
| LS (Left Side) | D8 (TRIG), D9 (ECHO) | Left side (middle) |
| RS (Right Side) | D10 (TRIG), D11 (ECHO) | Right side (middle) |

**Note:** You can use different Arduino pins, but make sure to update your Arduino code accordingly!

---

## How the System Uses the Sensors

### Forward Movement:
- **Speed controlled by:** Minimum distance from **FL, FR, FW** (3 front sensors)
- **Ignores:** BC, LS, RS

### Backward Movement:
- **Speed controlled by:** Distance from **BC** (1 back sensor)
- **Ignores:** FL, FR, FW, LS, RS

### Side sensors (LS, RS):
- Currently **not used** for speed control
- Can be used for future features (e.g., lane keeping, obstacle avoidance)

---

## Testing Your Arduino Setup

### Test 1: Verify Serial Output

1. Upload Arduino code
2. Open Arduino Serial Monitor (115200 baud)
3. You should see:
   ```
   {"FL":123.4,"FR":234.5,"BC":345.6,"LS":456.7,"RS":567.8}
   {"FL":125.1,"FR":233.2,"BC":344.8,"LS":455.3,"RS":566.4}
   ...
   ```

### Test 2: Verify Sensor Names

**CRITICAL:** The JSON must use **exactly** these names:
- ✅ `"FL"` (correct)
- ❌ `"FrontLeft"` (wrong)
- ❌ `"front_left"` (wrong)
- ❌ `"fl"` (wrong - must be uppercase)

### Test 3: Verify with Raspberry Pi

On Raspberry Pi, run:
```bash
cd /home/sherry007/laptop_controlled_vehicle/raspberry_pi
sudo python3 obstacle_monitor.py
```

You should see output showing all 6 sensors (5 from Arduino + 1 from Pi):
```
Front:  FL=123.4  FW=234.5  FR=345.6
Back:   BC=456.7
Sides:  LS=567.8  RS=678.9
```

---

## Troubleshooting

### Problem: Pi doesn't receive Arduino data

**Check:**
1. Arduino is connected via USB
2. Serial port is correct (`/dev/ttyUSB0` or `/dev/ttyACM0`)
3. Baud rate is 115200 (both Arduino and Pi config)
4. Arduino code is uploaded and running

### Problem: Sensor names don't match

**Fix:** Update your Arduino code to use exact names: `FL`, `FR`, `BC`, `LS`, `RS`

### Problem: Some sensors read 0

**Check:**
1. Sensor wiring (VCC, GND, TRIG, ECHO)
2. Sensor power (5V)
3. Pin numbers in Arduino code match physical wiring

---

## Summary Checklist

- [ ] **5 HC-SR04 sensors** wired to Arduino
- [ ] **1 Waterproof sensor** wired to Raspberry Pi (GPIO 17, 27)
- [ ] Arduino code uses correct sensor names (`FL`, `FR`, `BC`, `LS`, `RS`)
- [ ] Arduino sends JSON at 115200 baud
- [ ] Arduino USB connected to Raspberry Pi
- [ ] Tested Arduino output in Serial Monitor
- [ ] Tested full system with `obstacle_monitor.py`

---

**Once all sensors are working, you're ready to run the full vehicle control system!**
