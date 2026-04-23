# Complete Pin Connection Guide

## ALL Pin Connections in One Place

---

## 1. IBT_2 Drive Motor Driver → Raspberry Pi

### Control Wires (5 connections):

| Wire # | IBT_2 Pin | → | Raspberry Pi | Physical Pin | Description |
|--------|-----------|---|--------------|--------------|-------------|
| 1 | **RPWM** | → | **GPIO 4** | **Pin 7** | Forward PWM signal |
| 2 | **LPWM** | → | **GPIO 17** | **Pin 11** | Backward PWM signal |
| 3 | **R_EN** | → | **GPIO 18** | **Pin 12** | Right Enable |
| 4 | **L_EN** | → | **GPIO 27** | **Pin 13** | Left Enable |
| 5 | **GND** | → | **GND** | **Pin 6** | Ground |

### Power Connections:

| IBT_2 Pin | Connect To | Description |
|-----------|------------|-------------|
| **Vcc (5V small pin)** | **LM2596 OUT+** | 5V logic power from LM2596 |
| **Vcc (big terminal)** | **Power Supply +** | Motor power (12V or 24V) |
| **GND (big terminal)** | **Power Supply -** | Motor power ground |
| **B+** | **Drive Motor +** | Motor positive wire (Red) |
| **B-** | **Drive Motor -** | Motor negative wire (Black) |

---

## 2. IBT_2 Steering Motor Driver → Raspberry Pi

### Control Wires (5 connections):

| Wire # | IBT_2 Pin | → | Raspberry Pi | Physical Pin | Description |
|--------|-----------|---|--------------|--------------|-------------|
| 1 | **RPWM** | → | **GPIO 22** | **Pin 15** | Steer Right PWM signal |
| 2 | **LPWM** | → | **GPIO 23** | **Pin 16** | Steer Left PWM signal |
| 3 | **R_EN** | → | **GPIO 24** | **Pin 18** | Right Enable |
| 4 | **L_EN** | → | **GPIO 25** | **Pin 22** | Left Enable |
| 5 | **GND** | → | **GND** | **Pin 14** | Ground |

### Power Connections:

| IBT_2 Pin | Connect To | Description |
|-----------|------------|-------------|
| **Vcc (5V small pin)** | **LM2596 OUT+** | 5V logic power from LM2596 |
| **Vcc (big terminal)** | **Power Supply +** | Motor power (12V or 24V) |
| **GND (big terminal)** | **Power Supply -** | Motor power ground |
| **B+** | **Steering Motor +** | Motor positive wire (Red) |
| **B-** | **Steering Motor -** | Motor negative wire (Black) |

---

## 3. Arduino → Raspberry Pi

| Arduino | → | Raspberry Pi |
|---------|---|--------------|
| **USB Port** | → | **USB Port** | (will appear as /dev/ttyUSB0 or /dev/ttyUSB1) |

---

## 4. Arduino → 5 HC-SR04 Sensors

**Your Arduino code should connect:**

| Sensor Position | Sensor Name | Arduino TRIG Pin | Arduino ECHO Pin |
|-----------------|-------------|------------------|------------------|
| Front Left | **FL** | D2 | D3 |
| Front Right | **FR** | D4 | D5 |
| Back Center | **BC** | D6 | D7 |
| Left Side | **LS** | D8 | D9 |
| Right Side | **RS** | D10 | D11 |

**Note:** Each HC-SR04 also needs:
- **VCC** → Arduino 5V
- **GND** → Arduino GND

---

## 5. LM2596 Buck Converter

### Input:

| LM2596 Pin | Connect To |
|------------|------------|
| **IN+** | Power Supply + (12V/24V) |
| **IN-** | Power Supply - (GND) |

### Output (adjust to 5V with potentiometer):

| LM2596 Pin | Connect To |
|------------|------------|
| **OUT+** | Both IBT_2 Vcc (5V logic pins) |
| **OUT-** | Common Ground |

---

## Complete Wiring Diagram

```
POWER SUPPLY (12V/24V)
        |
    ┌───┴───┐
   (+)     (-)
    │       │
    │       └────────────────────────────┐
    │                                    │
    ├──> LM2596 IN+                     │
    │                                    │
    ├──> IBT_2 Drive Motor Power Vcc    │
    │                                    │
    └──> IBT_2 Steering Motor Power Vcc │
                                         │
LM2596                              COMMON GND
  OUT+ (5V) ──> Both IBT_2 Logic Vcc    │
  OUT- ────────────────────────────────>│


RASPBERRY PI (40-pin header)               COMMON GND
════════════════════════════                    │
Pin 6  [GND]      ──> IBT_2 Drive GND ────────>│
Pin 7  [GPIO 4]   ──> IBT_2 Drive RPWM         │
Pin 11 [GPIO 17]  ──> IBT_2 Drive LPWM         │
Pin 12 [GPIO 18]  ──> IBT_2 Drive R_EN         │
Pin 13 [GPIO 27]  ──> IBT_2 Drive L_EN         │
Pin 14 [GND]      ──> IBT_2 Steering GND ─────>│
Pin 15 [GPIO 22]  ──> IBT_2 Steering RPWM      │
Pin 16 [GPIO 23]  ──> IBT_2 Steering LPWM      │
Pin 18 [GPIO 24]  ──> IBT_2 Steering R_EN      │
Pin 22 [GPIO 25]  ──> IBT_2 Steering L_EN      │
[USB Port]        ──> Arduino USB              │
                                                │
                                                │
IBT_2 DRIVE MOTOR DRIVER                        │
═══════════════════════                         │
RPWM  <─── Pi GPIO 4  (Pin 7)                  │
LPWM  <─── Pi GPIO 17 (Pin 11)                 │
R_EN  <─── Pi GPIO 18 (Pin 12)                 │
L_EN  <─── Pi GPIO 27 (Pin 13)                 │
GND   <─── Pi GND (Pin 6) ────────────────────>│
Vcc(5V) <─ LM2596 5V                           │
Vcc(12V) <─ Power Supply +                     │
GND(pwr) <─ Power Supply - ───────────────────>│
B+    ──> Drive Motor +                         │
B-    ──> Drive Motor -                         │
                                                │
                                                │
IBT_2 STEERING MOTOR DRIVER                     │
═══════════════════════════                     │
RPWM  <─── Pi GPIO 22 (Pin 15)                 │
LPWM  <─── Pi GPIO 23 (Pin 16)                 │
R_EN  <─── Pi GPIO 24 (Pin 18)                 │
L_EN  <─── Pi GPIO 25 (Pin 22)                 │
GND   <─── Pi GND (Pin 14) ───────────────────>│
Vcc(5V) <─ LM2596 5V                           │
Vcc(12V) <─ Power Supply +                     │
GND(pwr) <─ Power Supply - ───────────────────>│
B+    ──> Steering Motor +                      │
B-    ──> Steering Motor -                      │


ARDUINO UNO                                     │
═══════════                                     │
D2  ──> FL Sensor TRIG                         │
D3  ──> FL Sensor ECHO                         │
D4  ──> FR Sensor TRIG                         │
D5  ──> FR Sensor ECHO                         │
D6  ──> BC Sensor TRIG                         │
D7  ──> BC Sensor ECHO                         │
D8  ──> LS Sensor TRIG                         │
D9  ──> LS Sensor ECHO                         │
D10 ──> RS Sensor TRIG                         │
D11 ──> RS Sensor ECHO                         │
5V  ──> All 5 HC-SR04 VCC pins                 │
GND ──> All 5 HC-SR04 GND pins ────────────────>│
USB ──> Raspberry Pi USB                       │
```

---

## Sensor Layout on Vehicle

```
         FRONT OF VEHICLE

    FL          FW          FR
    (Arduino)  (Arduino)  (Arduino)
    D2/D3      D12/D13     D4/D5


LS  [====== VEHICLE ======]  RS
 (Arduino)                    (Arduino)
  D8/D9                        D10/D11



              BC
          (Arduino)
           D6/D7

         BACK OF VEHICLE
```

---

## GPIO Pin Summary

```
Pi Physical Pin Layout (pins used):

  Pin 6  [GND]     - Drive IBT_2 GND
  Pin 7  [GPIO 4]  - Drive RPWM (Forward)
  Pin 11 [GPIO 17] - Drive LPWM (Backward)
  Pin 12 [GPIO 18] - Drive R_EN
  Pin 13 [GPIO 27] - Drive L_EN
  Pin 14 [GND]     - Steering IBT_2 GND
  Pin 15 [GPIO 22] - Steering RPWM (Right)
  Pin 16 [GPIO 23] - Steering LPWM (Left)
  Pin 18 [GPIO 24] - Steering R_EN
  Pin 22 [GPIO 25] - Steering L_EN
```

---

## Ground Connections - CRITICAL!

**ALL these grounds MUST be connected together:**

- Raspberry Pi GND (Pin 6, 14, or any GND pin)
- IBT_2 Drive GND (control side)
- IBT_2 Drive GND (motor power side)
- IBT_2 Steering GND (control side)
- IBT_2 Steering GND (motor power side)
- Power Supply GND (-)
- LM2596 GND (OUT-)
- Arduino GND (connected via USB already)

**Use a terminal block or breadboard to connect all grounds to one point!**

---

## Quick Checklist

### Before Powering On:

- [ ] All 5 Drive IBT_2 control wires connected to Pi (Pin 7, 11, 12, 13, 6)
- [ ] All 5 Steering IBT_2 control wires connected to Pi (Pin 15, 16, 18, 22, 14)
- [ ] Both IBT_2 logic power from LM2596 5V output
- [ ] Both IBT_2 motor power from main power supply (12V/24V)
- [ ] Drive motor connected to Drive IBT_2 B+ and B-
- [ ] Steering motor connected to Steering IBT_2 B+ and B-
- [ ] Arduino USB connected to Pi
- [ ] Arduino has 5 HC-SR04 sensors wired
- [ ] LM2596 adjusted to output exactly 5.0V
- [ ] **ALL GROUNDS CONNECTED TOGETHER**

### After Powering On:

- [ ] LM2596 outputs 5V (verify with multimeter)
- [ ] Arduino power LED on
- [ ] No smoke, burning smell, or excessive heat

---

## Troubleshooting Pin Connections

### Drive motor doesn't move:
- Check all 5 Drive IBT_2 control wires (Pin 7, 11, 12, 13, 6)
- Verify GPIO pin numbers in code match physical connections
- Check motor power supply is on
- Verify LM2596 outputs 5V

### Steering motor doesn't move:
- Check all 5 Steering IBT_2 control wires (Pin 15, 16, 18, 22, 14)
- Verify GPIO pin numbers in code match physical connections
- Check motor power supply is on
- Verify LM2596 outputs 5V

### Arduino sensors don't work:
- Check USB connection
- Verify serial port (/dev/ttyUSB0 or /dev/ttyUSB1)
- Check Arduino code is uploaded
- Verify Arduino HC-SR04 wiring
- Test with Arduino Serial Monitor (115200 baud)

---

**Print this page and keep it with your project!**
