#!/usr/bin/env python3
"""
Laptop Remote Control
CLI interface with arrow keys to control vehicle remotely
"""

import socket
import json
import threading
import time
import sys
import os
from datetime import datetime

# Check if pynput is installed
try:
    from pynput import keyboard
except ImportError:
    print("ERROR: pynput library not installed")
    print("Please install it with: pip install pynput")
    sys.exit(1)

# Configuration (should match Raspberry Pi config)
PI_IP = input("Enter Raspberry Pi IP address (e.g., 192.168.1.100): ").strip()
PI_PORT = 5555

# Commands
CMD_FORWARD = 'FORWARD'
CMD_BACKWARD = 'BACKWARD'
CMD_STOP = 'STOP'
CMD_EMERGENCY = 'EMERGENCY'

# ANSI color codes
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'

class RemoteControl:
    """
    Remote control client for vehicle
    """

    def __init__(self, pi_ip, pi_port):
        """Initialize remote control"""
        self.pi_ip = pi_ip
        self.pi_port = pi_port
        self.sock = None
        self.connected = False
        self.running = True

        # Current state
        self.current_command = CMD_STOP
        self.status = None
        self.status_lock = threading.Lock()

        # Threads
        self.receive_thread = None
        self.display_thread = None

    def connect(self):
        """Connect to Raspberry Pi"""
        print(f"\nConnecting to Raspberry Pi at {self.pi_ip}:{self.pi_port}...")

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(5.0)
            self.sock.connect((self.pi_ip, self.pi_port))
            self.sock.settimeout(0.5)  # For receive operations
            self.connected = True
            print(f"{Colors.GREEN}✓ Connected!{Colors.RESET}\n")
            return True

        except Exception as e:
            print(f"{Colors.RED}✗ Connection failed: {e}{Colors.RESET}")
            self.connected = False
            return False

    def disconnect(self):
        """Disconnect from Raspberry Pi"""
        self.connected = False
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            self.sock = None

    def send_command(self, command):
        """Send command to Raspberry Pi"""
        if not self.connected:
            return False

        try:
            message = json.dumps({'command': command})
            self.sock.sendall(message.encode('utf-8'))
            self.current_command = command
            return True

        except Exception as e:
            # print(f"Error sending command: {e}")
            self.connected = False
            return False

    def receive_status(self):
        """Receive status updates from Pi (runs in thread)"""
        buffer = ""

        while self.running and self.connected:
            try:
                data = self.sock.recv(4096)
                if not data:
                    # Connection closed
                    self.connected = False
                    break

                buffer += data.decode('utf-8')

                # Process complete JSON messages (newline delimited)
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    try:
                        status = json.loads(line)
                        with self.status_lock:
                            self.status = status
                    except json.JSONDecodeError:
                        pass

            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    # print(f"Error receiving status: {e}")
                    self.connected = False
                break

    def get_alert_color(self, alert_level):
        """Get color for alert level"""
        if alert_level == 'CLEAR':
            return Colors.GREEN
        elif alert_level == 'WARNING':
            return Colors.YELLOW
        elif alert_level == 'CRITICAL':
            return Colors.MAGENTA
        elif alert_level == 'EMERGENCY':
            return Colors.RED
        else:
            return Colors.RESET

    def display_status(self):
        """Display status in terminal (runs in thread)"""
        while self.running:
            # Clear screen
            os.system('clear' if os.name != 'nt' else 'cls')

            # Header
            print(Colors.BOLD + "╔" + "═" * 62 + "╗" + Colors.RESET)
            print(Colors.BOLD + "║" + " " * 16 + "VEHICLE REMOTE CONTROL" + " " * 24 + "║" + Colors.RESET)
            print(Colors.BOLD + "╠" + "═" * 62 + "╣" + Colors.RESET)

            # Connection status
            if self.connected:
                conn_text = f"{Colors.GREEN}CONNECTED{Colors.RESET} to {self.pi_ip}"
            else:
                conn_text = f"{Colors.RED}DISCONNECTED{Colors.RESET}"

            print(f"║  Connection: {conn_text:45s}     ║")
            print(Colors.BOLD + "╠" + "═" * 62 + "╣" + Colors.RESET)

            # Status from Pi
            if self.status:
                with self.status_lock:
                    cmd = self.status.get('current_command', 'UNKNOWN')
                    actual_speed = self.status.get('actual_speed', 0)
                    requested_speed = self.status.get('requested_speed', 0)
                    alert = self.status.get('alert_level', 'UNKNOWN')
                    distances = self.status.get('distances', {})
                    min_front = self.status.get('min_distance_front', 0)
                    min_back = self.status.get('min_distance_back', 0)

                # Command and speed
                cmd_color = Colors.CYAN if cmd in [CMD_FORWARD, CMD_BACKWARD] else Colors.RESET
                print(f"║  Status: {cmd_color}{cmd:8s}{Colors.RESET}          "
                      f"Speed: {actual_speed:3d}%  (requested: {requested_speed:3d}%)   ║")

                # Alert
                alert_color = self.get_alert_color(alert)
                alert_text = f"{alert_color}{alert}{Colors.RESET}"
                padding = " " * (9 - len(alert))  # Adjust for ANSI codes
                print(f"║  Alert: {alert_text}{padding}                                             ║")

                print(Colors.BOLD + "╠" + "═" * 62 + "╣" + Colors.RESET)

                # Sensor distances
                print("║  Sensor Distances:                                         ║")

                fl = distances.get('FL', 0)
                fc = distances.get('FC', 0)
                fr = distances.get('FR', 0)
                bl = distances.get('BL', 0)
                bc = distances.get('BC', 0)
                br = distances.get('BR', 0)

                print(f"║    Front:  FL={fl:5.1f}cm  FC={fc:5.1f}cm  FR={fr:5.1f}cm  "
                      f"[MIN: {min_front:5.1f}cm] ║")
                print(f"║    Back:   BL={bl:5.1f}cm  BC={bc:5.1f}cm  BR={br:5.1f}cm  "
                      f"[MIN: {min_back:5.1f}cm] ║")

            else:
                print("║  Waiting for status data from Pi...                       ║")

            # Controls
            print(Colors.BOLD + "╠" + "═" * 62 + "╣" + Colors.RESET)
            print("║  Controls:                                                 ║")
            print("║    ↑ (Up Arrow)    : Move Forward                          ║")
            print("║    ↓ (Down Arrow)  : Move Backward                         ║")
            print("║    SPACE           : Stop                                  ║")
            print("║    ESC             : Emergency Stop & Quit                 ║")
            print(Colors.BOLD + "╚" + "═" * 62 + "╝" + Colors.RESET)

            # Current command indicator
            cmd_display = self.current_command
            if cmd_display == CMD_FORWARD:
                cmd_display = f"{Colors.CYAN}↑ FORWARD{Colors.RESET}"
            elif cmd_display == CMD_BACKWARD:
                cmd_display = f"{Colors.CYAN}↓ BACKWARD{Colors.RESET}"
            elif cmd_display == CMD_STOP:
                cmd_display = f"{Colors.YELLOW}■ STOPPED{Colors.RESET}"

            print(f"\nCurrent Command: {cmd_display}")

            time.sleep(0.1)  # Update display 10 times per second

    def on_key_press(self, key):
        """Handle key press events"""
        try:
            if key == keyboard.Key.up:
                # Forward
                self.send_command(CMD_FORWARD)
            elif key == keyboard.Key.down:
                # Backward
                self.send_command(CMD_BACKWARD)
            elif key == keyboard.Key.space:
                # Stop
                self.send_command(CMD_STOP)
            elif key == keyboard.Key.esc:
                # Emergency stop and quit
                print("\n\nEmergency stop - Exiting...")
                self.send_command(CMD_EMERGENCY)
                time.sleep(0.2)
                self.running = False
                return False  # Stop listener

        except Exception as e:
            pass

    def run(self):
        """Main run loop"""
        # Connect to Pi
        if not self.connect():
            print("\nFailed to connect. Please check:")
            print("1. Raspberry Pi IP address is correct")
            print("2. Raspberry Pi is running main.py")
            print("3. Network connection is working")
            return

        # Start receive thread
        self.receive_thread = threading.Thread(target=self.receive_status, daemon=True)
        self.receive_thread.start()

        # Start display thread
        self.display_thread = threading.Thread(target=self.display_status, daemon=True)
        self.display_thread.start()

        # Wait a moment for first status
        time.sleep(0.5)

        # Start keyboard listener
        print("Starting keyboard listener...")
        with keyboard.Listener(on_press=self.on_key_press) as listener:
            listener.join()

        # Cleanup
        print("\nStopping vehicle...")
        self.send_command(CMD_STOP)
        time.sleep(0.2)
        self.disconnect()
        print("Disconnected.\n")


def main():
    """Main entry point"""
    print()
    print("╔════════════════════════════════════════════════════════════╗")
    print("║                                                            ║")
    print("║           VEHICLE REMOTE CONTROL - LAPTOP SIDE             ║")
    print("║                                                            ║")
    print("╚════════════════════════════════════════════════════════════╝")

    if not PI_IP:
        print("\nERROR: No IP address provided")
        return

    try:
        controller = RemoteControl(PI_IP, PI_PORT)
        controller.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")

    print("\nGoodbye!\n")


if __name__ == "__main__":
    main()
