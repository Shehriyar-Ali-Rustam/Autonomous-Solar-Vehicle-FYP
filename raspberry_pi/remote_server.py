#!/usr/bin/env python3
"""
Remote Server
TCP socket server for receiving commands from laptop
"""

import socket
import json
import threading
import time
from datetime import datetime
import config

class RemoteServer:
    """
    TCP socket server for remote control from laptop
    """

    def __init__(self):
        """Initialize remote server"""
        self.server_socket = None
        self.client_socket = None
        self.client_address = None
        self.is_running = False
        self.is_client_connected = False

        # Latest commands from client
        self.latest_command = config.CMD_STOP
        self.latest_steer = config.CMD_STEER_STOP
        self.latest_speed = 0  # 0-100, set by ML or manual
        self.command_timestamp = time.time()
        self.lock = threading.Lock()

        # Server thread
        self.server_thread = None
        self.client_thread = None

    def start_server(self):
        """Start TCP server"""
        if self.is_running:
            print("[RemoteServer] Server already running")
            return

        # Create socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            self.server_socket.bind((config.SERVER_HOST, config.SERVER_PORT))
            self.server_socket.listen(1)
            self.is_running = True

            print(f"[RemoteServer] Server started on {config.SERVER_HOST}:{config.SERVER_PORT}")
            print("[RemoteServer] Waiting for laptop connection...")

            # Start server thread
            self.server_thread = threading.Thread(target=self._accept_connections, daemon=True)
            self.server_thread.start()

        except Exception as e:
            print(f"[RemoteServer] Failed to start server: {e}")
            self.is_running = False
            raise

    def _accept_connections(self):
        """Accept incoming client connections (runs in thread)"""
        while self.is_running:
            try:
                # Accept connection (blocking)
                self.server_socket.settimeout(1.0)  # Timeout to check is_running flag
                try:
                    client_sock, client_addr = self.server_socket.accept()
                except socket.timeout:
                    continue

                print(f"[RemoteServer] Client connected from {client_addr}")

                with self.lock:
                    # Close previous client if exists
                    if self.client_socket:
                        try:
                            self.client_socket.close()
                        except:
                            pass

                    self.client_socket = client_sock
                    self.client_address = client_addr
                    self.is_client_connected = True

                # Handle client in separate thread
                self.client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_sock, client_addr),
                    daemon=True
                )
                self.client_thread.start()

            except Exception as e:
                if self.is_running:
                    print(f"[RemoteServer] Error accepting connection: {e}")
                time.sleep(1)

    def _handle_client(self, client_sock, client_addr):
        """Handle client connection (runs in thread)"""
        buffer = ""
        try:
            while self.is_running and self.is_client_connected:
                try:
                    client_sock.settimeout(0.5)
                    try:
                        data = client_sock.recv(4096)
                    except socket.timeout:
                        continue

                    if not data:
                        print(f"[RemoteServer] Client {client_addr} disconnected")
                        break

                    buffer += data.decode('utf-8')

                    # Try to parse each JSON object in the buffer
                    # Messages may arrive concatenated like: {"command":"FORWARD"}{"command":"LEFT"}
                    while buffer:
                        buffer = buffer.strip()
                        if not buffer:
                            break

                        # Find the end of first JSON object
                        depth = 0
                        end_idx = -1
                        for i, ch in enumerate(buffer):
                            if ch == '{':
                                depth += 1
                            elif ch == '}':
                                depth -= 1
                                if depth == 0:
                                    end_idx = i
                                    break

                        if end_idx == -1:
                            break  # Incomplete JSON, wait for more data

                        json_str = buffer[:end_idx + 1]
                        buffer = buffer[end_idx + 1:]

                        try:
                            message = json.loads(json_str)
                            command = message.get('command', config.CMD_STOP)
                            steer = message.get('steer', None)
                            speed = message.get('speed', None)

                            with self.lock:
                                if command in config.VALID_COMMANDS:
                                    self.latest_command = command
                                    self.command_timestamp = time.time()
                                if steer in [config.CMD_LEFT, config.CMD_RIGHT, config.CMD_STEER_STOP]:
                                    self.latest_steer = steer
                                if speed is not None:
                                    try:
                                        self.latest_speed = max(0, min(100, int(speed)))
                                    except (TypeError, ValueError):
                                        pass

                        except json.JSONDecodeError:
                            pass

                except Exception as e:
                    if self.is_running:
                        print(f"[RemoteServer] Error handling client: {e}")
                    break

        finally:
            with self.lock:
                self.is_client_connected = False
                self.latest_command = config.CMD_STOP
                try:
                    client_sock.close()
                except:
                    pass
            print(f"[RemoteServer] Connection closed: {client_addr}")

    def get_latest_command(self):
        """
        Get latest command from client

        Returns:
            str: Latest command (FORWARD, BACKWARD, STOP, etc.)

        Note: Implements watchdog - returns STOP if no command received recently
        """
        with self.lock:
            # Check watchdog timeout
            if time.time() - self.command_timestamp > config.WATCHDOG_TIMEOUT:
                if self.latest_command != config.CMD_STOP:
                    print("[RemoteServer] Watchdog timeout - stopping for safety")
                    self.latest_command = config.CMD_STOP

            return self.latest_command

    def get_latest_steer(self):
        """Get latest steer command from client"""
        with self.lock:
            return self.latest_steer

    def get_latest_speed(self):
        """Get latest speed (0-100) from client. Default 50 if no speed sent."""
        with self.lock:
            return self.latest_speed if self.latest_speed > 0 else 50

    def get_command_timestamp(self):
        """Return Unix timestamp of last command received."""
        with self.lock:
            return self.command_timestamp

    def reset_command(self):
        """Reset command to STOP (used after processing one-shot commands like AUTONOMOUS)"""
        with self.lock:
            self.latest_command = config.CMD_STOP
            self.command_timestamp = time.time()

    def send_status(self, status_dict):
        """
        Send status update to connected client

        Args:
            status_dict (dict): Status information to send
        """
        if not self.is_client_connected or not self.client_socket:
            return

        try:
            # Add timestamp
            status_dict['timestamp'] = datetime.now().isoformat()

            # Convert to JSON and send
            message = json.dumps(status_dict) + '\n'
            self.client_socket.sendall(message.encode('utf-8'))

        except Exception as e:
            # Client probably disconnected
            # print(f"[RemoteServer] Error sending status: {e}")
            pass

    def is_connected(self):
        """
        Check if client is connected

        Returns:
            bool: True if client is connected
        """
        with self.lock:
            return self.is_client_connected

    def stop_server(self):
        """Stop server and close all connections"""
        print("[RemoteServer] Stopping server...")
        self.is_running = False

        # Close client connection
        with self.lock:
            if self.client_socket:
                try:
                    self.client_socket.close()
                except:
                    pass
                self.client_socket = None
            self.is_client_connected = False

        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
            self.server_socket = None

        # Wait for threads to finish
        if self.server_thread:
            self.server_thread.join(timeout=2.0)
        if self.client_thread:
            self.client_thread.join(timeout=2.0)

        print("[RemoteServer] Server stopped")


# Test code - run this file directly to test remote server
if __name__ == "__main__":
    print("Testing Remote Server...")
    print(f"Server will listen on port {config.SERVER_PORT}")
    print("Use laptop/remote_control.py to connect\n")

    server = RemoteServer()

    try:
        server.start_server()

        print("Server running. Monitoring commands for 60 seconds...")
        print("Connect from laptop and send commands\n")

        for i in range(60):
            cmd = server.get_latest_command()
            connected = server.is_connected()

            print(f"[{i+1}/60] Connected: {connected}, Command: {cmd}")

            # Simulate sending status back
            if connected:
                status = {
                    'current_command': cmd,
                    'actual_speed': 50,
                    'test_message': f'Status update {i+1}'
                }
                server.send_status(status)

            time.sleep(1)

        print("\nTest complete!")

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        server.stop_server()
