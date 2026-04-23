#!/bin/bash
# Ethernet Connection Keepalive Script
# Monitors and maintains ethernet connection

while true; do
    # Check if connection is up
    if ! nmcli connection show --active | grep -q "vehicle-ethernet"; then
        echo "[$(date)] Connection down, bringing up..."
        nmcli connection up vehicle-ethernet
    fi

    # Check if IP is assigned
    if ! ip addr show eth0 | grep -q "192.168.1.100"; then
        echo "[$(date)] IP missing, reassigning..."
        ip addr add 192.168.1.100/24 dev eth0 2>/dev/null
        ip link set eth0 up
    fi

    # Wait before next check
    sleep 5
done
