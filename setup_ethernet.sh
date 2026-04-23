#!/bin/bash
# Setup Ethernet Connection for Vehicle Control
# Run this if ethernet connection is lost

echo "Setting up ethernet connection..."

# Add IP address to eth0
sudo ip addr flush dev eth0
sudo ip addr add 192.168.1.100/24 dev eth0
sudo ip link set eth0 up

# Verify
echo ""
echo "Ethernet configuration:"
ip addr show eth0 | grep "inet "

# Test connection to laptop
echo ""
echo "Testing connection to laptop (192.168.1.101)..."
ping -c 2 192.168.1.101

echo ""
echo "✅ Ethernet setup complete!"
echo "Pi IP: 192.168.1.100"
echo "Laptop IP should be: 192.168.1.101"
