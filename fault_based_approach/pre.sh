#!/bin/bash
# pre.sh — Minimal Experimental Bed Setup for NVIDIA Quadro P5000 VM
# Run as: bash pre.sh
# Requires sudo privileges (cs614 password will be prompted)

set -e  # Exit on any error

echo "============================================="
echo " NVIDIA Driver Setup Script"
echo " GPU: NVIDIA Quadro P5000 | Driver: 470.x"
echo "============================================="

# ─── STEP 1: Install build tools ─────────────────────────────────────────────
echo ""
echo "[Step 1] Installing build tools..."
sudo apt update -y
sudo apt install -y build-essential dkms git

echo ""
echo "[Step 1] Kernel version: $(uname -r)"
if [ ! -d "/lib/modules/$(uname -r)/build" ]; then
    echo "ERROR: Kernel headers not found at /lib/modules/$(uname -r)/build"
    exit 1
fi
echo "[Step 1] Kernel headers found. OK."

# ─── STEP 2: Install NVIDIA driver ───────────────────────────────────────────
echo ""
echo "[Step 2] Installing nvidia-driver-470..."
sudo apt install -y nvidia-driver-470

# ─── STEP 3: Fix .bashrc ─────────────────────────────────────────────────────
echo ""
echo "[Step 3] Cleaning up ~/.bashrc to remove conflicting LD_LIBRARY_PATH..."

# Remove any line referencing an NVIDIA installer directory
sed -i '/NVIDIA-Linux-x86_64/d' ~/.bashrc

# Ensure CUDA lib64 path is present but not duplicated
if ! grep -q "LD_LIBRARY_PATH=/usr/local/cuda/lib64" ~/.bashrc; then
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
fi

# Ensure cuda-toolkit bin is in PATH but not duplicated
if ! grep -q "cuda-toolkit/bin" ~/.bashrc; then
    echo 'export PATH=$HOME/data/cuda-toolkit/bin:$PATH' >> ~/.bashrc
fi

echo "[Step 3] .bashrc cleaned. OK."

# ─── STEP 4: Apply LD_LIBRARY_PATH fix for current session ───────────────────
echo ""
echo "[Step 4] Unsetting any conflicting LD_LIBRARY_PATH for this session..."
export LD_LIBRARY_PATH=/usr/local/cuda/lib64

# ─── STEP 5: Reboot prompt ───────────────────────────────────────────────────
echo ""
echo "============================================="
echo " Setup complete. A reboot is required."
echo " After reboot, run: nvidia-smi"
echo " Expected: Quadro P5000, Driver 470.256.02"
echo "============================================="
echo ""
read -p "Reboot now? [Y/n]: " REBOOT_CHOICE
REBOOT_CHOICE=${REBOOT_CHOICE:-Y}

if [[ "$REBOOT_CHOICE" =~ ^[Yy]$ ]]; then
    echo "Rebooting..."
    sudo reboot
else
    echo "Skipping reboot. Remember to reboot before using nvidia-smi."
fi
