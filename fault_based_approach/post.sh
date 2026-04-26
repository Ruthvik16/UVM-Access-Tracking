#!/bin/bash
# post.sh — Post-reboot: apply UVM sampling patch, build, and reload nvidia-uvm.
# Run as: bash artifacts/post.sh
# Requires: artifacts/uvm_changes.patch, uvm_sampling_tracker.h,
#           uvm_sampling_tracker.c, uvm_sampling_procfs.c

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="$SCRIPT_DIR/uvm_changes.patch"
UVM_SRC="/usr/src/nvidia-470.256.02"
UVM_DIR="$UVM_SRC/nvidia-uvm"

echo "============================================="
echo " Post-Reboot NVIDIA UVM Patch Script"
echo "============================================="

# ─── STEP 1: Fix LD_LIBRARY_PATH ─────────────────────────────────────────────
echo ""
echo "[Step 1] Checking environment..."
export LD_LIBRARY_PATH=/usr/local/cuda/lib64

if ! nvidia-smi > /dev/null 2>&1; then
    echo "ERROR: nvidia-smi failed. Ensure the driver is loaded after reboot."
    echo "       Try: unset LD_LIBRARY_PATH && nvidia-smi"
    exit 1
fi
echo "[Step 1] nvidia-smi OK."

# ─── STEP 2: Verify patch file exists ─────────────────────────────────────────
echo ""
echo "[Step 2] Checking required files..."
if [ ! -f "$PATCH_FILE" ]; then
    echo "ERROR: Missing patch file: $PATCH_FILE"
    exit 1
fi
echo "[Step 2] Patch file found. OK."

# ─── STEP 3: Verify DKMS source and dkms.conf exist (restore if missing) ─────
echo ""
echo "[Step 3] Checking DKMS source at $UVM_DIR..."
if [ ! -d "$UVM_DIR" ] || [ ! -f "$UVM_SRC/dkms.conf" ]; then
    echo "[Step 3] Source or dkms.conf missing — reinstalling nvidia-kernel-source-470 and nvidia-dkms-470..."
    sudo apt install --reinstall -y nvidia-kernel-source-470 nvidia-dkms-470
    if [ ! -d "$UVM_DIR" ] || [ ! -f "$UVM_SRC/dkms.conf" ]; then
        echo "ERROR: Source still missing after reinstall. Aborting."
        exit 1
    fi
    echo "[Step 3] Source and dkms.conf restored. OK."
else
    echo "[Step 3] nvidia-uvm source found. OK."
fi

# ─── STEP 4: Copy new source files and apply patch ────────────────────────────
echo ""
echo "[Step 4] Installing new source files and applying patch..."

# Verify new source files are present in artifacts
for f in uvm_sampling_tracker.h uvm_sampling_tracker.c uvm_sampling_procfs.c; do
    if [ ! -f "$SCRIPT_DIR/$f" ]; then
        echo "ERROR: Missing required file: $SCRIPT_DIR/$f"
        exit 1
    fi
done

# Always copy the canonical new files (prevents doubling from re-runs)
sudo cp "$SCRIPT_DIR/uvm_sampling_tracker.h" "$UVM_DIR/"
sudo cp "$SCRIPT_DIR/uvm_sampling_tracker.c" "$UVM_DIR/"
sudo cp "$SCRIPT_DIR/uvm_sampling_procfs.c"  "$UVM_DIR/"
echo "[Step 4] New source files copied."

# Apply patch to existing files (idempotent: patch will skip already-patched hunks)
if grep -q "uvm_sampling_tracker_init" "$UVM_DIR/uvm.c" 2>/dev/null; then
    echo "[Step 4] Existing-file patch already applied, skipping."
else
    sudo patch -p1 -d "$UVM_SRC" < "$PATCH_FILE"
    echo "[Step 4] Patch applied successfully."
fi

# ─── STEP 5: Rebuild and install nvidia-uvm via DKMS ─────────────────────────
echo ""
echo "[Step 5] Rebuilding nvidia-uvm via DKMS (this may take a few minutes)..."
KERN=$(uname -r)
# Remove built/installed modules for this kernel — keeps source registered
sudo dkms remove -m nvidia -v 470.256.02 -k "$KERN" 2>/dev/null || true
sudo dkms build  -m nvidia -v 470.256.02 -k "$KERN"
sudo dkms install -m nvidia -v 470.256.02 -k "$KERN"
echo "[Step 5] DKMS build and install complete."

# ─── STEP 6: Reload the nvidia-uvm kernel module ──────────────────────────────
echo ""
echo "[Step 6] Reloading nvidia-uvm kernel module..."
sudo rmmod nvidia_uvm 2>/dev/null || true
sudo modprobe nvidia_uvm
echo "[Step 6] nvidia-uvm loaded."

# ─── STEP 7: Verify ───────────────────────────────────────────────────────────
echo ""
echo "[Step 7] Final verification..."
nvidia-smi
lsmod | grep nvidia

echo ""
echo "============================================="
echo " Done. Patched nvidia-uvm is now loaded."
echo " Tracker procfs: /proc/uvm_sampling_tracker"
echo " (appears after a GPU process opens /dev/nvidia-uvm)"
echo "============================================="
