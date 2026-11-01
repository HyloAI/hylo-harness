#!/usr/bin/env bash
# One-time system setup for the two AMD Radeon AI PRO R9700 (RDNA4, gfx1201) GPUs. Needs root and a REBOOT.
#
# Why: kernel 6.8 (Ubuntu 24.04 GA) has no amdgpu support for RDNA4, so the GPUs are not bound to a driver
# (`lspci -k` shows no "Kernel driver in use", /sys/class/kfd lists only CPU nodes). AMD's DKMS driver adds it.
#
# What it does:
#   1. installs kernel headers + AMD's amdgpu-install package (driver 31.60 / ROCm 7.2 line) from repo.radeon.com
#   2. builds the amdgpu DKMS driver (no ROCm userspace: PyTorch ROCm wheels bring their own runtime)
#   3. installs Mesa EGL/GBM so MuJoCo can render on the GPU (MUJOCO_GL=egl)
#   4. adds the user to the render and video groups (needed to open /dev/kfd and /dev/dri/renderD*)
#
# Before running: this machine also hosts other services (docker containers, Nextcloud); schedule the reboot.
# If Secure Boot is enabled, DKMS will ask you to enroll a MOK key at the next boot.
#
#   sudo scripts/gpu/setup-amd-gpu-root.sh bingao && sudo reboot
set -euo pipefail
USER_NAME="${1:?usage: setup-amd-gpu-root.sh <username>}"
VER=31.60
DEB="amdgpu-install_31.60.316000-1_all.deb"
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }

apt-get update
apt-get install -y "linux-headers-$(uname -r)" linux-modules-extra-"$(uname -r)" dkms wget
wget -q -O "/tmp/$DEB" "https://repo.radeon.com/amdgpu-install/$VER/ubuntu/noble/$DEB"
apt-get install -y "/tmp/$DEB"
apt-get update
amdgpu-install -y --usecase=dkms --no-32
apt-get install -y libegl1 libegl-mesa0 libgbm1 libgl1-mesa-dri
usermod -aG render,video "$USER_NAME"
dkms status | grep -i amdgpu || true
echo
echo "Done. Reboot, then run (no sudo): scripts/gpu/verify-amd-gpu.sh"
