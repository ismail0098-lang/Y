#!/bin/bash
# y_os/run_os.sh - Script to run YSU-OS kernel in QEMU VM

if ! command -v qemu-system-x86_64 &> /dev/null; then
    echo "[!] Error: qemu-system-x86_64 not found."
    echo "    Please install it first."
    echo "    On Arch/Gentoo: sudo pacman -S qemu-system-x86"
    echo "    On Debian/Ubuntu: sudo apt install qemu-system-x86"
    exit 1
fi

echo "[*] Starting YSU-OS kernel in QEMU VM..."
qemu-system-x86_64 -kernel ysu_kernel.bin
