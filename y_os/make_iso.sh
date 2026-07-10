#!/bin/bash
set -e

# make_iso.sh - Build a bootable ISO for YSU-OS

echo "[*] Creating scratch directories..."
mkdir -p iso_scratch/boot/grub

echo "[*] Copying kernel binary..."
cp ysu_kernel.bin iso_scratch/boot/

echo "[*] Generating grub.cfg..."
cat << 'EOF' > iso_scratch/boot/grub/grub.cfg
set timeout=0
set default=0

menuentry "YSU-OS" {
    multiboot /boot/ysu_kernel.bin
    boot
}
EOF

echo "[*] Creating bootable ISO using grub-mkrescue..."
if grub-mkrescue -o ysu_os.iso iso_scratch; then
    echo "      -> SUCCESS: ysu_os.iso successfully generated!"
    echo "      -> You can now run this ISO in VirtualBox, VMware, or burn it to a USB drive!"
else
    echo "      -> ERROR: grub-mkrescue failed. Ensure xorriso or libisoburn is installed."
    exit 1
fi

echo "[*] Cleaning up scratch directory..."
rm -rf iso_scratch
