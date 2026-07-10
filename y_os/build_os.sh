#!/bin/bash
set -e

# y_os/build_os.sh - Build Script for YSU-OS Kernel
cd "$(dirname "$0")"

echo "[*] Compiling kernel.ysu to LLVM IR using Y compiler..."
../target/release/Y kernel.ysu --emit-llvm --output=kernel.ll

echo "[*] Compiling and linking assembly bootloader & kernel IR via clang..."
# We use -m32 target to produce a 32-bit Multiboot ELF kernel executable
clang -m32 -ffreestanding -O2 -nostdlib -no-pie -static -T linker.ld -o ysu_kernel.bin boot.s kernel.ll

echo "[*] Verifying Multiboot header..."
if grub-file --is-x86-multiboot ysu_kernel.bin; then
    echo "      -> SUCCESS: ysu_kernel.bin is a valid Multiboot header!"
else
    echo "      -> WARNING: grub-file failed to verify Multiboot header. If grub-file is not installed, this is expected."
fi

echo "[*] Build Complete: ysu_kernel.bin generated."
