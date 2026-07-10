#!/bin/bash
set -e

echo "--- Cleaning Workspace ---"
rm -f *.o ysu_kernel.bin

echo "--- Assembling & Compiling ---"
if ! command -v nasm &> /dev/null; then
    echo "[ERROR] 'nasm' is not installed. Please install it using your package manager (e.g., sudo pacman -S nasm)"
    exit 1
fi

nasm -f elf32 c_src/boot.asm -o boot.o
gcc -c c_src/gdt.c -o gdt.o -ffreestanding -fno-stack-protector -m32 -Wno-pointer-to-int-cast
gcc -c c_src/kernel.c -o kernel.o -ffreestanding -fno-stack-protector -m32
gcc -c c_src/ysu_shm_portal.c -o ysu_shm_portal.o -ffreestanding -fno-stack-protector -m32

echo "--- Linking YSU Kernel ---"
ld -m elf_i386 -T linker.ld -o ysu_kernel.bin boot.o gdt.o kernel.o ysu_shm_portal.o

if [ -f "ysu_kernel.bin" ]; then
    echo -e "\n[SUCCESS] YSU_KERNEL.BIN IS READY."
    
    if command -v qemu-system-i386 &> /dev/null; then
        echo "--- Launching YSU Engine ---"
        qemu-system-i386 -kernel ysu_kernel.bin
    else
        echo "[WARNING] 'qemu-system-i386' is not installed. You can install it (e.g., sudo pacman -S qemu-desktop) to run the kernel in an emulator."
    fi
else
    echo -e "\n[ERROR] Build failed."
    exit 1
fi
