; ysu_boot.asm - The Genesis for MSYS2/Windows/Linux
section .multiboot
align 4
    dd 0x1BADB002             ; Magic number for Multiboot
    dd 0x00                   ; Flags
    dd - (0x1BADB002 + 0x00)  ; Checksum

section .text
global _start
extern _ysu_main              ; C entry point function

_start:
    ; Jumping from Assembly to C
    call _ysu_main            
    
    ; If the kernel returns, halt the CPU
    cli                       
.halt:
    hlt
    jmp .halt
