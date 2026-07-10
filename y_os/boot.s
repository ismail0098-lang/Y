# boot.s - Multiboot 1 Bootloader with GDT, IDT and Keyboard Interrupts for YSU-OS

.set ALIGN,    1<<0             
.set MEMINFO,  1<<1             
.set FLAGS,    ALIGN | MEMINFO  
.set MAGIC,    0x1BADB002       
.set CHECKSUM, -(MAGIC + FLAGS) 

.section .multiboot
.align 4
.long MAGIC
.long FLAGS
.long CHECKSUM

.section .bss
.align 16
stack_bottom:
.skip 16384 # 16 KiB stack
stack_top:

# IDT structure: 256 entries of 8 bytes each
.align 16
idt_start:
.skip 2048
idt_end:

.section .data
# Save area for Multiboot parameters
.global multiboot_magic
.global multiboot_info
multiboot_magic:
    .long 0
multiboot_info:
    .long 0

# GDT structure
.align 16
gdt_start:
    # null descriptor
    .long 0x00000000
    .long 0x00000000
gdt_code:
    # base 0, limit 0xffffffff, type 0x9a (exec/read), granularity 0xcf
    .word 0xffff
    .word 0x0000
    .byte 0x00
    .byte 0x9a
    .byte 0xcf
    .byte 0x00
gdt_data:
    # base 0, limit 0xffffffff, type 0x92 (read/write), granularity 0xcf
    .word 0xffff
    .word 0x0000
    .byte 0x00
    .byte 0x92
    .byte 0xcf
    .byte 0x00
gdt_end:

gdt_pointer:
    .word gdt_end - gdt_start - 1
    .long gdt_start

# IDT pointer containing reference to idt_start (fixup)
idt_pointer:
.word 2047      # limit (256 entries * 8 bytes - 1)
.long idt_start # base

.section .text
.global _start
.type _start, @function
_start:
    # Disable interrupts during setup
    cli

    # Save Multiboot parameters immediately before they are overwritten
    mov %eax, multiboot_magic
    mov %ebx, multiboot_info

    # Load GDT
    lgdt gdt_pointer
    # Perform far jump to set CS register to 0x08
    ljmp $0x08, $.reload_segments

.reload_segments:
    # Update data segment registers
    mov $0x10, %ax
    mov %ax, %ds
    mov %ax, %es
    mov %ax, %fs
    mov %ax, %gs
    mov %ax, %ss

    # Initialize Stack Pointer
    mov $stack_top, %esp

    # Enable SSE (Clear TS, set MP, NE in CR0; set OSFXSR, OSXMMEXCPT in CR4)
    mov %cr0, %eax
    and $0xfffffffb, %eax
    or $0x22, %eax
    mov %eax, %cr0

    mov %cr4, %eax
    or $0x600, %eax
    mov %eax, %cr4

    # Enable AVX and AVX-512 (Set OSXSAVE in CR4, enable x87+SSE+YMM+ZMM in XCR0)
    mov %cr4, %eax
    or $0x40000, %eax
    mov %eax, %cr4

    xor %ecx, %ecx
    xgetbv
    or $0xe7, %eax
    xsetbv

    # 1. Initialize PIC (Remap IRQs 0-15 to interrupts 32-47)
    # ICW1: initialization
    mov $0x11, %al
    out %al, $0x20
    out %al, $0xA0
    # ICW2: Vector offsets
    mov $0x20, %al # Master PIC offset to 32 (0x20)
    out %al, $0x21
    mov $0x28, %al # Slave PIC offset to 40 (0x28)
    out %al, $0xA1
    # ICW3: Cascade setup
    mov $0x04, %al
    out %al, $0x21
    mov $0x02, %al
    out %al, $0xA1
    # ICW4: 8086 mode
    mov $0x01, %al
    out %al, $0x21
    out %al, $0xA1

    # 2. Mask all interrupts except Keyboard (IRQ 1)
    mov $0xFD, %al
    out %al, $0x21
    mov $0xFF, %al
    out %al, $0xA1

    # 3. Setup IDT entry for Keyboard Interrupt (vector 33, which is Master IRQ 1)
    mov $keyboard_isr, %eax
    mov $idt_start, %ebx
    # Entry 33 starts at offset 33 * 8 = 264
    add $264, %ebx
    
    # Store low offset
    mov %ax, (%ebx)
    # Store selector (0x08)
    movw $0x08, 2(%ebx)
    # Store zero & type flag (0x8E00)
    movw $0x8E00, 4(%ebx)
    # Store high offset
    shr $16, %eax
    mov %ax, 6(%ebx)

    # Load IDT
    lidt idt_pointer

    # Push Multiboot parameters and kernel end address onto the stack for kernel_main
    push $_kernel_end
    push multiboot_info
    push multiboot_magic

    # Call the Y-lang kernel main function
    call kernel_main

    # Loop infinitely if kernel returns
1:  hlt
    jmp 1b

# Keyboard Interrupt Service Routine
.global keyboard_isr
.type keyboard_isr, @function
keyboard_isr:
    pusha           # Save all general-purpose registers

    # Read scancode from keyboard controller (port 0x60)
    xor %eax, %eax
    in $0x60, %al

    # Pass scancode as argument to the Y keyboard_handler
    push %eax
    call keyboard_handler
    add $4, %esp    # Clean up stack

    # Send End Of Interrupt (EOI) to Master PIC (port 0x20)
    mov $0x20, %al
    out %al, $0x20

    popa            # Restore registers
    iret            # Return from interrupt

# Identity string constructor for Y-compiler string literal support on bare-metal
.global ystr_new
.type ystr_new, @function
ystr_new:
    mov 4(%esp), %eax
    ret
