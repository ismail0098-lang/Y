/* linker.ld - Kernel Memory Layout for YSU-OS */
ENTRY(_start)

SECTIONS
{
    /* Load the kernel at 1MB (standard location for Multiboot kernels) */
    . = 1M;

    /* Text section: Multiboot header must be at the very beginning, followed by code */
    .text BLOCK(4K) : ALIGN(4K)
    {
        *(.multiboot)
        *(.text)
    }

    /* Read-only data */
    .rodata BLOCK(4K) : ALIGN(4K)
    {
        *(.rodata)
    }

    /* Initialized data */
    .data BLOCK(4K) : ALIGN(4K)
    {
        *(.data)
    }

    /* Uninitialized data and Stack */
    .bss BLOCK(4K) : ALIGN(4K)
    {
        *(COMMON)
        *(.bss)
    }

    _kernel_end = .;
}
