; ==============================================================================
; YSU Engine - Sentinel Hardware Probe
; sentinel_probe.asm
; 
; Architecture: x86_64
; Syntax: NASM
; 
; This module provides zero-overhead hardware probing. It uses CPUID and RDTSC
; to gather cache line sizes, AVX-512 support, and memory latency.
; It is designed to be called directly from the Y compiler driver.
; ==============================================================================

bits 64
section .text

; Export the functions so C/Rust can call them
global probe_cpu_features
global measure_l1_latency

; ------------------------------------------------------------------------------
; fn probe_cpu_features(uint32_t* out_buffer)
; 
; Uses CPUID to detect AVX, AVX-512, and cache line sizes.
; RCX contains the pointer to the output buffer (Windows x64 calling convention).
; ------------------------------------------------------------------------------
probe_cpu_features:
    push rbx
    push rbp
    mov rbp, rcx            ; Save the out_buffer pointer in RBP

    ; 1. Standard CPUID (EAX=1) - Get Feature Flags
    mov eax, 1
    cpuid
    ; ECX contains features like AVX
    ; EDX contains features like SSE2
    mov dword [rbp + 0], ecx
    mov dword [rbp + 4], edx

    ; 2. Extended CPUID (EAX=7, ECX=0) - Get Extended Features (AVX-512)
    mov eax, 7
    xor ecx, ecx
    cpuid
    ; EBX contains AVX2, AVX-512F, etc.
    mov dword [rbp + 8], ebx

    ; 3. Cache Line Size (EAX=0x80000006)
    mov eax, 0x80000006
    cpuid
    ; ECX contains L2 cache info
    mov dword [rbp + 12], ecx

    pop rbp
    pop rbx
    ret

; ------------------------------------------------------------------------------
; fn measure_l1_latency() -> uint64_t
; 
; Returns the number of CPU cycles it takes to perform a barrier-enclosed
; nop or simple memory read, used to baseline the L1 cache latency.
; Returns the cycle count in RAX.
; ------------------------------------------------------------------------------
measure_l1_latency:
    push rbx

    ; Serialize pipeline before reading start time
    lfence                  
    rdtsc                   ; EDX:EAX = timestamp
    shl rdx, 32
    or rax, rdx
    mov rbx, rax            ; RBX = start_time

    ; --- PAYLOAD START ---
    ; This is the microbenchmark payload.
    ; (Currently just nops to measure baseline overhead)
    nop
    nop
    nop
    nop
    ; --- PAYLOAD END ---

    ; Serialize pipeline before reading end time
    lfence                  
    rdtscp                  ; EDX:EAX = timestamp (RDTSCP is serializing)
    shl rdx, 32
    or rax, rdx             ; RAX = end_time

    ; Calculate delta
    sub rax, rbx            ; RAX = end_time - start_time

    pop rbx
    ret

