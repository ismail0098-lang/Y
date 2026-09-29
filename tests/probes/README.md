# Phase 0 probe: cost of exact accumulation

Build and run (kernels MUST be a separate TU with no LTO -- see the header comment):

```sh
clang -O3 -march=native -c tests/probes/exact_kernels.c -o /tmp/ek.o
clang -O3 -march=native -c tests/probes/exact_probe.c   -o /tmp/ep.o
clang -o /tmp/exact_probe /tmp/ep.o /tmp/ek.o -lm && /tmp/exact_probe
```

Result on the reference box (Zen 5, one core): f32 FMA 166.9 G MAC/s (95% of
peak), exact Q16.16 65.2 G MAC/s -- **2.56x**. See docs/proof_carrying_kernels.md.
