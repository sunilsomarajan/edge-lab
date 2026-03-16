# BM25 + Integer Merge Benchmark

Scalar vs batched vs RVV implementations of BM25 scoring and integer merge — the code that produced the cycle counts in [Zero-Cost Abstractions on RISC-V](https://ssunil.dev/posts/when-abstractions-hide-the-hardware/).

## Variants

- **Scalar** — one element at a time (simulates `Iterator::next()`)
- **Batched** — 512-element inner loop (compiler can see the full loop)
- **RVV intrinsics** — explicit vector operations via `<riscv_vector.h>`

## Context

Inspired by [turbopuffer's BM25 post](https://turbopuffer.com/blog/turbopuffer-bm25) on why Rust's iterator abstraction costs real cycles. We ran the same workloads on RISC-V with the Vector extension to measure the gap.

## Building

This is reference code, stripped of target-specific infrastructure for readability. To compile-check:

```bash
riscv64-elf-gcc -O2 -march=rv64gcv -mabi=lp64d -c bench_bm25.c
```

Running on real hardware requires porting `main()` to your target's startup and I/O (RTOS task, UART, linker script, etc.).
