/**
 * @file riscv_cycles.h
 * @brief Lightweight cycle-accurate profiling for RISC-V targets.
 *
 * Reads the mcycle CSR (0xB00) directly via inline assembly.
 * No dependencies on vendor intrinsic headers.
 */

#ifndef RISCV_CYCLES_H
#define RISCV_CYCLES_H

/**
 * @brief Read the CPU cycle counter (mcycle CSR 0xB00).
 *
 * Returns the current count of CPU cycles since reset.
 *
 * @return Current cycle count (64-bit).
 */
__attribute__((always_inline)) static inline unsigned long long
rdcycles(void)
{
    unsigned long long val;
    __asm__ volatile("csrr %0, 0xB00" : "=r"(val));
    return val;
}

/**
 * @brief Start a profiling region.
 * @param var Variable name to store the start timestamp.
 */
#define PROFILE_START(var) unsigned long long var = rdcycles()

/**
 * @brief End a profiling region.
 * @param var Variable name from the matching PROFILE_START.
 * @return Elapsed cycles.
 */
#define PROFILE_END(var) (rdcycles() - (var))

#endif /* RISCV_CYCLES_H */
