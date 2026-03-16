/*
 * bench_bm25.c — BM25 + integer merge: scalar vs batched vs RVV
 *
 * This is the benchmark code that produced the cycle counts in:
 *   https://ssunil.dev/posts/when-abstractions-hide-the-hardware/
 *
 * Standalone version for reference.  The scoring kernels and
 * integer merge loops are identical to what ran on hardware.
 *
 * Original target: rv64gcv (zve32f) on custom silicon.
 * Profiling via mcycle CSR — see profiling/riscv_cycles.h.
 *
 * Reference:
 *   https://turbopuffer.com/blog/turbopuffer-bm25
 *
 * Build (compile-check only):
 *   riscv64-elf-gcc -O2 -march=rv64gcv -mabi=lp64d -c bench_bm25.c
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __riscv_vector
#include <riscv_vector.h>
#endif

#include "../../profiling/riscv_cycles.h"

/*-----------------------------------------------------------*/
/* Configuration                                             */
/*-----------------------------------------------------------*/

#define BM25_N_DOCS (20000U)
#define BM25_K1 (1.2f)
#define BM25_B (0.75f)
#define BM25_BATCH (512U)
#define BM25_RUNS (100U)

/*-----------------------------------------------------------*/
/* Results                                                   */
/*-----------------------------------------------------------*/

typedef struct
{
    uint64_t ullBestCycles;
    uint32_t ulRuns;
    uint32_t ulMismatches; /* vs scalar reference */
} Bm25Result_t;

static volatile Bm25Result_t xScalarResult;
static volatile Bm25Result_t xBatchedResult;
#ifdef __riscv_vector
static volatile Bm25Result_t xRvvResult;
#endif

/*-----------------------------------------------------------*/
/* Data arrays                                               */
/*-----------------------------------------------------------*/

static float fTfData[BM25_N_DOCS];
static float fDlData[BM25_N_DOCS];
static float fOutScalar[BM25_N_DOCS];
static float fOutBatched[BM25_N_DOCS];

#ifdef __riscv_vector
static float fOutRvv[BM25_N_DOCS];
#endif

/*-----------------------------------------------------------*/
/* BM25 implementations                                      */
/*-----------------------------------------------------------*/

static void
prvBm25Scalar(const float *tf, const float *dl, float *out, size_t n, float idf,
              float avgdl)
{
    const float k1 = BM25_K1;
    const float b = BM25_B;

    for (size_t i = 0; i < n; i++)
    {
        float t = tf[i];
        float d = dl[i];
        float norm = k1 * (1.0f - b + b * (d / avgdl));
        out[i] = idf * (t * (k1 + 1.0f)) / (t + norm);
    }
}

static void
prvBm25Batched(const float *tf, const float *dl, float *out, size_t n,
               float idf, float avgdl)
{
    const float k1 = BM25_K1;
    const float k1p1 = k1 + 1.0f;
    const float omb = 1.0f - BM25_B;
    const float boa = BM25_B / avgdl;

    size_t pos = 0;

    while (pos + BM25_BATCH <= n)
    {
        for (size_t i = 0; i < BM25_BATCH; i++)
        {
            float t = tf[pos + i];
            float d = dl[pos + i];
            float norm = k1 * (omb + boa * d);
            out[pos + i] = idf * (t * k1p1) / (t + norm);
        }

        pos += BM25_BATCH;
    }

    for (size_t i = pos; i < n; i++)
    {
        float t = tf[i];
        float d = dl[i];
        float norm = k1 * (omb + boa * d);
        out[i] = idf * (t * k1p1) / (t + norm);
    }
}

#ifdef __riscv_vector
static void
prvBm25Rvv(const float *tf, const float *dl, float *out, size_t n, float idf,
           float avgdl)
{
    const float k1 = BM25_K1;
    const float k1p1 = k1 + 1.0f;
    const float omb = 1.0f - BM25_B;
    const float boa = BM25_B / avgdl;
    const float idf_k1p1 = idf * k1p1;

    size_t pos = 0;

    while (pos < n)
    {
        size_t vl = __riscv_vsetvl_e32m4(n - pos);

        vfloat32m4_t v_tf = __riscv_vle32_v_f32m4(&tf[pos], vl);
        vfloat32m4_t v_dl = __riscv_vle32_v_f32m4(&dl[pos], vl);

        /* norm = k1 * (1-b + b/avgdl * dl) */
        vfloat32m4_t v_norm = __riscv_vfmul_vf_f32m4(v_dl, boa, vl);
        v_norm = __riscv_vfadd_vf_f32m4(v_norm, omb, vl);
        v_norm = __riscv_vfmul_vf_f32m4(v_norm, k1, vl);

        /* numerator = idf * (k1+1) * tf */
        vfloat32m4_t v_num = __riscv_vfmul_vf_f32m4(v_tf, idf_k1p1, vl);

        /* denominator = tf + norm */
        vfloat32m4_t v_den = __riscv_vfadd_vv_f32m4(v_tf, v_norm, vl);

        /* score = num / den */
        vfloat32m4_t v_score = __riscv_vfdiv_vv_f32m4(v_num, v_den, vl);

        __riscv_vse32_v_f32m4(&out[pos], v_score, vl);
        pos += vl;
    }
}
#endif

/*-----------------------------------------------------------*/
/* Integer Merge Benchmark (matches turbopuffer's workload)  */
/*                                                           */
/* Inner loop: x = val + 1; if (x % 2 == 0) sum += x;       */
/* 100K u64 in turbopuffer, 20K u32 here (zve32f limit)      */
/*-----------------------------------------------------------*/

#define INT_N (20000U)

/* Reuse fTfData memory (reinterpreted as uint32_t) */
#define ulIntData ((uint32_t *)fTfData)

typedef struct
{
    uint64_t ullBestCycles;
    uint32_t ulRuns;
    uint64_t ullSum;
} IntResult_t;

static volatile IntResult_t xIntScalarResult;
static volatile IntResult_t xIntBatchedResult;
#ifdef __riscv_vector
static volatile IntResult_t xIntRvvResult;
#endif

static uint64_t
prvIntScalar(const uint32_t *data, size_t n)
{
    uint64_t sum = 0;

    for (size_t i = 0; i < n; i++)
    {
        uint32_t x = data[i] + 1;

        if (x % 2 == 0)
        {
            sum += x;
        }
    }

    return sum;
}

static uint64_t
prvIntBatched(const uint32_t *data, size_t n)
{
    uint64_t sum = 0;
    size_t pos = 0;

    while (pos + BM25_BATCH <= n)
    {
        for (size_t i = 0; i < BM25_BATCH; i++)
        {
            uint32_t x = data[pos + i] + 1;

            if (x % 2 == 0)
            {
                sum += x;
            }
        }

        pos += BM25_BATCH;
    }

    for (size_t i = pos; i < n; i++)
    {
        uint32_t x = data[i] + 1;

        if (x % 2 == 0)
        {
            sum += x;
        }
    }

    return sum;
}

#ifdef __riscv_vector
static uint64_t
prvIntRvv(const uint32_t *data, size_t n)
{
    uint64_t sum = 0;
    size_t pos = 0;

    while (pos < n)
    {
        size_t vl = __riscv_vsetvl_e32m4(n - pos);

        vuint32m4_t v = __riscv_vle32_v_u32m4(&data[pos], vl);

        /* x = v + 1 */
        v = __riscv_vadd_vx_u32m4(v, 1, vl);

        /* mask: x % 2 == 0 (bit 0 is 0) */
        vuint32m4_t v_bit = __riscv_vand_vx_u32m4(v, 1, vl);
        vbool8_t mask = __riscv_vmseq_vx_u32m4_b8(v_bit, 0, vl);

        /* Masked reduction: zero non-matching, reduce sum.
         * Safe with small values (0-999) — no uint32 overflow. */
        vuint32m4_t v_zero = __riscv_vmv_v_x_u32m4(0, vl);
        vuint32m4_t v_masked = __riscv_vmerge_vvm_u32m4(v_zero, v, mask, vl);
        vuint32m1_t v_sinit = __riscv_vmv_v_x_u32m1(0, 1);
        vuint32m1_t v_partial =
            __riscv_vredsum_vs_u32m4_u32m1(v_masked, v_sinit, vl);
        sum += (uint64_t)__riscv_vmv_x_s_u32m1_u32(v_partial);

        pos += vl;
    }

    return sum;
}
#endif

/*-----------------------------------------------------------*/
/* Verification helper                                       */
/*-----------------------------------------------------------*/

static uint32_t
prvCountMismatches(const float *a, const float *b, size_t n)
{
    uint32_t mismatches = 0;

    for (size_t i = 0; i < n; i++)
    {
        float diff = a[i] - b[i];

        if (diff > 0.001f || diff < -0.001f)
        {
            mismatches++;
        }
    }

    return mismatches;
}

/*-----------------------------------------------------------*/
/* Benchmark entry point                                     */
/*-----------------------------------------------------------*/

int
main(void)
{
    /* Init test data (deterministic PRNG) */
    uint32_t seed = 42;
    float total_dl = 0.0f;

    for (size_t i = 0; i < BM25_N_DOCS; i++)
    {
        /* tf: 0-10 */
        seed = seed * 1103515245 + 12345;
        fTfData[i] = (float)(seed & 0x7FFFFFFF) / (float)0x7FFFFFFF * 10.0f;

        /* dl: 50-5000 */
        seed = seed * 1103515245 + 12345;
        fDlData[i] =
            50.0f + (float)(seed & 0x7FFFFFFF) / (float)0x7FFFFFFF * 4950.0f;

        total_dl += fDlData[i];
    }

    float avgdl = total_dl / (float)BM25_N_DOCS;
    float idf = 4.6f; /* term in 1% of docs */

    uint64_t t0, t1, best;

    /* ---- BM25 Float Benchmark ---- */

    /* Scalar */
    best = UINT64_MAX;

    for (uint32_t r = 0; r < BM25_RUNS; r++)
    {
        t0 = rdcycles();
        prvBm25Scalar(fTfData, fDlData, fOutScalar, BM25_N_DOCS, idf, avgdl);
        t1 = rdcycles();

        if (t1 - t0 < best)
        {
            best = t1 - t0;
        }
    }

    xScalarResult.ullBestCycles = best;
    xScalarResult.ulRuns = BM25_RUNS;
    xScalarResult.ulMismatches = 0;

    /* Batched */
    best = UINT64_MAX;

    for (uint32_t r = 0; r < BM25_RUNS; r++)
    {
        t0 = rdcycles();
        prvBm25Batched(fTfData, fDlData, fOutBatched, BM25_N_DOCS, idf, avgdl);
        t1 = rdcycles();

        if (t1 - t0 < best)
        {
            best = t1 - t0;
        }
    }

    xBatchedResult.ullBestCycles = best;
    xBatchedResult.ulRuns = BM25_RUNS;
    xBatchedResult.ulMismatches =
        prvCountMismatches(fOutScalar, fOutBatched, BM25_N_DOCS);

#ifdef __riscv_vector
    /* RVV */
    best = UINT64_MAX;

    for (uint32_t r = 0; r < BM25_RUNS; r++)
    {
        t0 = rdcycles();
        prvBm25Rvv(fTfData, fDlData, fOutRvv, BM25_N_DOCS, idf, avgdl);
        t1 = rdcycles();

        if (t1 - t0 < best)
        {
            best = t1 - t0;
        }
    }

    xRvvResult.ullBestCycles = best;
    xRvvResult.ulRuns = BM25_RUNS;
    xRvvResult.ulMismatches =
        prvCountMismatches(fOutScalar, fOutRvv, BM25_N_DOCS);
#endif

    /* ---- Integer Merge Benchmark ---- */
    /* Init integer data (reuse fTfData memory as uint32_t) */
    {
        uint32_t iseed = 12345;

        for (size_t i = 0; i < INT_N; i++)
        {
            iseed = iseed * 1103515245 + 12345;
            ulIntData[i] = iseed % 1000; /* 0-999 range */
        }
    }

    /* Integer scalar */
    best = UINT64_MAX;
    {
        uint64_t isum = 0;

        for (uint32_t r = 0; r < BM25_RUNS; r++)
        {
            t0 = rdcycles();
            isum = prvIntScalar(ulIntData, INT_N);
            t1 = rdcycles();

            if (t1 - t0 < best)
            {
                best = t1 - t0;
            }
        }

        xIntScalarResult.ullBestCycles = best;
        xIntScalarResult.ulRuns = BM25_RUNS;
        xIntScalarResult.ullSum = isum;
    }

    /* Integer batched */
    best = UINT64_MAX;
    {
        uint64_t isum = 0;

        for (uint32_t r = 0; r < BM25_RUNS; r++)
        {
            t0 = rdcycles();
            isum = prvIntBatched(ulIntData, INT_N);
            t1 = rdcycles();

            if (t1 - t0 < best)
            {
                best = t1 - t0;
            }
        }

        xIntBatchedResult.ullBestCycles = best;
        xIntBatchedResult.ulRuns = BM25_RUNS;
        xIntBatchedResult.ullSum = isum;
    }

#ifdef __riscv_vector
    /* Integer RVV */
    best = UINT64_MAX;
    {
        uint64_t isum = 0;

        for (uint32_t r = 0; r < BM25_RUNS; r++)
        {
            t0 = rdcycles();
            isum = prvIntRvv(ulIntData, INT_N);
            t1 = rdcycles();

            if (t1 - t0 < best)
            {
                best = t1 - t0;
            }
        }

        xIntRvvResult.ullBestCycles = best;
        xIntRvvResult.ulRuns = BM25_RUNS;
        xIntRvvResult.ullSum = isum;
    }
#endif

    return 0;
}
