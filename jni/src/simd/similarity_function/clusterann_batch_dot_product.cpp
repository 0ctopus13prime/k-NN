/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

// ClusterANN batch quantized dot products.
//
//   1-bit / 2-bit documents : transposed bit planes x transposed 4-bit query (popcount kernel, below)
//   4-bit documents         : split-half nibbles x unpacked 4-bit query (multiply-accumulate kernel, further down)
//
// The bit-plane kernel is the same decode scheme as the 1-bit SQ kernels in
// {avx512,arm_neon,default}_simd_similarity_function.cpp, generalised to DOC_PLANES document bit planes:
//
//   For every document plane dp (stride `stripe` bytes):
//     w = popcount(Q0 & D[dp]) * 1 + popcount(Q1 & D[dp]) * 2 + popcount(Q2 & D[dp]) * 4 + popcount(Q3 & D[dp]) * 8
//         (per byte, max 8 * 15 = 120, fits in a uint8)
//     dot += sum(w) << dp
//
// DOC_PLANES = 1 reproduces VectorUtil.int4BitDotProduct, 2 reproduces int4DibitDotProduct and 4 reproduces
// QuantizedVectorReader.int4NibbleDotProduct. Before each chunk is processed, the next chunk of every plane is
// prefetched into L1 (see prefetchNextChunk), as in the 1-bit kernels.
//
// Pure functions, no state, thread-safe.

#include "simd/clusterann_batch_dot_product.h"

#include <cstring>

#include "platform_defs.h"

#if defined(KNN_HAVE_AVX512) || defined(KNN_HAVE_AVX512_SPR)
    #include <immintrin.h>
    #include <cpuid.h>
    #define CLUSTERANN_KERNEL_AVX512 1
#elif defined(KNN_HAVE_AVX2_F16C)
    #include <immintrin.h>
    #define CLUSTERANN_KERNEL_AVX2 1
#elif defined(KNN_HAVE_ARM_FP16)
    #include <arm_neon.h>
    #define CLUSTERANN_KERNEL_NEON 1
#else
    #define CLUSTERANN_KERNEL_SCALAR 1
#endif

namespace knn_jni::simd::clusterann {

namespace {

// ------------------------------------------------------------------------------------------------
// Scalar reference
// ------------------------------------------------------------------------------------------------

template <int DOC_PLANES>
FORCE_INLINE int64_t scalarDot(const uint8_t* q, const uint8_t* d, const int32_t bytesPerCode) {
    const int32_t stripe = bytesPerCode / DOC_PLANES;
    const int32_t words = stripe >> 3;
    const int32_t remainStart = words * 8;

    const uint8_t* q0 = q;
    const uint8_t* q1 = q + stripe;
    const uint8_t* q2 = q + 2 * stripe;
    const uint8_t* q3 = q + 3 * stripe;

    int64_t total = 0;
    for (int32_t dp = 0; dp < DOC_PLANES; ++dp) {
        const uint8_t* plane = d + dp * stripe;
        int64_t planeSum = 0;

        for (int32_t w = 0; w < words; ++w) {
            const int32_t off = w * 8;
            uint64_t dw, w0, w1, w2, w3;
            std::memcpy(&dw, plane + off, sizeof(uint64_t));
            std::memcpy(&w0, q0 + off, sizeof(uint64_t));
            std::memcpy(&w1, q1 + off, sizeof(uint64_t));
            std::memcpy(&w2, q2 + off, sizeof(uint64_t));
            std::memcpy(&w3, q3 + off, sizeof(uint64_t));
            planeSum += __builtin_popcountll(w0 & dw)
                      + __builtin_popcountll(w1 & dw) * 2
                      + __builtin_popcountll(w2 & dw) * 4
                      + __builtin_popcountll(w3 & dw) * 8;
        }
        for (int32_t r = remainStart; r < stripe; ++r) {
            const uint32_t db = plane[r];
            planeSum += __builtin_popcount((q0[r] & db) & 0xFFu)
                      + __builtin_popcount((q1[r] & db) & 0xFFu) * 2
                      + __builtin_popcount((q2[r] & db) & 0xFFu) * 4
                      + __builtin_popcount((q3[r] & db) & 0xFFu) * 8;
        }
        total += planeSum << dp;
    }
    return total;
}

// Before the chunk at byte offset `i` is processed, pull the chunk at `i + CHUNK` of every plane into L1:
// the 4 query planes plus DOC_PLANES planes of each of the BATCH documents. `__builtin_prefetch(p, 0, 3)` is
// a read prefetch with maximum temporal locality (prefetcht0 on x86, PLDL1KEEP on aarch64). No-op past the end
// of the stripe, so the last full chunk still prefetches the masked / scalar tail.
template <int DOC_PLANES, int BATCH, int CHUNK>
FORCE_INLINE void prefetchNextChunk(const uint8_t* q0, const uint8_t* q1, const uint8_t* q2, const uint8_t* q3,
                                    const uint8_t* const* docs, const int32_t stripe, const int32_t i) {
    const int32_t next = i + CHUNK;
    if (next >= stripe) {
        return;
    }
    __builtin_prefetch(q0 + next, 0, 3);
    __builtin_prefetch(q1 + next, 0, 3);
    __builtin_prefetch(q2 + next, 0, 3);
    __builtin_prefetch(q3 + next, 0, 3);
    #pragma unroll
    for (int32_t b = 0; b < BATCH; ++b) {
        #pragma unroll
        for (int32_t dp = 0; dp < DOC_PLANES; ++dp) {
            __builtin_prefetch(docs[b] + dp * stripe + next, 0, 3);
        }
    }
}

// Scalar byte tail shared by the AVX2 and NEON kernels (bytes [from, stripe) of every plane).
template <int DOC_PLANES, int BATCH>
FORCE_INLINE void scalarTail(const uint8_t* q, const uint8_t* const* docs, const int32_t stripe,
                             const int32_t from, int64_t* acc) {
    const uint8_t* q0 = q;
    const uint8_t* q1 = q + stripe;
    const uint8_t* q2 = q + 2 * stripe;
    const uint8_t* q3 = q + 3 * stripe;
    for (int32_t r = from; r < stripe; ++r) {
        const uint32_t b0 = q0[r], b1 = q1[r], b2 = q2[r], b3 = q3[r];
        for (int32_t b = 0; b < BATCH; ++b) {
            for (int32_t dp = 0; dp < DOC_PLANES; ++dp) {
                const uint32_t db = docs[b][dp * stripe + r];
                const int64_t w = __builtin_popcount((b0 & db) & 0xFFu)
                                + __builtin_popcount((b1 & db) & 0xFFu) * 2
                                + __builtin_popcount((b2 & db) & 0xFFu) * 4
                                + __builtin_popcount((b3 & db) & 0xFFu) * 8;
                acc[b] += w << dp;
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------
// AVX-512 (F + BW + VL)
// ------------------------------------------------------------------------------------------------
#if defined(CLUSTERANN_KERNEL_AVX512)

constexpr int32_t kMaxBatch = 8;

// Per-byte popcount via a 4-bit lookup table (same as avx512_popcnt_epi8 in the 1-bit kernel).
FORCE_INLINE __m512i popcnt_epi8(const __m512i v) {
    const __m512i lut = _mm512_setr_epi64(
        0x0302020102010100LL, 0x0403030203020201LL,
        0x0302020102010100LL, 0x0403030203020201LL,
        0x0302020102010100LL, 0x0403030203020201LL,
        0x0302020102010100LL, 0x0403030203020201LL);
    const __m512i lowMask = _mm512_set1_epi8(0x0F);
    const __m512i lo = _mm512_and_si512(v, lowMask);
    const __m512i hi = _mm512_and_si512(_mm512_srli_epi16(v, 4), lowMask);
    return _mm512_add_epi8(_mm512_shuffle_epi8(lut, lo), _mm512_shuffle_epi8(lut, hi));
}

// w = popcnt(q0&d) + 2*popcnt(q1&d) + 4*popcnt(q2&d) + 8*popcnt(q3&d), per byte (<= 120).
// The 16-bit shifts cannot bleed across bytes: popcounts are <= 8 (4 bits) and shift by <= 3.
FORCE_INLINE __m512i weightedPlane(const __m512i q0, const __m512i q1, const __m512i q2, const __m512i q3,
                                   const __m512i d) {
    const __m512i p0 = popcnt_epi8(_mm512_and_si512(q0, d));
    const __m512i p1 = popcnt_epi8(_mm512_and_si512(q1, d));
    const __m512i p2 = popcnt_epi8(_mm512_and_si512(q2, d));
    const __m512i p3 = popcnt_epi8(_mm512_and_si512(q3, d));
    __m512i w = _mm512_add_epi8(p0, _mm512_slli_epi16(p1, 1));
    w = _mm512_add_epi8(w, _mm512_slli_epi16(p2, 2));
    return _mm512_add_epi8(w, _mm512_slli_epi16(p3, 3));
}

// acc += sum over bytes of (w * (1 << dp)) as int32 lanes.
// maddubs: u8 * s8 -> adjacent pairs summed into i16 (max 2 * 120 * 8 = 1920). madd with ones -> i32.
FORCE_INLINE __m512i accumulatePlane(const __m512i acc, const __m512i w, const __m512i planeWeight,
                                     const __m512i ones16) {
    const __m512i w16 = _mm512_maddubs_epi16(w, planeWeight);
    return _mm512_add_epi32(acc, _mm512_madd_epi16(w16, ones16));
}

template <int DOC_PLANES, int BATCH>
FORCE_INLINE void batchDot(const uint8_t* q, const uint8_t* const* docs, const int32_t bytesPerCode, float* out) {
    const int32_t stripe = bytesPerCode / DOC_PLANES;
    const uint8_t* q0 = q;
    const uint8_t* q1 = q + stripe;
    const uint8_t* q2 = q + 2 * stripe;
    const uint8_t* q3 = q + 3 * stripe;

    const __m512i ones16 = _mm512_set1_epi16(1);
    __m512i planeWeight[DOC_PLANES];
    #pragma unroll
    for (int32_t dp = 0; dp < DOC_PLANES; ++dp) {
        planeWeight[dp] = _mm512_set1_epi8(static_cast<char>(1 << dp));
    }

    __m512i acc[BATCH];
    #pragma unroll
    for (int32_t b = 0; b < BATCH; ++b) {
        acc[b] = _mm512_setzero_si512();
    }

    const int32_t full = stripe & ~63;
    int32_t i = 0;
    for (; i < full; i += 64) {
        const __m512i vq0 = _mm512_loadu_si512(q0 + i);
        const __m512i vq1 = _mm512_loadu_si512(q1 + i);
        const __m512i vq2 = _mm512_loadu_si512(q2 + i);
        const __m512i vq3 = _mm512_loadu_si512(q3 + i);

        // Pull the next 64-byte chunk of every plane (query and each document) into L1 while this chunk is
        // being multiplied. Same pattern as the 1-bit kernel; also covers the masked tail chunk.
        prefetchNextChunk<DOC_PLANES, BATCH, 64>(q0, q1, q2, q3, docs, stripe, i);

        #pragma unroll
        for (int32_t b = 0; b < BATCH; ++b) {
            #pragma unroll
            for (int32_t dp = 0; dp < DOC_PLANES; ++dp) {
                const __m512i d = _mm512_loadu_si512(docs[b] + dp * stripe + i);
                acc[b] = accumulatePlane(acc[b], weightedPlane(vq0, vq1, vq2, vq3, d), planeWeight[dp], ones16);
            }
        }
    }

    // Masked tail: remaining [1, 63] bytes of every plane, zero-filled lanes contribute nothing.
    const int32_t rem = stripe - full;
    if (rem > 0) {
        const __mmask64 m = static_cast<__mmask64>((1ULL << rem) - 1ULL);
        const __m512i vq0 = _mm512_maskz_loadu_epi8(m, q0 + i);
        const __m512i vq1 = _mm512_maskz_loadu_epi8(m, q1 + i);
        const __m512i vq2 = _mm512_maskz_loadu_epi8(m, q2 + i);
        const __m512i vq3 = _mm512_maskz_loadu_epi8(m, q3 + i);
        #pragma unroll
        for (int32_t b = 0; b < BATCH; ++b) {
            #pragma unroll
            for (int32_t dp = 0; dp < DOC_PLANES; ++dp) {
                const __m512i d = _mm512_maskz_loadu_epi8(m, docs[b] + dp * stripe + i);
                acc[b] = accumulatePlane(acc[b], weightedPlane(vq0, vq1, vq2, vq3, d), planeWeight[dp], ones16);
            }
        }
    }

    #pragma unroll
    for (int32_t b = 0; b < BATCH; ++b) {
        out[b] = static_cast<float>(_mm512_reduce_add_epi32(acc[b]));
    }
}

const char* kKernelName = "avx512";

// ------------------------------------------------------------------------------------------------
// AVX2
// ------------------------------------------------------------------------------------------------
#elif defined(CLUSTERANN_KERNEL_AVX2)

constexpr int32_t kMaxBatch = 4;

FORCE_INLINE __m256i popcnt_epi8(const __m256i v) {
    const __m256i lut = _mm256_setr_epi64x(
        0x0302020102010100LL, 0x0403030203020201LL,
        0x0302020102010100LL, 0x0403030203020201LL);
    const __m256i lowMask = _mm256_set1_epi8(0x0F);
    const __m256i lo = _mm256_and_si256(v, lowMask);
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), lowMask);
    return _mm256_add_epi8(_mm256_shuffle_epi8(lut, lo), _mm256_shuffle_epi8(lut, hi));
}

FORCE_INLINE __m256i weightedPlane(const __m256i q0, const __m256i q1, const __m256i q2, const __m256i q3,
                                   const __m256i d) {
    const __m256i p0 = popcnt_epi8(_mm256_and_si256(q0, d));
    const __m256i p1 = popcnt_epi8(_mm256_and_si256(q1, d));
    const __m256i p2 = popcnt_epi8(_mm256_and_si256(q2, d));
    const __m256i p3 = popcnt_epi8(_mm256_and_si256(q3, d));
    __m256i w = _mm256_add_epi8(p0, _mm256_slli_epi16(p1, 1));
    w = _mm256_add_epi8(w, _mm256_slli_epi16(p2, 2));
    return _mm256_add_epi8(w, _mm256_slli_epi16(p3, 3));
}

FORCE_INLINE __m256i accumulatePlane(const __m256i acc, const __m256i w, const __m256i planeWeight,
                                     const __m256i ones16) {
    const __m256i w16 = _mm256_maddubs_epi16(w, planeWeight);
    return _mm256_add_epi32(acc, _mm256_madd_epi16(w16, ones16));
}

FORCE_INLINE int32_t reduce_add_epi32(const __m256i v) {
    const __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    const __m128i s2 = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(_mm_hadd_epi32(s2, s2));
}

template <int DOC_PLANES, int BATCH>
FORCE_INLINE void batchDot(const uint8_t* q, const uint8_t* const* docs, const int32_t bytesPerCode, float* out) {
    const int32_t stripe = bytesPerCode / DOC_PLANES;
    const uint8_t* q0 = q;
    const uint8_t* q1 = q + stripe;
    const uint8_t* q2 = q + 2 * stripe;
    const uint8_t* q3 = q + 3 * stripe;

    const __m256i ones16 = _mm256_set1_epi16(1);
    __m256i planeWeight[DOC_PLANES];
    #pragma unroll
    for (int32_t dp = 0; dp < DOC_PLANES; ++dp) {
        planeWeight[dp] = _mm256_set1_epi8(static_cast<char>(1 << dp));
    }

    __m256i acc[BATCH];
    #pragma unroll
    for (int32_t b = 0; b < BATCH; ++b) {
        acc[b] = _mm256_setzero_si256();
    }

    const int32_t full = stripe & ~31;
    int32_t i = 0;
    for (; i < full; i += 32) {
        const __m256i vq0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q0 + i));
        const __m256i vq1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q1 + i));
        const __m256i vq2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q2 + i));
        const __m256i vq3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q3 + i));

        // L1 prefetch of the next 32-byte chunk of every plane while this chunk is being multiplied.
        prefetchNextChunk<DOC_PLANES, BATCH, 32>(q0, q1, q2, q3, docs, stripe, i);

        #pragma unroll
        for (int32_t b = 0; b < BATCH; ++b) {
            #pragma unroll
            for (int32_t dp = 0; dp < DOC_PLANES; ++dp) {
                const __m256i d = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(docs[b] + dp * stripe + i));
                acc[b] = accumulatePlane(acc[b], weightedPlane(vq0, vq1, vq2, vq3, d), planeWeight[dp], ones16);
            }
        }
    }

    int64_t tail[BATCH];
    #pragma unroll
    for (int32_t b = 0; b < BATCH; ++b) {
        tail[b] = reduce_add_epi32(acc[b]);
    }
    if (i < stripe) {
        scalarTail<DOC_PLANES, BATCH>(q, docs, stripe, i, tail);
    }
    #pragma unroll
    for (int32_t b = 0; b < BATCH; ++b) {
        out[b] = static_cast<float>(tail[b]);
    }
}

const char* kKernelName = "avx2";

// ------------------------------------------------------------------------------------------------
// NEON
// ------------------------------------------------------------------------------------------------
#elif defined(CLUSTERANN_KERNEL_NEON)

constexpr int32_t kMaxBatch = 8;

FORCE_INLINE uint8x16_t weightedPlane(const uint8x16_t q0, const uint8x16_t q1, const uint8x16_t q2,
                                      const uint8x16_t q3, const uint8x16_t d) {
    const uint8x16_t p0 = vcntq_u8(vandq_u8(q0, d));
    const uint8x16_t p1 = vcntq_u8(vandq_u8(q1, d));
    const uint8x16_t p2 = vcntq_u8(vandq_u8(q2, d));
    const uint8x16_t p3 = vcntq_u8(vandq_u8(q3, d));
    uint8x16_t w = vaddq_u8(p0, vshlq_n_u8(p1, 1));
    w = vaddq_u8(w, vshlq_n_u8(p2, 2));
    return vaddq_u8(w, vshlq_n_u8(p3, 3));
}

// acc += (pairwise-widened w) << dp. u8 (<=120) -> u16 pairs (<=240) -> shifted (<=1920) -> u32 accumulate.
template <int DP>
FORCE_INLINE uint32x4_t accumulatePlane(const uint32x4_t acc, const uint8x16_t w) {
    uint16x8_t w16 = vpaddlq_u8(w);
    if constexpr (DP > 0) {
        w16 = vshlq_n_u16(w16, DP);
    }
    return vpadalq_u16(acc, w16);
}

template <int DOC_PLANES, int BATCH>
FORCE_INLINE void batchDot(const uint8_t* q, const uint8_t* const* docs, const int32_t bytesPerCode, float* out) {
    const int32_t stripe = bytesPerCode / DOC_PLANES;
    const uint8_t* q0 = q;
    const uint8_t* q1 = q + stripe;
    const uint8_t* q2 = q + 2 * stripe;
    const uint8_t* q3 = q + 3 * stripe;

    uint32x4_t acc[BATCH];
    #pragma unroll
    for (int32_t b = 0; b < BATCH; ++b) {
        acc[b] = vdupq_n_u32(0);
    }

    const int32_t full = stripe & ~15;
    int32_t i = 0;
    for (; i < full; i += 16) {
        const uint8x16_t vq0 = vld1q_u8(q0 + i);
        const uint8x16_t vq1 = vld1q_u8(q1 + i);
        const uint8x16_t vq2 = vld1q_u8(q2 + i);
        const uint8x16_t vq3 = vld1q_u8(q3 + i);

        // L1 prefetch (PLDL1KEEP) of the next 16-byte chunk of every plane while this chunk is being multiplied.
        prefetchNextChunk<DOC_PLANES, BATCH, 16>(q0, q1, q2, q3, docs, stripe, i);

        #pragma unroll
        for (int32_t b = 0; b < BATCH; ++b) {
            const uint8_t* doc = docs[b] + i;
            acc[b] = accumulatePlane<0>(acc[b], weightedPlane(vq0, vq1, vq2, vq3, vld1q_u8(doc)));
            if constexpr (DOC_PLANES > 1) {
                acc[b] = accumulatePlane<1>(acc[b], weightedPlane(vq0, vq1, vq2, vq3, vld1q_u8(doc + stripe)));
            }
            if constexpr (DOC_PLANES > 2) {
                acc[b] = accumulatePlane<2>(acc[b], weightedPlane(vq0, vq1, vq2, vq3, vld1q_u8(doc + 2 * stripe)));
                acc[b] = accumulatePlane<3>(acc[b], weightedPlane(vq0, vq1, vq2, vq3, vld1q_u8(doc + 3 * stripe)));
            }
        }
    }

    int64_t tail[BATCH];
    #pragma unroll
    for (int32_t b = 0; b < BATCH; ++b) {
        tail[b] = vaddvq_u32(acc[b]);
    }
    if (i < stripe) {
        scalarTail<DOC_PLANES, BATCH>(q, docs, stripe, i, tail);
    }
    #pragma unroll
    for (int32_t b = 0; b < BATCH; ++b) {
        out[b] = static_cast<float>(tail[b]);
    }
}

const char* kKernelName = "neon";

// ------------------------------------------------------------------------------------------------
// Scalar (auto-vectorised by -O3 where possible)
// ------------------------------------------------------------------------------------------------
#else

constexpr int32_t kMaxBatch = 4;

template <int DOC_PLANES, int BATCH>
FORCE_INLINE void batchDot(const uint8_t* q, const uint8_t* const* docs, const int32_t bytesPerCode, float* out) {
    for (int32_t b = 0; b < BATCH; ++b) {
        out[b] = static_cast<float>(scalarDot<DOC_PLANES>(q, docs[b], bytesPerCode));
    }
}

const char* kKernelName = "scalar";

#endif

// ================================================================================================
// 4-bit: split-half NIBBLE documents x unpacked 4-bit query (Lucene104 PACKED_NIBBLE layout)
//
//   doc byte i   = code[i] << 4 | code[i + packedLen]
//   query        = 2 * packedLen bytes: qHi = codes [0, packedLen) pairs with the high nibbles,
//                                       qLo = codes [packedLen, 2 * packedLen) pairs with the low nibbles
//   dot          = sum(hi_nibble(i) * qHi[i]) + sum(lo_nibble(i) * qLo[i])
//
// Both operands are in [0, 15], so a u8 x s8 multiply-accumulate (vpdpbusd / vpmaddubsw, udot) is exact and
// cannot overflow: max 4 * 225 = 900 per 32-bit lane per step. Ported from the bench kernel in
// kdy/4bit-bulksimd-results/results/code/native/kernel_{avx512,neon}.cpp.
// ================================================================================================

FORCE_INLINE int64_t scalarNibbleDot(const uint8_t* q, const uint8_t* d, const int32_t packedLen) {
    const uint8_t* qHi = q;
    const uint8_t* qLo = q + packedLen;
    int64_t sum = 0;
    for (int32_t i = 0; i < packedLen; ++i) {
        const uint32_t p = d[i];
        sum += static_cast<int64_t>(p >> 4) * qHi[i] + static_cast<int64_t>(p & 0x0Fu) * qLo[i];
    }
    return sum;
}

template <int BATCH>
FORCE_INLINE void scalarNibbleTail(const uint8_t* q, const uint8_t* const* docs, const int32_t packedLen,
                                   const int32_t from, int64_t* acc) {
    const uint8_t* qHi = q;
    const uint8_t* qLo = q + packedLen;
    for (int32_t i = from; i < packedLen; ++i) {
        for (int32_t b = 0; b < BATCH; ++b) {
            const uint32_t p = docs[b][i];
            acc[b] += static_cast<int64_t>(p >> 4) * qHi[i] + static_cast<int64_t>(p & 0x0Fu) * qLo[i];
        }
    }
}

// Prefetch the next CHUNK of the two query halves and of every document in the batch into L1.
template <int BATCH, int CHUNK>
FORCE_INLINE void prefetchNextNibbleChunk(const uint8_t* qHi, const uint8_t* qLo, const uint8_t* const* docs,
                                          const int32_t packedLen, const int32_t i) {
    const int32_t next = i + CHUNK;
    if (next >= packedLen) {
        return;
    }
    __builtin_prefetch(qHi + next, 0, 3);
    __builtin_prefetch(qLo + next, 0, 3);
    #pragma unroll
    for (int32_t b = 0; b < BATCH; ++b) {
        __builtin_prefetch(docs[b] + next, 0, 3);
    }
}

#if defined(CLUSTERANN_KERNEL_AVX512)

// CPUID leaf 7 sub-leaf 0, ECX bit 11 = AVX512_VNNI. The library itself is built for AVX512F/BW/VL, so the
// only question at runtime is whether vpdpbusd is available (Ice Lake / Sapphire Rapids yes, Skylake-X no).
bool detectAvx512Vnni() {
    unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        return false;
    }
    return ((ecx >> 11) & 1u) != 0;
}
const bool kHasVnni = detectAvx512Vnni();

// Two flavours of "acc += codes(u8) . query(s8)" over 64 bytes:
//   VNNI : one vpdpbusd (4-byte groups -> i32)
//   BW   : vpmaddubsw (2-byte groups -> i16, max 450) then vpmaddwd with ones (-> i32)
#define CLUSTERANN_ACCUM_VNNI(acc, codes, q) _mm512_dpbusd_epi32((acc), (codes), (q))
#define CLUSTERANN_ACCUM_BW(acc, codes, q) \
    _mm512_add_epi32((acc), _mm512_madd_epi16(_mm512_maddubs_epi16((codes), (q)), ones16))

// The batch body is a macro so the VNNI and non-VNNI versions cannot drift apart. `ATTR` carries the
// target attribute for the VNNI instantiation; intrinsics that need vpdpbusd may only appear inside it.
#define CLUSTERANN_NIBBLE_BATCH_AVX512(NAME, ATTR, ACCUM)                                                       \
    template <int BATCH>                                                                                       \
    ATTR void NAME(const uint8_t* q, const uint8_t* const* docs, const int32_t packedLen, float* out) {        \
        const uint8_t* qHi = q;                                                                                \
        const uint8_t* qLo = q + packedLen;                                                                    \
        const __m512i m0f = _mm512_set1_epi8(0x0F);                                                            \
        const __m512i ones16 = _mm512_set1_epi16(1);                                                           \
        (void) ones16;                                                                                         \
        __m512i acc[BATCH];                                                                                    \
        _Pragma("unroll")                                                                                      \
        for (int32_t b = 0; b < BATCH; ++b) {                                                                  \
            acc[b] = _mm512_setzero_si512();                                                                   \
        }                                                                                                      \
        const int32_t full = packedLen & ~63;                                                                  \
        int32_t i = 0;                                                                                         \
        for (; i < full; i += 64) {                                                                            \
            const __m512i vqh = _mm512_loadu_si512(qHi + i);                                                   \
            const __m512i vql = _mm512_loadu_si512(qLo + i);                                                   \
            prefetchNextNibbleChunk<BATCH, 64>(qHi, qLo, docs, packedLen, i);                                  \
            _Pragma("unroll")                                                                                  \
            for (int32_t b = 0; b < BATCH; ++b) {                                                              \
                const __m512i p = _mm512_loadu_si512(docs[b] + i);                                             \
                const __m512i lo = _mm512_and_si512(p, m0f);                                                   \
                const __m512i hi = _mm512_and_si512(_mm512_srli_epi16(p, 4), m0f);                             \
                acc[b] = ACCUM(acc[b], hi, vqh);                                                               \
                acc[b] = ACCUM(acc[b], lo, vql);                                                               \
            }                                                                                                  \
        }                                                                                                      \
        const int32_t rem = packedLen - full;                                                                  \
        if (rem > 0) {                                                                                         \
            const __mmask64 m = static_cast<__mmask64>((1ULL << rem) - 1ULL);                                 \
            const __m512i vqh = _mm512_maskz_loadu_epi8(m, qHi + i);                                           \
            const __m512i vql = _mm512_maskz_loadu_epi8(m, qLo + i);                                           \
            _Pragma("unroll")                                                                                  \
            for (int32_t b = 0; b < BATCH; ++b) {                                                              \
                const __m512i p = _mm512_maskz_loadu_epi8(m, docs[b] + i);                                     \
                const __m512i lo = _mm512_and_si512(p, m0f);                                                   \
                const __m512i hi = _mm512_and_si512(_mm512_srli_epi16(p, 4), m0f);                             \
                acc[b] = ACCUM(acc[b], hi, vqh);                                                               \
                acc[b] = ACCUM(acc[b], lo, vql);                                                               \
            }                                                                                                  \
        }                                                                                                      \
        _Pragma("unroll")                                                                                      \
        for (int32_t b = 0; b < BATCH; ++b) {                                                                  \
            out[b] = static_cast<float>(_mm512_reduce_add_epi32(acc[b]));                                      \
        }                                                                                                      \
    }

CLUSTERANN_NIBBLE_BATCH_AVX512(batchNibbleVnni, __attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni"))),
                               CLUSTERANN_ACCUM_VNNI)
CLUSTERANN_NIBBLE_BATCH_AVX512(batchNibbleBw, /* no extra target */, CLUSTERANN_ACCUM_BW)

#undef CLUSTERANN_NIBBLE_BATCH_AVX512
#undef CLUSTERANN_ACCUM_VNNI
#undef CLUSTERANN_ACCUM_BW

template <int BATCH>
FORCE_INLINE void batchNibble(const uint8_t* q, const uint8_t* const* docs, const int32_t packedLen, float* out) {
    if (kHasVnni) {
        batchNibbleVnni<BATCH>(q, docs, packedLen, out);
    } else {
        batchNibbleBw<BATCH>(q, docs, packedLen, out);
    }
}

const char* kNibbleKernelName() { return kHasVnni ? "avx512vnni" : "avx512bw"; }

#elif defined(CLUSTERANN_KERNEL_AVX2)

template <int BATCH>
FORCE_INLINE void batchNibble(const uint8_t* q, const uint8_t* const* docs, const int32_t packedLen, float* out) {
    const uint8_t* qHi = q;
    const uint8_t* qLo = q + packedLen;
    const __m256i m0f = _mm256_set1_epi8(0x0F);
    const __m256i ones16 = _mm256_set1_epi16(1);

    __m256i acc[BATCH];
    #pragma unroll
    for (int32_t b = 0; b < BATCH; ++b) {
        acc[b] = _mm256_setzero_si256();
    }

    const int32_t full = packedLen & ~31;
    int32_t i = 0;
    for (; i < full; i += 32) {
        const __m256i vqh = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qHi + i));
        const __m256i vql = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qLo + i));
        prefetchNextNibbleChunk<BATCH, 32>(qHi, qLo, docs, packedLen, i);
        #pragma unroll
        for (int32_t b = 0; b < BATCH; ++b) {
            const __m256i p = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(docs[b] + i));
            const __m256i lo = _mm256_and_si256(p, m0f);
            const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(p, 4), m0f);
            // u8 x s8 -> i16 pairs (max 450), then -> i32
            acc[b] = _mm256_add_epi32(acc[b], _mm256_madd_epi16(_mm256_maddubs_epi16(hi, vqh), ones16));
            acc[b] = _mm256_add_epi32(acc[b], _mm256_madd_epi16(_mm256_maddubs_epi16(lo, vql), ones16));
        }
    }

    int64_t tail[BATCH];
    #pragma unroll
    for (int32_t b = 0; b < BATCH; ++b) {
        tail[b] = reduce_add_epi32(acc[b]);
    }
    if (i < packedLen) {
        scalarNibbleTail<BATCH>(q, docs, packedLen, i, tail);
    }
    #pragma unroll
    for (int32_t b = 0; b < BATCH; ++b) {
        out[b] = static_cast<float>(tail[b]);
    }
}

const char* kNibbleKernelName() { return "avx2"; }

#elif defined(CLUSTERANN_KERNEL_NEON)

// udot (FEAT_DotProd) is mandatory from Armv8.4-A, which the library is built for (-march=armv8.4-a+fp16).
// Keep a vmull fallback in case the flags are ever lowered.
FORCE_INLINE uint32x4_t nibbleAccum(const uint32x4_t acc, const uint8x16_t codes, const uint8x16_t q) {
#if defined(__ARM_FEATURE_DOTPROD)
    return vdotq_u32(acc, codes, q);
#else
    const uint16x8_t lo = vmull_u8(vget_low_u8(codes), vget_low_u8(q));
    const uint16x8_t hi = vmull_u8(vget_high_u8(codes), vget_high_u8(q));
    return vpadalq_u16(vpadalq_u16(acc, lo), hi);
#endif
}

template <int BATCH>
FORCE_INLINE void batchNibble(const uint8_t* q, const uint8_t* const* docs, const int32_t packedLen, float* out) {
    const uint8_t* qHi = q;
    const uint8_t* qLo = q + packedLen;
    const uint8x16_t m0f = vdupq_n_u8(0x0F);

    uint32x4_t acc[BATCH];
    #pragma unroll
    for (int32_t b = 0; b < BATCH; ++b) {
        acc[b] = vdupq_n_u32(0);
    }

    const int32_t full = packedLen & ~15;
    int32_t i = 0;
    for (; i < full; i += 16) {
        const uint8x16_t vqh = vld1q_u8(qHi + i);
        const uint8x16_t vql = vld1q_u8(qLo + i);
        prefetchNextNibbleChunk<BATCH, 16>(qHi, qLo, docs, packedLen, i);
        #pragma unroll
        for (int32_t b = 0; b < BATCH; ++b) {
            const uint8x16_t p = vld1q_u8(docs[b] + i);
            acc[b] = nibbleAccum(acc[b], vshrq_n_u8(p, 4), vqh);
            acc[b] = nibbleAccum(acc[b], vandq_u8(p, m0f), vql);
        }
    }

    int64_t tail[BATCH];
    #pragma unroll
    for (int32_t b = 0; b < BATCH; ++b) {
        tail[b] = vaddvq_u32(acc[b]);
    }
    if (i < packedLen) {
        scalarNibbleTail<BATCH>(q, docs, packedLen, i, tail);
    }
    #pragma unroll
    for (int32_t b = 0; b < BATCH; ++b) {
        out[b] = static_cast<float>(tail[b]);
    }
}

#if defined(__ARM_FEATURE_DOTPROD)
const char* kNibbleKernelName() { return "neon-dotprod"; }
#else
const char* kNibbleKernelName() { return "neon"; }
#endif

#else

template <int BATCH>
FORCE_INLINE void batchNibble(const uint8_t* q, const uint8_t* const* docs, const int32_t packedLen, float* out) {
    for (int32_t b = 0; b < BATCH; ++b) {
        out[b] = static_cast<float>(scalarNibbleDot(q, docs[b], packedLen));
    }
}

const char* kNibbleKernelName() { return "scalar"; }

#endif

// 4-bit driver: 8-block (where the ISA has the registers), 4-block, then scalar tail.
void bulkNibbleDot(const uint8_t* query, const uint8_t* docs, const int32_t* offsets, const int32_t numOffsets,
                   float* results, const int32_t packedLen) {
    const uint8_t* ptrs[8];
    float tmp[8];
    int32_t k = 0;

    if constexpr (kMaxBatch >= 8) {
        for (; k + 8 <= numOffsets; k += 8) {
            for (int32_t b = 0; b < 8; ++b) {
                ptrs[b] = docs + static_cast<int64_t>(offsets[k + b]) * packedLen;
            }
            batchNibble<8>(query, ptrs, packedLen, tmp);
            for (int32_t b = 0; b < 8; ++b) {
                results[offsets[k + b]] = tmp[b];
            }
        }
    }

    for (; k + 4 <= numOffsets; k += 4) {
        for (int32_t b = 0; b < 4; ++b) {
            ptrs[b] = docs + static_cast<int64_t>(offsets[k + b]) * packedLen;
        }
        batchNibble<4>(query, ptrs, packedLen, tmp);
        for (int32_t b = 0; b < 4; ++b) {
            results[offsets[k + b]] = tmp[b];
        }
    }

    for (; k < numOffsets; ++k) {
        const uint8_t* doc = docs + static_cast<int64_t>(offsets[k]) * packedLen;
        results[offsets[k]] = static_cast<float>(scalarNibbleDot(query, doc, packedLen));
    }
}

// ------------------------------------------------------------------------------------------------
// 1-bit / 2-bit driver over transposed bit planes: 8-block (where the ISA has the registers), 4-block, tail.
// ------------------------------------------------------------------------------------------------
template <int DOC_PLANES>
void bulkDot(const uint8_t* query, const uint8_t* docs, const int32_t* offsets, const int32_t numOffsets,
             float* results, const int32_t bytesPerCode) {
    const uint8_t* ptrs[8];
    float tmp[8];
    int32_t k = 0;

    if constexpr (kMaxBatch >= 8) {
        for (; k + 8 <= numOffsets; k += 8) {
            for (int32_t b = 0; b < 8; ++b) {
                ptrs[b] = docs + static_cast<int64_t>(offsets[k + b]) * bytesPerCode;
            }
            batchDot<DOC_PLANES, 8>(query, ptrs, bytesPerCode, tmp);
            for (int32_t b = 0; b < 8; ++b) {
                results[offsets[k + b]] = tmp[b];
            }
        }
    }

    for (; k + 4 <= numOffsets; k += 4) {
        for (int32_t b = 0; b < 4; ++b) {
            ptrs[b] = docs + static_cast<int64_t>(offsets[k + b]) * bytesPerCode;
        }
        batchDot<DOC_PLANES, 4>(query, ptrs, bytesPerCode, tmp);
        for (int32_t b = 0; b < 4; ++b) {
            results[offsets[k + b]] = tmp[b];
        }
    }

    for (; k < numOffsets; ++k) {
        const uint8_t* doc = docs + static_cast<int64_t>(offsets[k]) * bytesPerCode;
        results[offsets[k]] = static_cast<float>(scalarDot<DOC_PLANES>(query, doc, bytesPerCode));
    }
}

}  // namespace

void bulkDotProduct1bit(const uint8_t* query, const uint8_t* docs, const int32_t* offsets, const int32_t numOffsets,
                        float* results, const int32_t bytesPerCode) {
    bulkDot<1>(query, docs, offsets, numOffsets, results, bytesPerCode);
}

void bulkDotProduct2bit(const uint8_t* query, const uint8_t* docs, const int32_t* offsets, const int32_t numOffsets,
                        float* results, const int32_t bytesPerCode) {
    bulkDot<2>(query, docs, offsets, numOffsets, results, bytesPerCode);
}

void bulkDotProduct4bit(const uint8_t* query, const uint8_t* docs, const int32_t* offsets, const int32_t numOffsets,
                        float* results, const int32_t bytesPerCode) {
    bulkNibbleDot(query, docs, offsets, numOffsets, results, bytesPerCode);
}

int64_t scalarDotProduct1bit(const uint8_t* query, const uint8_t* doc, const int32_t bytesPerCode) {
    return scalarDot<1>(query, doc, bytesPerCode);
}

int64_t scalarDotProduct2bit(const uint8_t* query, const uint8_t* doc, const int32_t bytesPerCode) {
    return scalarDot<2>(query, doc, bytesPerCode);
}

int64_t scalarDotProduct4bit(const uint8_t* query, const uint8_t* doc, const int32_t bytesPerCode) {
    return scalarNibbleDot(query, doc, bytesPerCode);
}

const char* kernelName() {
    return kKernelName;
}

const char* nibbleKernelName() {
    return kNibbleKernelName();
}

}  // namespace knn_jni::simd::clusterann
