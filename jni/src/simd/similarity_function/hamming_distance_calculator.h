#pragma once

#include <cstdint>
#include "platform_defs.h"
#include "faiss/impl/DistanceComputer.h"

struct HammingDistanceCalculatorInterface : faiss::DistanceComputer {
    void set_query(const float* x) final {
        // No-op
    }

    float operator()(faiss::idx_t i) final {
        // No-op
        return 0;
    }


    float symmetric_dis(faiss::idx_t i, faiss::idx_t j) final {
        // No-op
        return 0;
    }

    virtual float calculate(const uint8_t* vec) = 0;

    virtual void calculateBatch4(float* RESTRICT score0,
                                 float* RESTRICT score1,
                                 float* RESTRICT score2,
                                 float* RESTRICT score3,
                                 const uint8_t* RESTRICT vec0,
                                 const uint8_t* RESTRICT vec1, const uint8_t* RESTRICT vec2,
                                 const uint8_t* RESTRICT vec3) = 0;

    virtual void calculateBatch8(float* RESTRICT score0,
                                float* RESTRICT score1,
                                 float* RESTRICT score2,
                                 float* RESTRICT score3,
                                 float* RESTRICT score4,
                                 float* RESTRICT score5,
                                 float* RESTRICT score6,
                                 float* RESTRICT score7,
                                 const uint8_t* RESTRICT vec0,
                                 const uint8_t* RESTRICT vec1,
                                 const uint8_t* RESTRICT vec2,
                                 const uint8_t* RESTRICT vec3,
                                 const uint8_t* RESTRICT vec4,
                                 const uint8_t* RESTRICT vec5,
                                 const uint8_t* RESTRICT vec6,
                                 const uint8_t* RESTRICT vec7) = 0;
};

template <class HammingComputer>
struct HammingDistanceCalculator final : HammingDistanceCalculatorInterface {
    const int32_t codeSize;
    HammingComputer hc;

    HammingDistanceCalculator(const int32_t _codeSize, const uint8_t* query)
      : HammingDistanceCalculatorInterface(),
        codeSize(_codeSize),
        hc() {
        hc.set(query, codeSize);
    }

    float calculate(const uint8_t* vec) final {
        return hc.hamming(vec);
    }

    void calculateBatch4(float* RESTRICT score0,
                         float* RESTRICT score1,
                         float* RESTRICT score2,
                         float* RESTRICT score3,
                         const uint8_t* RESTRICT vec0,
                         const uint8_t* RESTRICT vec1,
                         const uint8_t* RESTRICT vec2,
                         const uint8_t* RESTRICT vec3) final {
        *score0 = hc.hamming(vec0);
        *score1 = hc.hamming(vec1);
        *score2 = hc.hamming(vec2);
        *score3 = hc.hamming(vec3);
    }

    void calculateBatch8(float* RESTRICT score0,
                         float* RESTRICT score1,
                         float* RESTRICT score2,
                         float* RESTRICT score3,
                         float* RESTRICT score4,
                         float* RESTRICT score5,
                         float* RESTRICT score6,
                         float* RESTRICT score7,
                         const uint8_t* RESTRICT vec0,
                         const uint8_t* RESTRICT vec1,
                         const uint8_t* RESTRICT vec2,
                         const uint8_t* RESTRICT vec3,
                         const uint8_t* RESTRICT vec4,
                         const uint8_t* RESTRICT vec5,
                         const uint8_t* RESTRICT vec6,
                         const uint8_t* RESTRICT vec7) final {
        *score0 = hc.hamming(vec0);
        *score1 = hc.hamming(vec1);
        *score2 = hc.hamming(vec2);
        *score3 = hc.hamming(vec3);
        *score4 = hc.hamming(vec4);
        *score5 = hc.hamming(vec5);
        *score6 = hc.hamming(vec6);
        *score7 = hc.hamming(vec7);
    }
};
