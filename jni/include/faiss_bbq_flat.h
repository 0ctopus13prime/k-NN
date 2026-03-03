#ifndef KNNPLUGIN_JNI_FAISS_BBQ_FLAT_H
#define KNNPLUGIN_JNI_FAISS_BBQ_FLAT_H

#include "faiss/Index.h"
#include "faiss/MetricType.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <vector>

// Minimum MSE grid for 4-bit quantization (index 3, i.e. bits=4)
static constexpr float MINIMUM_MSE_GRID_4BIT[2] = {-2.514f, 2.514f};
static constexpr float FOUR_BIT_SCALE = 1.0f / 15.0f;
static constexpr float DEFAULT_LAMBDA = 0.1f;
static constexpr int DEFAULT_ITERS = 5;

static inline int discretize(int value, int bucket) {
    return ((value + (bucket - 1)) / bucket) * bucket;
}

static inline double clampd(double x, double a, double b) {
    return std::min(std::max(x, a), b);
}

static inline float clampf(float x, float a, float b) {
    return std::min(std::max(x, a), b);
}

/**
 * Compute quantization loss for interval optimization.
 */
static double quantizationLoss(const float* vector, int dim, float a, float b, int points, float norm2, float lambda) {
    double step = (b - a) / (points - 1.0);
    double stepInv = 1.0 / step;
    double xe = 0.0;
    double e = 0.0;
    for (int i = 0; i < dim; ++i) {
        double xi = vector[i];
        double xiq = a + step * std::round((clampd(xi, a, b) - a) * stepInv);
        xe += xi * (xi - xiq);
        e += (xi - xiq) * (xi - xiq);
    }
    return (1.0 - lambda) * xe * xe / norm2 + lambda * e;
}

/**
 * Optimize quantization intervals via coordinate descent.
 */
static void optimizeIntervals(float* interval, const float* vector, int dim, float norm2, int points, float lambda, int iters) {
    double initialLoss = quantizationLoss(vector, dim, interval[0], interval[1], points, norm2, lambda);
    float scale = (1.0f - lambda) / norm2;
    if (!std::isfinite(scale)) return;

    for (int iter = 0; iter < iters; ++iter) {
        float a = interval[0];
        float b = interval[1];
        float stepInv = (points - 1.0f) / (b - a);
        double daa = 0, dab = 0, dbb = 0, dax = 0, dbx = 0;
        for (int i = 0; i < dim; ++i) {
            float k = std::round((clampf(vector[i], a, b) - a) * stepInv);
            float s = k / (points - 1);
            daa += (1.0 - s) * (1.0 - s);
            dab += (1.0 - s) * s;
            dbb += s * s;
            dax += vector[i] * (1.0 - s);
            dbx += vector[i] * s;
        }
        double m0 = scale * dax * dax + lambda * daa;
        double m1 = scale * dax * dbx + lambda * dab;
        double m2 = scale * dbx * dbx + lambda * dbb;
        double det = m0 * m2 - m1 * m1;
        if (det == 0) return;
        float aOpt = (float)((m2 * dax - m1 * dbx) / det);
        float bOpt = (float)((m0 * dbx - m1 * dax) / det);
        if (std::abs(interval[0] - aOpt) < 1e-8 && std::abs(interval[1] - bOpt) < 1e-8) return;
        double newLoss = quantizationLoss(vector, dim, aOpt, bOpt, points, norm2, lambda);
        if (newLoss > initialLoss) return;
        interval[0] = aOpt;
        interval[1] = bOpt;
        initialLoss = newLoss;
    }
}

/**
 * Scalar quantize a float vector to 4-bit, centered on centroid.
 * Mutates `centered` in place (vector - centroid).
 * Writes 4-bit quantized values (0..15) into `dest`.
 * Returns: {lowerInterval, upperInterval, centroidDot, quantizedComponentSum}
 */
struct ScalarQuantizeResult {
    float lowerInterval;
    float upperInterval;
    float additionalCorrection; // centroidDot for inner product
    int quantizedComponentSum;
};

static ScalarQuantizeResult scalarQuantize4bit(
    const float* vector, int dim, const float* centroid,
    float* centered, uint8_t* dest)
{
    constexpr int bits = 4;
    constexpr int points = 1 << bits; // 16
    constexpr float nSteps = (float)(points - 1); // 15

    double vecMean = 0;
    double vecVar = 0;
    float norm2 = 0;
    float centroidDot = 0;
    float minVal = std::numeric_limits<float>::max();
    float maxVal = -std::numeric_limits<float>::max();

    for (int i = 0; i < dim; ++i) {
        centroidDot += vector[i] * centroid[i];
        centered[i] = vector[i] - centroid[i];
        minVal = std::min(minVal, centered[i]);
        maxVal = std::max(maxVal, centered[i]);
        norm2 += centered[i] * centered[i];
        double delta = centered[i] - vecMean;
        vecMean += delta / (i + 1);
        vecVar += delta * (centered[i] - vecMean);
    }
    vecVar /= dim;
    double vecStd = std::sqrt(vecVar);

    float interval[2];
    interval[0] = (float)clampd(MINIMUM_MSE_GRID_4BIT[0] * vecStd + vecMean, minVal, maxVal);
    interval[1] = (float)clampd(MINIMUM_MSE_GRID_4BIT[1] * vecStd + vecMean, minVal, maxVal);

    optimizeIntervals(interval, centered, dim, norm2, points, DEFAULT_LAMBDA, DEFAULT_ITERS);

    float a = interval[0];
    float b = interval[1];
    float step = (b - a) / nSteps;
    int sumQuery = 0;
    for (int i = 0; i < dim; ++i) {
        float xi = clampf(centered[i], a, b);
        int assignment = (int)std::round((xi - a) / step);
        sumQuery += assignment;
        dest[i] = (uint8_t)assignment;
    }

    return {interval[0], interval[1], centroidDot, sumQuery};
}

/**
 * Transpose half-byte (4-bit) quantized values into bit-plane layout.
 * Input: q[dim] with values 0..15
 * Output: quantQueryByte[4 * ceil(dim/8)] in bit-plane order
 */
static void transposeHalfByte(const uint8_t* q, int dim, uint8_t* quantQueryByte, int outLen) {
    int quarterLen = outLen / 4;
    int i = 0;
    while (i < dim) {
        int lowerByte = 0;
        int lowerMiddleByte = 0;
        int upperMiddleByte = 0;
        int upperByte = 0;
        for (int j = 7; j >= 0 && i < dim; --j) {
            lowerByte |= (q[i] & 1) << j;
            lowerMiddleByte |= ((q[i] >> 1) & 1) << j;
            upperMiddleByte |= ((q[i] >> 2) & 1) << j;
            upperByte |= ((q[i] >> 3) & 1) << j;
            i++;
        }
        int index = ((i + 7) / 8) - 1;
        quantQueryByte[index] = (uint8_t)lowerByte;
        quantQueryByte[index + quarterLen] = (uint8_t)lowerMiddleByte;
        quantQueryByte[index + 2 * quarterLen] = (uint8_t)upperMiddleByte;
        quantQueryByte[index + 3 * quarterLen] = (uint8_t)upperByte;
    }
}

/**
 * Compute dot product between 4-bit transposed query and 1-bit binary data vector.
 * q: transposed 4-bit query, length = 4 * binaryLen
 * d: 1-bit packed binary vector, length = binaryLen
 * Returns the weighted popcount sum.
 */
static int64_t int4BitDotProduct(const uint8_t* q, const uint8_t* d, int binaryLen) {
    int64_t ret = 0;
    for (int bitPlane = 0; bitPlane < 4; ++bitPlane) {
        int64_t subRet = 0;
        for (int r = 0; r < binaryLen; ++r) {
            subRet += __builtin_popcount(q[bitPlane * binaryLen + r] & d[r]);
        }
        ret += subRet << bitPlane;
    }
    return ret;
}

struct BBQDistanceComputer final : faiss::DistanceComputer {
    const int64_t oneElementByteSize;
    const uint64_t quantizedVectorBytes; // bytes for 1-bit packed vector per element
    const uint8_t* data;
    const float centroidDp;
    int32_t dimension;
    int32_t numVectors;
    const float* centroid; // pointer to centroid stored in FaissBBQFlat

    // fp32 query vector (centered on centroid)
    std::vector<float> centeredQuery;
    // Query-side correction terms
    float queryCentroidDot;  // query · centroid
    float queryCenteredSum;  // sum of (query - centroid)

    BBQDistanceComputer(int32_t _oneElementByteSize, const void* _data, float _centroidDp,
                        int32_t _dimension, int32_t _numVectors, const float* _centroid)
      : faiss::DistanceComputer(),
        oneElementByteSize(_oneElementByteSize),
        quantizedVectorBytes(_oneElementByteSize - (sizeof(float) * 3 + sizeof(int32_t))),
        data((const uint8_t*) _data),
        centroidDp(_centroidDp),
        dimension(_dimension),
        numVectors(_numVectors),
        centroid(_centroid),
        queryCentroidDot(0), queryCenteredSum(0)
    {
        centeredQuery.resize(_dimension, 0);
    }

    void set_query(const float* x) final {
        queryCentroidDot = 0;
        queryCenteredSum = 0;
        for (int i = 0; i < dimension; ++i) {
            queryCentroidDot += x[i] * centroid[i];
            centeredQuery[i] = x[i] - centroid[i];
            queryCenteredSum += centeredQuery[i];
        }
    }

    void setCorrectionFactors(const void* target, float& lowerInterval, float& intervalLength,
                              float& additionalCorrection, float& quantizedComponentSum) {
        const auto* correctionFactors = (const float*) ((const uint8_t*) target + quantizedVectorBytes);
        lowerInterval = correctionFactors[0];
        intervalLength = correctionFactors[1] - correctionFactors[0];
        additionalCorrection = correctionFactors[2];
        quantizedComponentSum = *((const int32_t*) (&correctionFactors[3]));
    }

    // Compute dot product between fp32 centered query and 1-bit packed vector
    // For each set bit at position h, accumulate centeredQuery[h]
    float fp32BitDotProduct(const uint8_t* binaryVec) const {
        float result = 0;
        int h = 0;
        for (int byteIdx = 0; byteIdx < (int)quantizedVectorBytes && h < dimension; ++byteIdx) {
            uint8_t byte = binaryVec[byteIdx];
            // Bits are packed MSB first (bit 7 = first dimension in the group)
            for (int bit = 7; bit >= 0 && h < dimension; --bit, ++h) {
                if ((byte >> bit) & 1) {
                    result += centeredQuery[h];
                }
            }
        }
        return result;
    }

    float scoreFromFp32(const void* target) {
        float ax, lx, additional, x1;
        setCorrectionFactors(target, ax, lx, additional, x1);

        float dp = fp32BitDotProduct((const uint8_t*)target);

        // score = centroidDot_query + centroidDot_stored - centroidDp
        //       + ax * queryCenteredSum + lx * dp
        return queryCentroidDot
               + additional       // stored centroidDot
               - centroidDp
               + ax * queryCenteredSum
               + lx * dp;
    }

    /// compute distance of vector i to current query using fp32 x 1-bit
    float operator()(faiss::idx_t i) final {
        const uint8_t* target = data + i * oneElementByteSize;
        return scoreFromFp32(target);
    }

    void distances_batch_4(
            const faiss::idx_t idx0,
            const faiss::idx_t idx1,
            const faiss::idx_t idx2,
            const faiss::idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) final {
        dis0 = scoreFromFp32(data + idx0 * oneElementByteSize);
        dis1 = scoreFromFp32(data + idx1 * oneElementByteSize);
        dis2 = scoreFromFp32(data + idx2 * oneElementByteSize);
        dis3 = scoreFromFp32(data + idx3 * oneElementByteSize);
    }

    /// compute distance between two stored vectors — stays 1-bit x 1-bit symmetric
    float symmetric_dis(faiss::idx_t i, faiss::idx_t j) {
        const uint64_t* target1 = reinterpret_cast<const uint64_t*>(data + i * oneElementByteSize);
        const uint64_t* target2 = reinterpret_cast<const uint64_t*>(data + j * oneElementByteSize);

        const uint64_t words = quantizedVectorBytes >> 3;
        uint32_t dp = 0;
        for (size_t w = 0; w < words; ++w) {
            dp += __builtin_popcountll(target1[w] & target2[w]);
        }

        float ax, lx, additional, x1;
        setCorrectionFactors(target1, ax, lx, additional, x1);
        float az, lz, additionalz, z1;
        setCorrectionFactors(target2, az, lz, additionalz, z1);

        return ax * az * dimension
               + az * lx * x1
               + ax * lz * z1
               + lx * lz * dp
               + additional
               + additionalz
               - centroidDp;
    }
};

struct FaissBBQFlat final : public faiss::Index {
    int64_t numVectors;
    int32_t quantizedVectorBytes;
    float centroidDp;
    int32_t oneElementSize;
    std::vector<uint8_t> quantizedVectorsAndCorrectionFactors;
    int32_t dimension;
    std::vector<float> centroid;

    FaissBBQFlat(int64_t _numVectors, int32_t _quantizedVectorBytes, float _centroidDp,
                 int32_t _dimension, const float* _centroid)
        : faiss::Index(_dimension, faiss::MetricType::METRIC_INNER_PRODUCT),
          numVectors(_numVectors),
          quantizedVectorBytes(_quantizedVectorBytes),
          centroidDp(_centroidDp),
          oneElementSize(_quantizedVectorBytes + 3 * sizeof(float) + sizeof(int32_t)),
          quantizedVectorsAndCorrectionFactors(_numVectors * oneElementSize),
          dimension(_dimension),
          centroid(_centroid, _centroid + _dimension) {

        quantizedVectorsAndCorrectionFactors.resize(0);
    }

    faiss::DistanceComputer* get_distance_computer() const final {
        return new BBQDistanceComputer(
            oneElementSize,
            quantizedVectorsAndCorrectionFactors.data(),
            centroidDp,
            dimension,
            numVectors,
            centroid.data());
    }

    void reset() final {
        throw std::runtime_error("FaissBBQFlat::reset() not implemented.");
    }

    void search(
            faiss::idx_t n,
            const float* x,
            faiss::idx_t k,
            float* distances,
            faiss::idx_t* labels,
            const faiss::SearchParameters* params = nullptr) const final {
        throw std::runtime_error("FaissBBQFlat::search() not implemented.");
    }

    void add(faiss::idx_t n, const float* x) final {
        ntotal += n;
    }
};

#endif //KNNPLUGIN_JNI_FAISS_BBQ_FLAT_H
