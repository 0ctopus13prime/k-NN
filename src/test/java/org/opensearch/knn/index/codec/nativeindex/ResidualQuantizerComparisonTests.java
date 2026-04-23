/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.opensearch.test.OpenSearchTestCase;

import java.util.Random;

/**
 * Comparison test: verifies our ResidualQuantizer pipeline produces the same residuals
 * as the POC's approach (scalarQuantize → raw codes → dequantize → residual).
 *
 * This test simulates the full POC flow:
 * 1. Start with raw vectors + centroid
 * 2. Call scalarQuantize() to get per-dimension codes (0/1) and correction factors
 *    (this also centers the vector in-place)
 * 3. Compute residuals from the raw codes: r[d] = centeredVec[d] - dequantize(code[d])
 * 4. Quantize residuals to 4-bit with per-vector bounds [-delta/2, delta/2]
 *
 * Then compares against our pipeline:
 * 1. packAsBinary(codes) → binary code
 * 2. Our computeResidual(centeredVec[d], binaryCode, d, lower, delta)
 * 3. Our quantizeResidualUniform(residuals, scratch, dim, delta)
 */
public class ResidualQuantizerComparisonTests extends OpenSearchTestCase {

    /**
     * Compare residual computation: POC (raw codes) vs ours (unpacked from binary).
     * Uses real scalarQuantize to get authentic codes and correction factors.
     */
    public void testResidualComputation_matchesPOC() {
        int dimension = 128;
        Random rng = new Random(42);

        // Generate random vectors and centroid
        float[] centroid = new float[dimension];
        for (int d = 0; d < dimension; d++) {
            centroid[d] = rng.nextFloat() * 2 - 1;
        }

        float[] originalVec = new float[dimension];
        for (int d = 0; d < dimension; d++) {
            originalVec[d] = rng.nextFloat() * 2 - 1;
        }

        // Copy for POC path (scalarQuantize mutates in-place)
        float[] vecForPOC = originalVec.clone();
        float[] vecForOurs = originalVec.clone();

        // Step 1: scalarQuantize — produces raw codes in quantizationScratch, centers vec in-place
        OptimizedScalarQuantizer quantizer = new OptimizedScalarQuantizer(VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT);
        int discreteDims = ((dimension + 63) / 64) * 64;
        byte[] quantizationScratch = new byte[discreteDims];

        OptimizedScalarQuantizer.QuantizationResult corrections = quantizer.scalarQuantize(
            vecForPOC,
            quantizationScratch,
            (byte) 1,
            centroid
        );
        // vecForPOC is now centered

        float lower = corrections.lowerInterval();
        float upper = corrections.upperInterval();
        float delta = upper - lower;

        // Step 2: POC residuals — from raw per-dimension codes
        float[] pocResiduals = new float[dimension];
        for (int d = 0; d < dimension; d++) {
            int code = quantizationScratch[d] & 0xFF;  // 0 or 1
            float dequantized = lower + code * delta;    // same as POC's dequantize()
            pocResiduals[d] = vecForPOC[d] - dequantized;
        }

        // Step 3: Our residuals — from bit-packed binary code
        // First, pack the raw codes to binary (same as Lucene does)
        byte[] binaryCode = new byte[discreteDims / 8];
        OptimizedScalarQuantizer.packAsBinary(quantizationScratch, binaryCode);

        // Center vecForOurs the same way scalarQuantize would
        for (int d = 0; d < dimension; d++) {
            vecForOurs[d] = vecForOurs[d] - centroid[d];
        }

        float[] ourResiduals = new float[dimension];
        for (int d = 0; d < dimension; d++) {
            ourResiduals[d] = ResidualQuantizer.computeResidual(vecForOurs[d], binaryCode, d, lower, delta);
        }

        // Compare residuals
        System.out.println("=== Residual comparison (first 16 dims) ===");
        for (int d = 0; d < Math.min(16, dimension); d++) {
            System.out.printf(
                "  dim %3d: POC=%.6f  Ours=%.6f  diff=%.9f  code=%d  unpackedBit=%d%n",
                d,
                pocResiduals[d],
                ourResiduals[d],
                Math.abs(pocResiduals[d] - ourResiduals[d]),
                quantizationScratch[d] & 0xFF,
                ResidualQuantizer.unpackBit(binaryCode, d)
            );
        }

        // Verify: raw code should match unpacked bit for every dimension
        int bitMismatches = 0;
        for (int d = 0; d < dimension; d++) {
            int rawCode = quantizationScratch[d] & 0xFF;
            int unpackedBit = ResidualQuantizer.unpackBit(binaryCode, d);
            if (rawCode != unpackedBit) {
                bitMismatches++;
                if (bitMismatches <= 10) {
                    System.out.printf("  BIT MISMATCH dim %d: rawCode=%d unpackedBit=%d%n", d, rawCode, unpackedBit);
                }
            }
        }
        System.out.println("Total bit mismatches: " + bitMismatches + " / " + dimension);
        assertEquals("All unpacked bits should match raw codes", 0, bitMismatches);

        // Verify: residuals should be identical
        float maxResidualDiff = 0;
        for (int d = 0; d < dimension; d++) {
            float diff = Math.abs(pocResiduals[d] - ourResiduals[d]);
            maxResidualDiff = Math.max(maxResidualDiff, diff);
        }
        System.out.println("Max residual diff: " + maxResidualDiff);
        assertEquals("Residuals should match", 0.0f, maxResidualDiff, 1e-6f);

        // Step 4: Compare 4-bit quantization
        byte[] pocScratch = new byte[dimension];
        int pocSum = pocQuantizeResidualUniform(pocResiduals, pocScratch, (byte) 4, delta);

        byte[] ourScratch = new byte[dimension];
        int ourSum = ResidualQuantizer.quantizeResidualUniform(ourResiduals, ourScratch, dimension, delta);

        int quantMismatches = 0;
        for (int d = 0; d < dimension; d++) {
            if ((pocScratch[d] & 0xFF) != (ourScratch[d] & 0xFF)) {
                quantMismatches++;
                if (quantMismatches <= 10) {
                    System.out.printf(
                        "  QUANT MISMATCH dim %d: POC=%d Ours=%d  pocResidual=%.6f ourResidual=%.6f%n",
                        d,
                        pocScratch[d] & 0xFF,
                        ourScratch[d] & 0xFF,
                        pocResiduals[d],
                        ourResiduals[d]
                    );
                }
            }
        }
        System.out.println("Total quant mismatches: " + quantMismatches + " / " + dimension);
        System.out.println("POC componentSum: " + pocSum + "  Ours: " + ourSum);

        assertEquals("Component sums should match", pocSum, ourSum);
        assertEquals("All 4-bit codes should match", 0, quantMismatches);
    }

    /**
     * Compare search-side score correction: POC's rerank() vs our computeCorrectedScore().
     *
     * Both start from the same:
     * - 4-bit quantized residual codes (from build side, verified identical above)
     * - Query vector, centroid, phase-1 MIP-scaled score
     *
     * POC flow: unscale → extract bits MSB-first → dequantize → dot product → add → re-scale
     * Our flow: unscale → extract nibbles → dequantize → dot product → add → re-scale
     *
     * The key difference: POC uses packBits (MSB-first bit layout) for 4-bit,
     * we use packNibbles (low-nibble-first). If the packed bytes differ, the
     * dequantized values will differ, and scores will diverge.
     */
    public void testSearchSideScoring_matchesPOC() {
        int dimension = 128;
        Random rng = new Random(42);

        // Generate random centroid and query
        float[] centroid = new float[dimension];
        float[] queryVector = new float[dimension];
        for (int d = 0; d < dimension; d++) {
            centroid[d] = rng.nextFloat() * 2 - 1;
            queryVector[d] = rng.nextFloat() * 2 - 1;
        }

        // Generate a random already-centered vector and quantize it
        float[] centeredVec = new float[dimension];
        for (int d = 0; d < dimension; d++) {
            centeredVec[d] = rng.nextFloat() * 2 - 1;
        }

        // Simulate 1-bit quantization to get lower/upper
        OptimizedScalarQuantizer quantizer = new OptimizedScalarQuantizer(VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT);
        // We need a fresh copy because scalarQuantize mutates in-place
        float[] vecCopy = centeredVec.clone();
        // Add centroid back so scalarQuantize can subtract it again
        for (int d = 0; d < dimension; d++) {
            vecCopy[d] += centroid[d];
        }
        int discreteDims = ((dimension + 63) / 64) * 64;
        byte[] quantScratch = new byte[discreteDims];
        OptimizedScalarQuantizer.QuantizationResult corrections = quantizer.scalarQuantize(vecCopy, quantScratch, (byte) 1, centroid);
        float lower = corrections.lowerInterval();
        float upper = corrections.upperInterval();
        float delta = upper - lower;
        float halfDelta = delta / 2.0f;

        // Compute residuals (same for both paths since build side is verified identical)
        float[] residuals = new float[dimension];
        for (int d = 0; d < dimension; d++) {
            int code = quantScratch[d] & 0xFF;
            float dequantized = lower + code * delta;
            residuals[d] = vecCopy[d] - dequantized;  // vecCopy is now centered
        }

        // Quantize residuals to 4-bit (same for both paths)
        byte[] residualCodes = new byte[dimension];
        int componentSum = pocQuantizeResidualUniform(residuals, residualCodes, (byte) 4, delta);

        // === POC path: packBits (MSB-first) + POC rerank ===
        int pocPackedSize = pocPackedBytesSize(dimension, 4);
        byte[] pocPacked = new byte[pocPackedSize];
        pocPackBits(residualCodes, pocPacked, 4);

        float[] centeredQuery = new float[dimension];
        for (int d = 0; d < dimension; d++) {
            centeredQuery[d] = queryVector[d] - centroid[d];
        }

        float pocCorrection = pocComputeCorrection(centeredQuery, pocPacked, dimension, -halfDelta, halfDelta, 4);

        // === Our path: packNibbles + computeCorrectedScore ===
        int ourPackedSize = (dimension + 1) / 2;
        byte[] ourPacked = new byte[ourPackedSize];
        for (int d = 0; d < dimension; d += 2) {
            int q0 = residualCodes[d] & 0x0F;
            int q1 = (d + 1 < dimension) ? (residualCodes[d + 1] & 0x0F) : 0;
            ourPacked[d / 2] = ResidualQuantizer.packNibbles(q0, q1);
        }

        float[] qPrime = ResidualQuantizer.computeQPrime(queryVector, centroid);
        // Use a known rawDot so we can compare after scale/unscale
        float rawDotPhase1 = 5.0f;
        float phase1Score = ResidualQuantizer.scaleMaxInnerProductScore(rawDotPhase1);

        float ourResult = ResidualQuantizer.computeCorrectedScore(qPrime, ourPacked, -halfDelta, halfDelta, phase1Score, dimension);
        float ourRawTotal = ResidualQuantizer.unscaleMaxInnerProductScore(ourResult);
        float ourCorrection = ourRawTotal - rawDotPhase1;

        // === Compare ===
        System.out.println("=== Search-side scoring comparison ===");
        System.out.printf("  POC correction:  %.6f%n", pocCorrection);
        System.out.printf("  Our correction:  %.6f%n", ourCorrection);
        System.out.printf("  Diff:            %.9f%n", Math.abs(pocCorrection - ourCorrection));
        System.out.printf("  POC packed size: %d bytes%n", pocPackedSize);
        System.out.printf("  Our packed size: %d bytes%n", ourPackedSize);

        // Compare individual nibbles to find where they diverge
        int nibbleMismatches = 0;
        for (int d = 0; d < dimension; d++) {
            int pocCode = pocExtractCode(pocPacked, d, 4);
            int ourNibble = (d % 2 == 0) ? (ourPacked[d / 2] & 0x0F) : ((ourPacked[d / 2] >>> 4) & 0x0F);
            if (pocCode != ourNibble) {
                nibbleMismatches++;
                if (nibbleMismatches <= 10) {
                    System.out.printf(
                        "  NIBBLE MISMATCH dim %d: POC=%d Ours=%d  residualCode=%d%n",
                        d,
                        pocCode,
                        ourNibble,
                        residualCodes[d] & 0xFF
                    );
                }
            }
        }
        System.out.println("Total nibble mismatches: " + nibbleMismatches + " / " + dimension);

        // Check per-dimension dequantized dot product contribution
        if (nibbleMismatches > 0) {
            float nSteps = 15.0f;
            float step = (halfDelta - (-halfDelta)) / nSteps;
            System.out.println("\n=== Per-dim dequantized value comparison (first 10 mismatches) ===");
            int shown = 0;
            for (int d = 0; d < dimension && shown < 10; d++) {
                int pocCode = pocExtractCode(pocPacked, d, 4);
                int ourNibble = (d % 2 == 0) ? (ourPacked[d / 2] & 0x0F) : ((ourPacked[d / 2] >>> 4) & 0x0F);
                if (pocCode != ourNibble) {
                    float pocDeq = -halfDelta + pocCode * step;
                    float ourDeq = -halfDelta + ourNibble * step;
                    System.out.printf(
                        "  dim %3d: pocCode=%2d ourNibble=%2d  pocDeq=%.4f ourDeq=%.4f  q'=%.4f%n",
                        d,
                        pocCode,
                        ourNibble,
                        pocDeq,
                        ourDeq,
                        centeredQuery[d]
                    );
                    shown++;
                }
            }
        }

        // The corrections should match if packing is consistent
        assertEquals("Corrections should match", pocCorrection, ourCorrection, 1e-3f);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // POC helper methods — copied from POC code
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * POC's quantizeResidualUniform.
     */
    private static int pocQuantizeResidualUniform(float[] residual, byte[] destination, byte bits, float delta) {
        float halfDelta = delta / 2.0f;
        int nSteps = (1 << bits) - 1;
        float step = delta / nSteps;
        int componentSum = 0;
        for (int i = 0; i < residual.length; i++) {
            float clamped = Math.min(Math.max(residual[i], -halfDelta), halfDelta);
            int code = Math.round((clamped + halfDelta) / step);
            code = Math.min(Math.max(code, 0), nSteps);
            destination[i] = (byte) code;
            componentSum += code;
        }
        return componentSum;
    }

    /**
     * POC's packBits — MSB-first bit packing for multi-bit codes.
     */
    private static void pocPackBits(byte[] codes, byte[] packed, int bitsPerCode) {
        java.util.Arrays.fill(packed, (byte) 0);
        for (int d = 0; d < codes.length; d++) {
            int code = codes[d] & 0xFF;
            int bitOffset = d * bitsPerCode;
            for (int b = 0; b < bitsPerCode; b++) {
                int globalBit = bitOffset + (bitsPerCode - 1 - b);
                int byteIdx = globalBit / 8;
                int bitIdx = 7 - (globalBit % 8);
                if ((code & (1 << b)) != 0) {
                    packed[byteIdx] |= (byte) (1 << bitIdx);
                }
            }
        }
    }

    /**
     * POC's packedBytesSize.
     */
    private static int pocPackedBytesSize(int dimension, int bitsPerCode) {
        int totalBits = dimension * bitsPerCode;
        return ((totalBits + 63) / 64) * 64 / 8;
    }

    /**
     * POC's multi-bit code extraction — MSB-first.
     */
    private static int pocExtractCode(byte[] packed, int d, int bitsPerCode) {
        int bitOffset = d * bitsPerCode;
        int code = 0;
        for (int b = 0; b < bitsPerCode; b++) {
            int globalBit = bitOffset + b;
            int byteIdx = globalBit / 8;
            int bitIdx = 7 - (globalBit % 8);
            code = (code << 1) | ((packed[byteIdx] >> bitIdx) & 1);
        }
        return code;
    }

    /**
     * POC's correction computation — extract codes MSB-first and compute dot product.
     */
    private static float pocComputeCorrection(
        float[] centeredQuery,
        byte[] packedResidual,
        int dimension,
        float rLow,
        float rHigh,
        int errorResidualBits
    ) {
        float qDotResidual = 0f;
        int nSteps = (1 << errorResidualBits) - 1;
        float step = (nSteps > 0) ? (rHigh - rLow) / nSteps : 0f;

        for (int d = 0; d < dimension; d++) {
            int bitOffset = d * errorResidualBits;
            int code = 0;
            for (int b = 0; b < errorResidualBits; b++) {
                int globalBit = bitOffset + b;
                int byteIdx = globalBit / 8;
                int bitIdx = 7 - (globalBit % 8);
                code = (code << 1) | ((packedResidual[byteIdx] >> bitIdx) & 1);
            }
            float dequantized = rLow + code * step;
            qDotResidual += centeredQuery[d] * dequantized;
        }
        return qDotResidual;
    }
}
