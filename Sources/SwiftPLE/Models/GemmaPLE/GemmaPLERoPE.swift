//
//  GemmaPLERoPE.swift
//  SwiftPLE
//
//  Rotary position embeddings for Gemma PLE sliding and full attention layers.
//  Accelerated via MLXNN.RoPE and Apple Silicon fused Metal kernels.
//

import Foundation
import MLX
import MLXNN

public final class GemmaPLERoPE: @unchecked Sendable {
    public let dimensions: Int
    public let rope: MLXNN.RoPE

    public init(dimensions: Int, isFullAttention: Bool) {
        self.dimensions = dimensions
        if isFullAttention {
            // Full attention: partialRotaryFactor = 0.25, base = 1_000_000.0
            let rotatedDims = Int(Float(dimensions) * 0.25)
            self.rope = MLXNN.RoPE(dimensions: rotatedDims, traditional: false, base: 1_000_000.0)
        } else {
            // Sliding window: theta = 10_000.0
            self.rope = MLXNN.RoPE(dimensions: dimensions, traditional: false, base: 10_000.0)
        }
    }

    public func callAsFunction(_ x: MLXArray, offset: Int) -> MLXArray {
        rope(x, offset: offset)
    }
}

