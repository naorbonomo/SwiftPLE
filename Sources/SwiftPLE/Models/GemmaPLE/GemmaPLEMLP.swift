//
//  GemmaPLEMLP.swift
//  SwiftPLE
//
//  Feed-forward neural network block with GeGLU activation for Gemma PLE.
//  Executes gated linear transformations in 4-bit quantized MLX space.
//

import Foundation
import MLX
import MLXNN

public final class GemmaPLEMLP: Module, @unchecked Sendable {
    @ModuleInfo(key: "gate_proj") public var gateProj: Linear
    @ModuleInfo(key: "up_proj") public var upProj: Linear
    @ModuleInfo(key: "down_proj") public var downProj: Linear

    nonisolated override init() { fatalError() }

    public init(hiddenSize: Int, intermediateSize: Int) {
        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gate = geluApproximate(gateProj(x))
        let up = upProj(x)
        return downProj(gate * up)
    }
}

