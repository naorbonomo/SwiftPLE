//
//  GemmaPLEAttention.swift
//  SwiftPLE
//
//  Multi-head attention with RMSNorm QK normalization and rotary position embeddings.
//  Accelerated with MLXFast fused FlashAttention and GQA support.
//

import Foundation
import MLX
import MLXFast
import MLXNN

public final class GemmaPLEKVCache {
    public var keys: MLXArray?
    public var values: MLXArray?
    public var offset: Int = 0

    public init() {}
}

public final class GemmaPLEAttention: Module, @unchecked Sendable {
    public let layerIndex: Int
    public let isSliding: Bool
    public let hasKV: Bool
    public let nHeads: Int
    public let nKVHeads: Int
    public let headDim: Int
    public let eps: Float
    public let slidingWindow: Int

    @ModuleInfo(key: "q_proj") public var qProj: Linear
    @ModuleInfo(key: "k_proj") public var kProj: Linear?
    @ModuleInfo(key: "v_proj") public var vProj: Linear?
    @ModuleInfo(key: "o_proj") public var oProj: Linear
    @ModuleInfo(key: "q_norm") public var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") public var kNorm: RMSNorm?
    public var rope: GemmaPLERoPE

    nonisolated override init() { fatalError() }

    public init(config: PLEModelConfiguration, layerIndex: Int) {
        self.layerIndex = layerIndex
        self.isSliding = !config.isFullAttention(layerIndex: layerIndex)
        let firstSharedIdx = config.numHiddenLayers - config.numKvSharedLayers
        self.hasKV = layerIndex < firstSharedIdx
        self.nHeads = config.numAttentionHeads
        self.nKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim(for: layerIndex)
        self.eps = config.rmsNormEps
        self.slidingWindow = config.slidingWindow
        self.rope = GemmaPLERoPE(dimensions: self.headDim, isFullAttention: !self.isSliding)

        self._qProj.wrappedValue = Linear(config.hiddenSize, nHeads * headDim, bias: false)
        self._oProj.wrappedValue = Linear(nHeads * headDim, config.hiddenSize, bias: false)
        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)

        if hasKV {
            self._kProj.wrappedValue = Linear(config.hiddenSize, nKVHeads * headDim, bias: false)
            self._vProj.wrappedValue = Linear(config.hiddenSize, nKVHeads * headDim, bias: false)
            self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        } else {
            self._kProj.wrappedValue = nil
            self._vProj.wrappedValue = nil
            self._kNorm.wrappedValue = nil
        }
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray?,
        cache: GemmaPLEKVCache?,
        offset: Int,
        sharedKV: (keys: MLXArray, values: MLXArray)? = nil
    ) -> (output: MLXArray, kv: (keys: MLXArray, values: MLXArray)) {
        let b = x.shape[0]
        let l = x.shape[1]

        var q = qProj(x).reshaped([b, l, nHeads, headDim])
        q = qNorm(q)
        q = q.transposed(0, 2, 1, 3)
        q = rope(q, offset: offset)

        var keys: MLXArray
        var values: MLXArray

        if let shared = sharedKV {
            keys = shared.keys
            values = shared.values
        } else if hasKV, let kP = kProj, let vP = vProj, let kN = kNorm {
            var k = kP(x).reshaped([b, l, nKVHeads, headDim])
            k = kN(k)
            k = k.transposed(0, 2, 1, 3)
            k = rope(k, offset: offset)

            var v = vP(x).reshaped([b, l, nKVHeads, headDim])
            let vNormScale = rsqrt(mean(v * v, axis: -1, keepDims: true) + eps)
            v = v * vNormScale
            v = v.transposed(0, 2, 1, 3)

            if let c = cache {
                if let ck = c.keys, let cv = c.values {
                    k = concatenated([ck, k], axis: 2)
                    v = concatenated([cv, v], axis: 2)
                }
                if isSliding, k.shape[2] > slidingWindow {
                    let start = k.shape[2] - slidingWindow
                    let slicedK = k[0..., 0..., start..., 0...]
                    let slicedV = v[0..., 0..., start..., 0...]
                    c.keys = slicedK
                    c.values = slicedV
                    if l == 1 {
                        k = slicedK
                        v = slicedV
                    }
                } else {
                    c.keys = k
                    c.values = v
                }
                c.offset += l
            }
            keys = k
            values = v
        } else {
            fatalError("Layer \(layerIndex) has no KV projections and received no shared KV")
        }

        // Hardware-accelerated FlashAttention kernel with native Grouped-Query Attention (GQA)
        let attnMask = (l > 1 && mask != nil) ? mask!.asType(q.dtype) : nil
        let attn = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: keys,
            values: values,
            scale: 1.0,
            mask: attnMask
        )

        let out = oProj(attn.transposed(0, 2, 1, 3).reshaped([b, l, -1]))
        return (out, (keys, values))
    }
}

