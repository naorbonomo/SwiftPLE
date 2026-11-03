//
//  GemmaPLEModel.swift
//  SwiftPLE
//
//  Gemma PLE transformer architecture with out-of-core per-layer embeddings and KV cache reuse.
//  Executes sequence prefill and single-token decode passes in Apple Silicon unified memory.
//

import Foundation
import MLX
import MLXNN

public final class GemmaPLEModel: Module, @unchecked Sendable {
    public let config: PLEModelConfiguration

    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
    @ModuleInfo(key: "per_layer_model_projection") public var perLayerModelProjection: Linear
    @ModuleInfo(key: "per_layer_projection_norm") public var perLayerProjectionNorm: RMSNorm
    @ModuleInfo(key: "layers") public var layers: [GemmaPLEDecoderLayer]
    @ModuleInfo(key: "norm") public var norm: RMSNorm

    nonisolated override init() { fatalError() }

    public init(config: PLEModelConfiguration) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self._perLayerModelProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.numHiddenLayers * config.hiddenSizePerLayerInput,
            bias: false
        )
        self._perLayerProjectionNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSizePerLayerInput,
            eps: config.rmsNormEps
        )
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map { i in
            GemmaPLEDecoderLayer(config: config, layerIndex: i)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    public func callAsFunction(
        inputs: MLXArray,
        externalPLE: MLXArray,
        caches: [GemmaPLEKVCache]? = nil,
        applySoftcapping: Bool = false
    ) -> MLXArray {
        let b = inputs.shape[0]
        let l = inputs.shape[1]

        var h = embedTokens(inputs) * config.embedScale

        // Per-layer model projection
        var pleProj = perLayerModelProjection(h) * config.perLayerProjectionScale
        pleProj = pleProj.reshaped([b, l, config.numHiddenLayers, config.hiddenSizePerLayerInput])
        pleProj = perLayerProjectionNorm(pleProj)

        // Combined per-layer embeddings
        let ple = externalPLE.asType(pleProj.dtype)
        let combinedPLE = (pleProj + ple) * config.perLayerInputScale

        // Prefill masks if sequence length > 1 (cast explicitly to model dtype for MLXFast SDPA)
        var fullMask: MLXArray? = nil
        var slidingMask: MLXArray? = nil
        if l > 1 {
            let negVal = MLXArray(Float(-1e9), dtype: h.dtype)
            let causal = triu(MLXArray.full([l, l], values: negVal), k: 1)
            fullMask = causal
            let lower = tril(MLXArray.full([l, l], values: negVal), k: -config.slidingWindow)
            slidingMask = (causal + lower).asType(h.dtype)
        }

        let currentOffset = caches?.first?.offset ?? 0
        var lastSlidingKV: (keys: MLXArray, values: MLXArray)?
        var lastFullKV: (keys: MLXArray, values: MLXArray)?
        let firstSharedIdx = config.numHiddenLayers - config.numKvSharedLayers

        for idx in 0..<config.numHiddenLayers {
            let layer = layers[idx]
            let cache = (caches != nil && idx < caches!.count) ? caches![idx] : nil
            let pleSlice = combinedPLE[0..., 0..., idx, 0...]

            let sharedKV: (keys: MLXArray, values: MLXArray)?
            if idx >= firstSharedIdx {
                sharedKV = config.isFullAttention(layerIndex: idx) ? lastFullKV : lastSlidingKV
            } else {
                sharedKV = nil
            }

            let mask = config.isFullAttention(layerIndex: idx) ? fullMask : slidingMask
            let (layerOut, kv) = layer(
                h,
                mask: mask,
                cache: cache,
                offset: currentOffset,
                perLayerInput: pleSlice,
                sharedKV: sharedKV
            )
            h = layerOut

            if idx < firstSharedIdx {
                if config.isFullAttention(layerIndex: idx) {
                    lastFullKV = kv
                } else {
                    lastSlidingKV = kv
                }
            }
        }

        h = norm(h)

        // Only compute logits for the final token during prefill/generation
        let lastH = (l > 1) ? h[0..., (l - 1)..., 0...] : h
        var logits = embedTokens.asLinear(lastH)

        // For greedy decoding, tanh is monotonic and does not alter the argmax index.
        if applySoftcapping && config.finalLogitSoftcapping > 0 {
            logits = tanh(logits / config.finalLogitSoftcapping) * config.finalLogitSoftcapping
        }

        return logits
    }

    public func makeCaches() -> [GemmaPLEKVCache] {
        let firstSharedIdx = config.numHiddenLayers - config.numKvSharedLayers
        return (0..<firstSharedIdx).map { _ in GemmaPLEKVCache() }
    }
}

