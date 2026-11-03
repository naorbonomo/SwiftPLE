//
//  GemmaPLEDecoderLayer.swift
//  SwiftPLE
//
//  Transformer decoder layer with pre/post layer norms, GeGLU MLP, and per-layer input gating.
//  Executes attention and residual updates with Apple Silicon Metal acceleration.
//

import Foundation
import MLX
import MLXNN

public final class GemmaPLEDecoderLayer: Module, @unchecked Sendable {
    public let layerIndex: Int

    @ModuleInfo(key: "input_layernorm") public var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "self_attn") public var selfAttn: GemmaPLEAttention
    @ModuleInfo(key: "post_attention_layernorm") public var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") public var preFeedforwardLayerNorm: RMSNorm
    @ModuleInfo(key: "mlp") public var mlp: GemmaPLEMLP
    @ModuleInfo(key: "post_feedforward_layernorm") public var postFeedforwardLayerNorm: RMSNorm

    @ModuleInfo(key: "per_layer_input_gate") public var perLayerInputGate: Linear
    @ModuleInfo(key: "per_layer_projection") public var perLayerProjection: Linear
    @ModuleInfo(key: "post_per_layer_input_norm") public var postPerLayerInputNorm: RMSNorm
    public var layerScalar: MLXArray?

    nonisolated override init() { fatalError() }

    public init(config: PLEModelConfiguration, layerIndex: Int) {
        self.layerIndex = layerIndex
        let hDim = config.hiddenSize
        let pleDim = config.hiddenSizePerLayerInput
        let eps = config.rmsNormEps

        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: hDim, eps: eps)
        self._selfAttn.wrappedValue = GemmaPLEAttention(config: config, layerIndex: layerIndex)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: hDim, eps: eps)
        self._preFeedforwardLayerNorm.wrappedValue = RMSNorm(dimensions: hDim, eps: eps)
        self._mlp.wrappedValue = GemmaPLEMLP(hiddenSize: hDim, intermediateSize: config.intermediateSize)
        self._postFeedforwardLayerNorm.wrappedValue = RMSNorm(dimensions: hDim, eps: eps)

        self._perLayerInputGate.wrappedValue = Linear(hDim, pleDim, bias: false)
        self._perLayerProjection.wrappedValue = Linear(pleDim, hDim, bias: false)
        self._postPerLayerInputNorm.wrappedValue = RMSNorm(dimensions: hDim, eps: eps)
        self.layerScalar = MLXArray.ones([1])
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray?,
        cache: GemmaPLEKVCache?,
        offset: Int,
        perLayerInput: MLXArray?,
        sharedKV: (keys: MLXArray, values: MLXArray)? = nil
    ) -> (output: MLXArray, kv: (keys: MLXArray, values: MLXArray)) {
        var residual = x
        var h = inputLayerNorm(x)
        let (attnOut, kv) = selfAttn(h, mask: mask, cache: cache, offset: offset, sharedKV: sharedKV)
        h = postAttentionLayerNorm(attnOut)
        h = residual + h

        residual = h
        h = preFeedforwardLayerNorm(h)
        h = mlp(h)
        h = postFeedforwardLayerNorm(h)
        h = residual + h

        if let ple = perLayerInput {
            residual = h
            var gate = perLayerInputGate(h)
            gate = geluApproximate(gate)
            gate = gate * ple
            gate = perLayerProjection(gate)
            gate = postPerLayerInputNorm(gate)
            h = residual + gate
        }

        if let scalar = layerScalar {
            h = h * scalar
        }

        return (h, kv)
    }
}

