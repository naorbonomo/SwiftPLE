//
//  PLEModelConfiguration.swift
//  SwiftPLE
//
//  Hyperparameters and scaling constants for Gemma 4 architectures.
//  Decodes config.json directly and binds to optional ple_manifest.json metadata.
//

import Foundation

public struct PLEModelConfiguration: Codable, Sendable {
    public var hiddenSize: Int = 2560
    public var numHiddenLayers: Int = 42
    public var intermediateSize: Int = 10240
    public var numAttentionHeads: Int = 8
    public var numKeyValueHeads: Int = 2
    public var headDim: Int = 256
    public var globalHeadDim: Int = 512
    public var numKvSharedLayers: Int = 18
    public var rmsNormEps: Float = 1e-6
    public var vocabSize: Int = 262144
    public var hiddenSizePerLayerInput: Int = 256
    public var slidingWindow: Int = 512
    public var finalLogitSoftcapping: Float = 30.0

    public var manifest: PLEManifest? = nil

    public var embedScale: Float {
        pow(Float(hiddenSize), 0.5)
    }

    public var perLayerInputScale: Float {
        pow(2.0, -0.5)
    }

    public var perLayerProjectionScale: Float {
        pow(Float(hiddenSize), -0.5)
    }

    public var embedTokensPerLayerScale: Float {
        pow(Float(hiddenSizePerLayerInput), 0.5)
    }

    public func isFullAttention(layerIndex: Int) -> Bool {
        // Pattern: 5 sliding attention layers followed by 1 full attention layer (modulo 6 == 5)
        (layerIndex % 6) == 5
    }

    public func headDim(for layerIndex: Int) -> Int {
        isFullAttention(layerIndex: layerIndex) ? globalHeadDim : headDim
    }

    public static func load(from directory: URL) throws -> PLEModelConfiguration {
        let configFile = directory.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configFile)

        var config: PLEModelConfiguration
        if let json = try JSONSerialization.jsonObject(with: configData) as? [String: Any],
           let textCfg = json["text_config"] as? [String: Any],
           let textData = try? JSONSerialization.data(withJSONObject: textCfg) {
            config = try JSONDecoder().decode(PLEModelConfiguration.self, from: textData)
        } else {
            config = try JSONDecoder().decode(PLEModelConfiguration.self, from: configData)
        }

        let manifestFile = directory.appendingPathComponent("ple_manifest.json")
        if FileManager.default.fileExists(atPath: manifestFile.path),
           let manifestData = try? Data(contentsOf: manifestFile),
           let decodedManifest = try? JSONDecoder().decode(PLEManifest.self, from: manifestData) {
            config.manifest = decodedManifest
            config.vocabSize = decodedManifest.vocabSize
            config.numHiddenLayers = decodedManifest.numLayers
            config.hiddenSizePerLayerInput = decodedManifest.pleDimPerLayer
        }

        return config
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case globalHeadDim = "global_head_dim"
        case numKvSharedLayers = "num_kv_shared_layers"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case slidingWindow = "sliding_window"
        case finalLogitSoftcapping = "final_logit_softcapping"
    }
}
