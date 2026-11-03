//
//  PLEWeightsLoader.swift
//  SwiftPLE
//
//  Loads 4-bit quantized safetensors weights for Gemma PLE models, applies tensor keys,
//  and initializes model parameters on Apple Silicon unified memory.
//

import Foundation
import MLX
import MLXNN

public enum PLEWeightsLoader {
    public static func load(from directory: URL, config: PLEModelConfiguration) throws -> GemmaPLEModel {
        let model = GemmaPLEModel(config: config)

        // Quantize Linear and Embedding modules to 4-bit with specified group size
        let groupSize = config.manifest?.quantization.groupSize ?? 64
        let bits = config.manifest?.quantization.bits ?? 4
        quantize(model: model, groupSize: groupSize, bits: bits) { _, module in
            module is Linear || module is Embedding
        }

        Memory.cacheLimit = 64 * 1024 * 1024

        let safetensors = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }

        var sanitized: [String: MLXArray] = [:]
        let prefixes = ["language_model.model.", "model."]

        for url in safetensors {
            let shard = try MLX.loadArrays(url: url)
            for (key, array) in shard {
                var strippedKey: String?
                for p in prefixes {
                    if key.hasPrefix(p) {
                        strippedKey = String(key.dropFirst(p.count))
                        break
                    }
                }
                guard let cleanKey = strippedKey else { continue }

                // Extract per-layer scalar multipliers directly
                if cleanKey.contains("layer_scalar") {
                    let parts = cleanKey.split(separator: ".")
                    if parts.count >= 2, let idx = Int(parts[1]), idx < model.layers.count {
                        model.layers[idx].layerScalar = array
                    }
                    continue
                }

                sanitized[cleanKey] = array
            }
        }

        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: .none)
        sanitized.removeAll(keepingCapacity: false)
        Memory.clearCache()
        eval(model)
        Memory.clearCache()
        return model
    }
}

