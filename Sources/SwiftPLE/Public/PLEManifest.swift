//
//  PLEManifest.swift
//  SwiftPLE
//
//  Describes the binary format, dimension offsets, and quantization schema
//  for out-of-core memory-mapped Per-Layer Embedding (.bin) tables.
//

import Foundation

public struct PLEManifest: Codable, Sendable {
    public var formatVersion: Int
    public var sourceRepo: String?
    public var pleBinaryFile: String
    public var pleBinarySizeBytes: Int64
    public var pleSha256: String?
    public var vocabSize: Int
    public var numLayers: Int
    public var pleDimPerLayer: Int
    public var totalPleDim: Int
    public var quantization: PLEQuantizationInfo
    public var rowLayout: PLERowLayout

    public struct PLEQuantizationInfo: Codable, Sendable {
        public var bits: Int
        public var groupSize: Int
        public var scalesDtype: String

        public init(bits: Int = 4, groupSize: Int = 64, scalesDtype: String = "bfloat16") {
            self.bits = bits
            self.groupSize = groupSize
            self.scalesDtype = scalesDtype
        }

        enum CodingKeys: String, CodingKey {
            case bits
            case groupSize = "group_size"
            case scalesDtype = "scales_dtype"
        }
    }

    public struct PLERowLayout: Codable, Sendable {
        public var bytesPerToken: Int
        public var weightsOffset: Int
        public var weightsBytes: Int
        public var scalesOffset: Int
        public var scalesBytes: Int
        public var biasesOffset: Int
        public var biasesBytes: Int

        public init(
            bytesPerToken: Int,
            weightsOffset: Int,
            weightsBytes: Int,
            scalesOffset: Int,
            scalesBytes: Int,
            biasesOffset: Int,
            biasesBytes: Int
        ) {
            self.bytesPerToken = bytesPerToken
            self.weightsOffset = weightsOffset
            self.weightsBytes = weightsBytes
            self.scalesOffset = scalesOffset
            self.scalesBytes = scalesBytes
            self.biasesOffset = biasesOffset
            self.biasesBytes = biasesBytes
        }

        enum CodingKeys: String, CodingKey {
            case bytesPerToken = "bytes_per_token"
            case weightsOffset = "weights_offset"
            case weightsBytes = "weights_bytes"
            case scalesOffset = "scales_offset"
            case scalesBytes = "scales_bytes"
            case biasesOffset = "biases_offset"
            case biasesBytes = "biases_bytes"
        }
    }

    public init(
        formatVersion: Int = 1,
        sourceRepo: String? = nil,
        pleBinaryFile: String = "gemma4_ple_table.bin",
        pleBinarySizeBytes: Int64,
        pleSha256: String? = nil,
        vocabSize: Int,
        numLayers: Int,
        pleDimPerLayer: Int,
        totalPleDim: Int,
        quantization: PLEQuantizationInfo,
        rowLayout: PLERowLayout
    ) {
        self.formatVersion = formatVersion
        self.sourceRepo = sourceRepo
        self.pleBinaryFile = pleBinaryFile
        self.pleBinarySizeBytes = pleBinarySizeBytes
        self.pleSha256 = pleSha256
        self.vocabSize = vocabSize
        self.numLayers = numLayers
        self.pleDimPerLayer = pleDimPerLayer
        self.totalPleDim = totalPleDim
        self.quantization = quantization
        self.rowLayout = rowLayout
    }

    enum CodingKeys: String, CodingKey {
        case formatVersion = "format_version"
        case sourceRepo = "source_repo"
        case pleBinaryFile = "ple_binary_file"
        case pleBinarySizeBytes = "ple_binary_size_bytes"
        case pleSha256 = "ple_sha256"
        case vocabSize = "vocab_size"
        case numLayers = "num_layers"
        case pleDimPerLayer = "ple_dim_per_layer"
        case totalPleDim = "total_ple_dim"
        case quantization
        case rowLayout = "row_layout"
    }
}

