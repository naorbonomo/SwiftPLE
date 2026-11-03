//
//  PLEMmapGatherer.swift
//  SwiftPLE
//
//  Memory-maps the out-of-core 4-bit Per-Layer Embedding (.bin) table via POSIX mmap.
//  Executes zero-allocation CPU gathers and native Apple Silicon GPU dequantization via MLX.
//

import Foundation
import MLX

public final class PLEMmapGatherer: @unchecked Sendable {
    private var fd: Int32 = -1
    private var mmapPtr: UnsafeMutableRawPointer?
    private var fileSize: Int = 0

    public let vocabSize: Int
    public let numLayers: Int
    public let hiddenPerLayer: Int
    public let groupSize: Int
    public let totalPleDim: Int
    public let numGroups: Int
    public let numU32PerRow: Int
    public let bytesWeights: Int
    public let bytesScales: Int
    public let bytesBiases: Int
    public let bytesPerToken: Int

    private var singleW: [UInt32]
    private var singleS: [UInt16]
    private var singleB: [UInt16]
    private let pleScale: MLXArray

    public init?(fileURL: URL, config: PLEModelConfiguration) {
        self.vocabSize = config.vocabSize
        self.numLayers = config.numHiddenLayers
        self.hiddenPerLayer = config.hiddenSizePerLayerInput
        self.groupSize = config.manifest?.quantization.groupSize ?? 64
        self.totalPleDim = numLayers * hiddenPerLayer
        self.numGroups = totalPleDim / groupSize
        self.numU32PerRow = totalPleDim / 8

        self.bytesWeights = numU32PerRow * 4
        self.bytesScales = numGroups * 2
        self.bytesBiases = numGroups * 2
        self.bytesPerToken = bytesWeights + bytesScales + bytesBiases

        self.singleW = [UInt32](repeating: 0, count: numU32PerRow)
        self.singleS = [UInt16](repeating: 0, count: numGroups)
        self.singleB = [UInt16](repeating: 0, count: numGroups)
        self.pleScale = MLXArray(Float(config.embedTokensPerLayerScale), dtype: .bfloat16)

        let path = fileURL.path
        fd = open(path, O_RDONLY)
        guard fd >= 0 else { return nil }

        var sb = stat()
        guard fstat(fd, &sb) == 0 else {
            close(fd)
            return nil
        }
        fileSize = Int(sb.st_size)
        guard fileSize >= vocabSize * bytesPerToken else {
            close(fd)
            return nil
        }

        mmapPtr = mmap(nil, fileSize, PROT_READ, MAP_SHARED, fd, 0)
        guard mmapPtr != MAP_FAILED, mmapPtr != nil else {
            close(fd)
            return nil
        }
        madvise(mmapPtr, fileSize, MADV_RANDOM)
    }

    deinit {
        if let ptr = mmapPtr, ptr != MAP_FAILED {
            munmap(ptr, fileSize)
        }
        if fd >= 0 {
            close(fd)
        }
    }

    /// Gathers raw packed quantized bytes from mmap and dequantizes on Apple Silicon GPU.
    /// Returns shape: [1, tokens.count, numLayers, hiddenPerLayer] in bfloat16.
    public func gather(tokenIds: [Int]) -> MLXArray {
        let nTokens = tokenIds.count
        guard let base = mmapPtr, nTokens > 0 else {
            return MLXArray.zeros([1, nTokens, numLayers, hiddenPerLayer], dtype: .bfloat16)
        }

        if nTokens == 1 {
            let clamped = min(max(tokenIds[0], 0), vocabSize - 1)
            let src = base.advanced(by: clamped * bytesPerToken)

            singleW.withUnsafeMutableBufferPointer { wBuf in
                memcpy(wBuf.baseAddress!, src, bytesWeights)
            }
            singleS.withUnsafeMutableBufferPointer { sBuf in
                memcpy(sBuf.baseAddress!, src.advanced(by: bytesWeights), bytesScales)
            }
            singleB.withUnsafeMutableBufferPointer { bBuf in
                memcpy(bBuf.baseAddress!, src.advanced(by: bytesWeights + bytesScales), bytesBiases)
            }

            let wArray = MLXArray(singleW, [1, numU32PerRow])
            let sArray = MLXArray(singleS, [1, numGroups]).view(dtype: .bfloat16)
            let bArray = MLXArray(singleB, [1, numGroups]).view(dtype: .bfloat16)

            let dequant = dequantized(wArray, scales: sArray, biases: bArray, groupSize: groupSize, bits: 4) * pleScale
            return dequant.reshaped([1, 1, numLayers, hiddenPerLayer])
        }

        var weightsBuffer = [UInt32](repeating: 0, count: nTokens * numU32PerRow)
        var scalesBuffer = [UInt16](repeating: 0, count: nTokens * numGroups)
        var biasesBuffer = [UInt16](repeating: 0, count: nTokens * numGroups)

        weightsBuffer.withUnsafeMutableBufferPointer { wBuf in
            scalesBuffer.withUnsafeMutableBufferPointer { sBuf in
                biasesBuffer.withUnsafeMutableBufferPointer { bBuf in
                    guard let wPtr = wBuf.baseAddress,
                          let sPtr = sBuf.baseAddress,
                          let bPtr = bBuf.baseAddress else { return }

                    for (i, token) in tokenIds.enumerated() {
                        let clamped = min(max(token, 0), vocabSize - 1)
                        let tokenOffset = clamped * bytesPerToken
                        let src = base.advanced(by: tokenOffset)

                        memcpy(wPtr.advanced(by: i * numU32PerRow), src, bytesWeights)
                        memcpy(sPtr.advanced(by: i * numGroups), src.advanced(by: bytesWeights), bytesScales)
                        memcpy(bPtr.advanced(by: i * numGroups), src.advanced(by: bytesWeights + bytesScales), bytesBiases)
                    }
                }
            }
        }

        let wArray = MLXArray(weightsBuffer, [nTokens, numU32PerRow])
        let sArray = MLXArray(scalesBuffer, [nTokens, numGroups]).view(dtype: .bfloat16)
        let bArray = MLXArray(biasesBuffer, [nTokens, numGroups]).view(dtype: .bfloat16)

        let dequant = dequantized(wArray, scales: sArray, biases: bArray, groupSize: groupSize, bits: 4) * pleScale
        return dequant.reshaped([1, nTokens, numLayers, hiddenPerLayer])
    }
}

