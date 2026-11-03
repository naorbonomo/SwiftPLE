//
//  PLEModelClient.swift
//  SwiftPLE
//
//  High-level inference client for Gemma PLE models on Apple Silicon MLX.
//  Streams tokens, pipelines GPU/CPU execution, and records generation metrics.
//

import Foundation
import MLX
import MLXNN
import Tokenizers

public actor PLEModelClient {
    private var model: GemmaPLEModel?
    private var gatherer: PLEMmapGatherer?
    private var tokenizer: (any Tokenizer)?
    private var initTime: Double?
    private let modelDirectory: URL
    private let repoOrURL: String?
    private let hfToken: String?

    public static let maxContextWindow = 2048
    public static let defaultCacheLimitBytes = 64 * 1024 * 1024

    public init(modelDirectory: URL) {
        self.modelDirectory = modelDirectory
        self.repoOrURL = nil
        self.hfToken = nil
    }

    public init(modelRepo: String, hfToken: String? = nil, cacheDirectory: URL? = nil) {
        self.repoOrURL = modelRepo
        self.hfToken = hfToken
        self.modelDirectory = cacheDirectory ?? PLEHubDownloader.defaultDestination(for: modelRepo)
    }

    public func unload() {
        self.model = nil
        self.gatherer = nil
        self.tokenizer = nil
        self.initTime = nil
        Memory.clearCache()
    }

    public func prepare(onProgress: (@Sendable (Double) -> Void)? = nil) async throws {
        if model != nil, gatherer != nil, tokenizer != nil { return }
        let loadStart = CFAbsoluteTimeGetCurrent()
        Memory.cacheLimit = Self.defaultCacheLimitBytes
        defer { Memory.clearCache() }

        if let repo = repoOrURL {
            _ = try await PLEHubDownloader.download(
                from: repo,
                hfToken: hfToken,
                destinationDir: modelDirectory,
                onProgress: onProgress
            )
        }

        let config = try PLEModelConfiguration.load(from: modelDirectory)
        let binName = config.manifest?.pleBinaryFile ?? "gemma4_ple_table.bin"
        var pleURL = modelDirectory.appendingPathComponent(binName)
        if !FileManager.default.fileExists(atPath: pleURL.path) {
            pleURL = modelDirectory.appendingPathComponent("gemma4_ple_table.bin")
        }

        guard let pleGatherer = PLEMmapGatherer(fileURL: pleURL, config: config) else {
            throw NSError(
                domain: "PLEModelClient",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Failed to mmap PLE table at \(pleURL.path)"]
            )
        }
        self.gatherer = pleGatherer
        self.tokenizer = try await AutoTokenizer.from(modelFolder: modelDirectory)
        let loadedModel = try PLEWeightsLoader.load(from: modelDirectory, config: config)

        // Pre-warm Metal pipeline shaders (both masked prefill and unmasked decode)
        let warmupCaches = loadedModel.makeCaches()
        let warmupTokens = [2, 106, 1, 2]
        let warmupPLE = pleGatherer.gather(tokenIds: warmupTokens)
        let warmupInputs = MLXArray(warmupTokens.map { Int32($0) }, [1, warmupTokens.count])
        let warmupLogits = loadedModel(inputs: warmupInputs, externalPLE: warmupPLE, caches: warmupCaches)
        let warmupNextToken = argMax(warmupLogits, axis: -1)
        eval(warmupNextToken)

        let warmupSinglePLE = pleGatherer.gather(tokenIds: [1])
        let warmupSingleInput = MLXArray([Int32(1)], [1, 1])
        let warmupDecodeLogits = loadedModel(inputs: warmupSingleInput, externalPLE: warmupSinglePLE, caches: warmupCaches)
        let warmupDecodeNextToken = argMax(warmupDecodeLogits, axis: -1)
        eval(warmupDecodeNextToken)

        self.model = loadedModel
        self.initTime = CFAbsoluteTimeGetCurrent() - loadStart
    }

    public func generate(
        prompt: String,
        maxNewTokens: Int = 256,
        onToken: (@Sendable (String) -> Void)? = nil
    ) async throws -> PLEGenerationOutput {
        Memory.cacheLimit = Self.defaultCacheLimitBytes
        defer { Memory.clearCache() }
        try await prepare()
        guard let model, let gatherer, let tokenizer else {
            throw NSError(domain: "PLEModelClient", code: 2, userInfo: [NSLocalizedDescriptionKey: "Model not initialized"])
        }

        let formattedPrompt = "<bos><|turn>user\n\(prompt)<turn|>\n<|turn>model\n"
        var promptTokens = tokenizer.encode(text: formattedPrompt)
        if promptTokens.count > Self.maxContextWindow {
            promptTokens = Array(promptTokens.suffix(Self.maxContextWindow))
        }

        let promptStartTime = CFAbsoluteTimeGetCurrent()
        let caches = model.makeCaches()

        // 1. Prefill pass
        let promptPLE = gatherer.gather(tokenIds: promptTokens)
        let promptInputs = MLXArray(promptTokens.map { Int32($0) }, [1, promptTokens.count])
        let prefillLogits = model(inputs: promptInputs, externalPLE: promptPLE, caches: caches)
        var nextTokenTensor = argMax(prefillLogits, axis: -1)
        eval(nextTokenTensor)

        let tftt = CFAbsoluteTimeGetCurrent() - promptStartTime
        var nextToken = nextTokenTensor.item(Int.self)

        var generatedTokens = 0
        var generatedText = ""
        let genStartTime = CFAbsoluteTimeGetCurrent()
        let stopTokens: Set<Int> = [1, 106] // <eos>, <turn|>

        while generatedTokens < maxNewTokens && !stopTokens.contains(nextToken) {
            let currentToken = nextToken
            generatedTokens += 1

            // 1. Dispatch step t+1 to Apple Silicon GPU immediately
            let tokenPLE = gatherer.gather(tokenIds: [currentToken])
            let tokenInput = MLXArray([Int32(currentToken)], [1, 1])
            let logits = model(inputs: tokenInput, externalPLE: tokenPLE, caches: caches)
            nextTokenTensor = argMax(logits, axis: -1)
            asyncEval(nextTokenTensor)

            // 2. Concurrently decode text and stream while GPU computes step t+1
            let chunk = tokenizer.decode(tokens: [currentToken])
            generatedText += chunk
            onToken?(chunk)

            if generatedTokens >= maxNewTokens {
                break
            }

            // 3. Resolve next token (GPU has already had time to execute in parallel)
            nextToken = nextTokenTensor.item(Int.self)
        }

        let genElapsed = CFAbsoluteTimeGetCurrent() - genStartTime
        let tokPerSec = genElapsed > 0 ? Double(generatedTokens) / genElapsed : 0.0

        let metrics = PLEGenerationMetrics(
            promptTokens: promptTokens.count,
            generationTokens: generatedTokens,
            promptTokensPerSecond: (tftt > 0) ? Double(promptTokens.count) / tftt : nil,
            generationTokensPerSecond: tokPerSec,
            timeToFirstTokenSeconds: tftt,
            initTimeSeconds: self.initTime
        )

        return PLEGenerationOutput(
            text: generatedText.trimmingCharacters(in: .whitespacesAndNewlines),
            tokensPerSecond: tokPerSec,
            metrics: metrics
        )
    }
}

