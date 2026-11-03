//
//  PLEGenerationMetrics.swift
//  SwiftPLE
//
//  Telemetry, tokens-per-second, and timing metrics for MLX-PLE sequence generation.
//

import Foundation

public struct PLEGenerationMetrics: Codable, Sendable {
    public let promptTokens: Int
    public let generationTokens: Int
    public let promptTokensPerSecond: Double?
    public let generationTokensPerSecond: Double
    public let timeToFirstTokenSeconds: Double
    public let initTimeSeconds: Double?

    public init(
        promptTokens: Int,
        generationTokens: Int,
        promptTokensPerSecond: Double?,
        generationTokensPerSecond: Double,
        timeToFirstTokenSeconds: Double,
        initTimeSeconds: Double?
    ) {
        self.promptTokens = promptTokens
        self.generationTokens = generationTokens
        self.promptTokensPerSecond = promptTokensPerSecond
        self.generationTokensPerSecond = generationTokensPerSecond
        self.timeToFirstTokenSeconds = timeToFirstTokenSeconds
        self.initTimeSeconds = initTimeSeconds
    }
}

public struct PLEGenerationOutput: Sendable {
    public let text: String
    public let tokensPerSecond: Double
    public let metrics: PLEGenerationMetrics

    public init(text: String, tokensPerSecond: Double, metrics: PLEGenerationMetrics) {
        self.text = text
        self.tokensPerSecond = tokensPerSecond
        self.metrics = metrics
    }
}

