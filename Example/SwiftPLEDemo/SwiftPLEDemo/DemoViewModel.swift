//
//  DemoViewModel.swift
//  SwiftPLEDemo
//
//  Observable view model coordinating SwiftPLE model downloading,
//  zero-copy mmap initialization, and streaming token generation.
//

import Combine
import Foundation
import SwiftPLE
import SwiftUI

@MainActor
public final class DemoViewModel: ObservableObject {
    @Published public var modelRepo: String {
        didSet { UserDefaults.standard.set(modelRepo, forKey: "demo_model_repo") }
    }
    // HF Token is commented out since the default model repository is public:
    // @Published public var hfToken: String = ""
    @Published public var prompt: String = "Explain the advantage of zero-copy memory mapping for on-device AI in three sentences."
    @Published public var outputText: String = ""
    @Published public var statusMessage: String = "Model not loaded"
    @Published public var downloadProgress: Double = 0.0
    @Published public var isPreparing: Bool = false
    @Published public var isModelReady: Bool = false
    @Published public var isGenerating: Bool = false
    @Published public var metrics: PLEGenerationMetrics?
    @Published public var errorMessage: String?

    private var client: PLEModelClient?

    public init() {
        self.modelRepo = UserDefaults.standard.string(forKey: "demo_model_repo")
            ?? "NaorBonomo/gemma-4-e4b-it-4bit-ple"
    }

    public func prepareModel() {
        guard !isPreparing else { return }
        isPreparing = true
        errorMessage = nil
        statusMessage = "Connecting to Hugging Face..."
        downloadProgress = 0.0

        let repo = modelRepo.trimmingCharacters(in: .whitespacesAndNewlines)
        // Public model repo requires no auth token; pass token if private
        let clientInstance = PLEModelClient(
            modelRepo: repo,
            hfToken: nil
        )
        self.client = clientInstance

        Task {
            do {
                try await clientInstance.prepare { [weak self] progress in
                    Task { @MainActor in
                        self?.downloadProgress = progress
                        self?.statusMessage = String(format: "Preparing model assets... %.0f%%", progress * 100)
                    }
                }
                self.isModelReady = true
                self.isPreparing = false
                self.statusMessage = "Ready for generation"
            } catch {
                self.isModelReady = false
                self.isPreparing = false
                self.errorMessage = error.localizedDescription
                self.statusMessage = "Error during setup"
            }
        }
    }

    public func generate() {
        guard let client = client, isModelReady, !isGenerating else { return }
        isGenerating = true
        outputText = ""
        errorMessage = nil
        metrics = nil
        statusMessage = "Generating..."

        let currentPrompt = prompt
        Task {
            do {
                let result = try await client.generate(
                    prompt: currentPrompt,
                    maxNewTokens: 256
                ) { [weak self] chunk in
                    Task { @MainActor in
                        self?.outputText += chunk
                    }
                }
                self.metrics = result.metrics
                self.isGenerating = false
                self.statusMessage = String(format: "Completed at %.1f tokens/sec", result.tokensPerSecond)
            } catch {
                self.isGenerating = false
                self.errorMessage = error.localizedDescription
                self.statusMessage = "Generation failed"
            }
        }
    }

    public func unload() {
        Task {
            await client?.unload()
            client = nil
            isModelReady = false
            statusMessage = "Model unloaded"
            outputText = ""
            metrics = nil
        }
    }
}
