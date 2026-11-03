//
//  ContentView.swift
//  SwiftPLEDemo
//
//  Main SwiftUI view providing interactive model preparation,
//  prompt execution, token streaming, and real-time generation metrics.
//

import SwiftUI
import SwiftPLE

public struct ContentView: View {
    @StateObject private var vm = DemoViewModel()
    // Uncomment if using a private model repository requiring an HF token:
    // @State private var isTokenVisible = false

    public init() {}

    public var body: some View {
        NavigationStack {
            Form {
                modelConfigSection
                promptSection
                metricsSection
                outputSection
            }
            .formStyle(.grouped)
            .navigationTitle("SwiftPLE Demo")
        }
    }

    private var modelConfigSection: some View {
        Section("Hugging Face Model") {
            TextField("Repository", text: $vm.modelRepo)
                .autocorrectionDisabled()

            // Token input (commented out: public repo requires no authentication)
            /*
            HStack {
                if isTokenVisible {
                    TextField("HF Token (hf_...)", text: $vm.hfToken)
                } else {
                    SecureField("HF Token (hf_...)", text: $vm.hfToken)
                }
                Button {
                    isTokenVisible.toggle()
                } label: {
                    Image(systemName: isTokenVisible ? "eye.slash" : "eye")
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            }
            */

            if vm.isPreparing {
                VStack(alignment: .leading, spacing: 6) {
                    ProgressView(value: vm.downloadProgress, total: 1.0)
                    Text(vm.statusMessage)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(.vertical, 4)
            } else {
                HStack {
                    Button(action: { vm.prepareModel() }) {
                        Label(vm.isModelReady ? "Reload Model" : "Download & Prepare",
                              systemImage: vm.isModelReady ? "arrow.clockwise" : "arrow.down.circle")
                    }
                    .disabled(vm.isPreparing || vm.isGenerating)

                    if vm.isModelReady {
                        Spacer()
                        Button(role: .destructive, action: { vm.unload() }) {
                            Text("Unload")
                        }
                    }
                }
            }

            if let error = vm.errorMessage {
                Text(error)
                    .font(.footnote)
                    .foregroundStyle(.red)
            }
        }
    }

    private var promptSection: some View {
        Section("Prompt") {
            TextEditor(text: $vm.prompt)
                .frame(minHeight: 70)
                .font(.body)

            Button(action: { vm.generate() }) {
                HStack {
                    Spacer()
                    if vm.isGenerating {
                        ProgressView()
                            .controlSize(.small)
                            .padding(.trailing, 4)
                        Text("Generating...")
                    } else {
                        Image(systemName: "sparkles")
                        Text("Generate")
                    }
                    Spacer()
                }
            }
            .buttonStyle(.borderedProminent)
            .disabled(!vm.isModelReady || vm.isGenerating || vm.prompt.isEmpty)
        }
    }

    @ViewBuilder
    private var metricsSection: some View {
        if let metrics = vm.metrics {
            Section("Performance Metrics") {
                HStack {
                    metricBadge(
                        title: "Decode",
                        value: String(format: "%.1f t/s", metrics.generationTokensPerSecond),
                        icon: "bolt.fill"
                    )
                    Spacer()
                    metricBadge(
                        title: "TTFT",
                        value: String(format: "%.2fs", metrics.timeToFirstTokenSeconds),
                        icon: "clock"
                    )
                    Spacer()
                    metricBadge(
                        title: "Tokens",
                        value: "\(metrics.generationTokens)",
                        icon: "number"
                    )
                }
                .padding(.vertical, 2)
            }
        }
    }

    private var outputSection: some View {
        Section("Output") {
            if vm.outputText.isEmpty && !vm.isGenerating {
                Text("Generated response will stream here...")
                    .font(.body)
                    .foregroundStyle(.tertiary)
                    .frame(minHeight: 100, alignment: .topLeading)
            } else {
                Text(vm.outputText)
                    .font(.system(.body, design: .rounded))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
                    .padding(.vertical, 4)
            }
        }
    }

    private func metricBadge(title: String, value: String, icon: String) -> some View {
        VStack(spacing: 3) {
            Label(title, systemImage: icon)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.headline)
                .foregroundStyle(.primary)
        }
    }
}
