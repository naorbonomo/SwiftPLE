//
//  PLEHubDownloader.swift
//  SwiftPLE
//
//  Downloads split PLE model weights, mmap binary tables, and tokenizers
//  from Hugging Face repositories or direct HTTP/HTTPS endpoints.
//

import Foundation

public enum PLEHubDownloader {
    public struct FileEntry: Sendable {
        public let name: String
        public let isRequired: Bool
        public let estimatedBytes: Int64

        public init(name: String, isRequired: Bool = true, estimatedBytes: Int64 = 1_000_000) {
            self.name = name
            self.isRequired = isRequired
            self.estimatedBytes = estimatedBytes
        }
    }

    public static let standardFiles: [FileEntry] = [
        FileEntry(name: "config.json", isRequired: true, estimatedBytes: 7_000),
        FileEntry(name: "ple_manifest.json", isRequired: false, estimatedBytes: 2_000),
        FileEntry(name: "tokenizer_config.json", isRequired: true, estimatedBytes: 3_000),
        FileEntry(name: "tokenizer.json", isRequired: true, estimatedBytes: 31_000_000),
        FileEntry(name: "model.safetensors", isRequired: true, estimatedBytes: 2_492_000_000),
        FileEntry(name: "gemma4_ple_table.bin", isRequired: false, estimatedBytes: 1_585_446_912)
    ]

    public static func defaultDestination(for repoOrName: String) -> URL {
        let safeName = repoOrName.replacingOccurrences(of: "/", with: "_")
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        return appSupport.appendingPathComponent("SwiftPLE/models/\(safeName)", isDirectory: true)
    }

    public static func resolveDownloadURL(for repoOrURL: String, file: String) -> URL? {
        if repoOrURL.hasPrefix("http://") || repoOrURL.hasPrefix("https://") {
            let clean = repoOrURL.hasSuffix("/") ? String(repoOrURL.dropLast()) : repoOrURL
            return URL(string: "\(clean)/\(file)")
        }
        // Hugging Face Hub CDN resolve URL
        return URL(string: "https://huggingface.co/\(repoOrURL)/resolve/main/\(file)")
    }

    public static func download(
        from repoOrURL: String,
        hfToken: String? = nil,
        destinationDir: URL? = nil,
        onProgress: (@Sendable (Double) -> Void)? = nil
    ) async throws -> URL {
        let dest = destinationDir ?? defaultDestination(for: repoOrURL)
        try FileManager.default.createDirectory(at: dest, withIntermediateDirectories: true)

        let totalEstimatedBytes: Double = 4_035_000_000.0
        var completedBytes: Int64 = 0

        for file in standardFiles {
            let destURL = dest.appendingPathComponent(file.name)
            if FileManager.default.fileExists(atPath: destURL.path),
               let attrs = try? FileManager.default.attributesOfItem(atPath: destURL.path),
               let size = attrs[.size] as? Int64, size > 1000 {
                completedBytes += file.estimatedBytes
                onProgress?(min(Double(completedBytes) / totalEstimatedBytes, 1.0))
                continue
            }

            guard let remoteURL = resolveDownloadURL(for: repoOrURL, file: file.name) else { continue }
            var request = URLRequest(url: remoteURL)
            if let token = hfToken, !token.isEmpty {
                request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
            }

            do {
                let (tempURL, response) = try await URLSession.shared.download(for: request)
                guard let http = response as? HTTPURLResponse else { continue }
                if http.statusCode == 200 {
                    if FileManager.default.fileExists(atPath: destURL.path) {
                        try? FileManager.default.removeItem(at: destURL)
                    }
                    try FileManager.default.moveItem(at: tempURL, to: destURL)
                    completedBytes += file.estimatedBytes
                } else if file.isRequired {
                    throw NSError(
                        domain: "PLEHubDownloader",
                        code: http.statusCode,
                        userInfo: [NSLocalizedDescriptionKey: "Failed to download \(file.name): HTTP \(http.statusCode)"]
                    )
                }
            } catch {
                if file.isRequired { throw error }
            }
            onProgress?(min(Double(completedBytes) / totalEstimatedBytes, 1.0))
        }

        onProgress?(1.0)
        return dest
    }
}

