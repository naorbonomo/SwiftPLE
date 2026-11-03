# SwiftPLE ⚡️

**SwiftPLE** is a high-performance Apple Silicon (Metal / MLX) Swift engine for running large language models with **Per-Layer Embeddings (PLE)**—specifically **Google Gemma 4 E4B / E2B**—on iOS and macOS with an out-of-core memory architecture.

By memory-mapping the massive ~1.5 GB quantized PLE embedding table outside dirty physical RAM (`mmap` + `MADV_RANDOM`) and dequantizing active token rows directly on the GPU, **SwiftPLE bypasses iOS Jetsam memory kills** (~3.8–4.5 GB limit) while achieving **up to 21 tok/s decode on iPhone 17 Pro**, **26 tok/s on M4 MacBook Air**, and **570+ tok/s prefill**.

> **High-performance out-of-core Apple Silicon (MLX/Metal) engine for Gemma 4 Per-Layer Embeddings on iOS and macOS.**
> Bypasses Jetsam memory crashes with zero-copy `mmap` PLE tables.

---

## Key Highlights

- 🚀 **Hardware Benchmarks (Gemma 4 E4B)**:
  - **M4 MacBook Air**: **26.0 tok/s** decode
  - **iPhone 17 Pro**: **up to 21.0 tok/s** decode (~16.2 tok/s sustained under load) & **578 tok/s** prefill
- 📉 **< 2.5 GB Wired RAM**: Keeps the 1.51 GB PLE table on flash storage via POSIX `mmap`, avoiding Jetsam memory crashes.
- 🧠 **Dynamic Architecture**: Supports Gemma 4 E4B (42 layers), Gemma 4 E2B (35 layers), and custom fine-tunes via `config.json` and `ple_manifest.json`.
- ⚡️ **Metal Hardware Acceleration**: Fused FlashAttention (`MLXFast.scaledDotProductAttention`), native Grouped-Query Attention (GQA), `MLXNN.RoPE` kernels, and 18-layer KV-cache sharing.
- 📦 **Zero-Config Hub Integration**: Point to any split model repository on Hugging Face or a local directory, and SwiftPLE handles download, preparation, and streaming.

---

## Installation via Swift Package Manager

In Xcode:
1. Go to **File → Add Package Dependencies...**
2. Enter the repository URL: `https://github.com/naorbonomo/SwiftPLE.git` (or add local folder).
3. Select your target and click **Add Package**.
4. In your app target's **Signing & Capabilities**, enable **Increased Memory Limit** (`com.apple.developer.kernel.increased-memory-limit`).

Or add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/naorbonomo/SwiftPLE.git", branch: "main")
]
```

---

## Quickstart (5 Lines of Swift)

```swift
import SwiftPLE

// 1. Initialize client pointing to any split PLE model on Hugging Face (or local directory)
let engine = PLEModelClient(
    modelRepo: "NaorBonomo/gemma-4-e4b-it-4bit-ple",
    hfToken: "your_hf_token_if_private" // optional for public models
)

// 2. Download (if needed), mmap the PLE table, and warm up GPU Metal pipelines
try await engine.prepare { progress in
    print("Preparing model: \(Int(progress * 100))%")
}

// 3. Generate text with streaming tokens and real-time telemetry
let result = try await engine.generate(
    prompt: "Explain why memory-mapping embedding tables prevents iOS memory kills.",
    maxNewTokens: 256
) { tokenChunk in
    print(tokenChunk, terminator: "")
}

print("\n\nPrefill: \(result.metrics.promptTokensPerSecond ?? 0) tok/s")
print("Decode:  \(result.metrics.generationTokensPerSecond) tok/s")
print("TTFT:    \(result.metrics.timeToFirstTokenSeconds) s")
```

---

## Example Demo App

A complete iOS and macOS SwiftUI demo app is included in [`Example/SwiftPLEDemo`](Example/SwiftPLEDemo):
- **Zero-copy mmap verification**: Downloads the split PLE model from Hugging Face, initializes the out-of-core memory map, and pre-warms Metal shaders.
- **Real-time telemetry HUD**: Displays prompt tokens/sec, decode tokens/sec, and Time to First Token (TTFT).
- **Cross-Platform**: Runs natively on both macOS (Apple Silicon) and iOS (17.0+).

To run the demo:
```bash
open Example/SwiftPLEDemo/SwiftPLEDemo.xcodeproj
```
Select **My Mac** or an **iOS Simulator / Device**, and press **⌘R**.

---

## How It Works (Zero-Slop Technical Architecture)

### 1. The iOS Memory Ceiling Problem
Gemma 4 E4B in 4-bit quantization requires ~4.0 GB of weights. Loading all 4.0 GB into Metal GPU wired buffers leaves zero headroom for KV caches, activations, and UI buffers, triggering an instant iOS Jetsam crash (`Code 11: Terminated due to memory issue`).

### 2. Per-Layer Embeddings (PLE)
In Gemma 4, every token ID indexes a 256-dimensional vector for each of the 42 transformer layers (10,752 parameters per token). With a 262,144 vocabulary, this table alone is **1.51 GB**.
Because PLE is indexed **purely by integer token IDs before the attention layers run**, `SwiftPLE` leaves `gemma4_ple_table.bin` on flash storage and maps it into virtual memory using POSIX `mmap(..., MAP_SHARED)`.

### 3. Interleaved Row-Major Binary Format
Quantized 4-bit weights, scales, and biases are interleaved token-by-token into contiguous 6,048-byte rows:
```
Offset = tokenID * 6048 bytes
├── Bytes 0..<5376:    1,344 x UInt32 (10,752 packed 4-bit weights)
├── Bytes 5376..<5712: 168   x bfloat16 group scales
└── Bytes 5712..<6048: 168   x bfloat16 group biases
```
During decoding, generating token $t+1$ touches only **one 16 KB virtual memory page**, while the GPU dequantizes the affine 4-bit weights in microseconds using native Apple Silicon Metal kernels.

---

## Coexistence with Other MLX Models (Whisper, Embeddings, etc.)

SwiftPLE is built as a standard Swift package on top of official **`mlx-swift` (0.31.4)** without forks or private patches.

- **Multi-Model Pipelines**: You can run other MLX models (such as `mlx-whisper` for speech-to-text, audio/vision models, or embedding models) in the **exact same app process** alongside SwiftPLE with zero conflicts.
- **Process-Global Memory Settings**: Note that `MLX.Memory.cacheLimit` is process-global across Apple Silicon unified memory. SwiftPLE defaults `cacheLimit` to 64 MB during inference to preserve headroom for iOS Jetsam limits. If your companion MLX models (e.g. Whisper) require higher cache limits, simply adjust `MLX.Memory.cacheLimit` in your app after calling `prepare()`.

---

## Model Conversion Tool (`ple-splitter`)

SwiftPLE includes a built-in Python tool in [`Tools/ple-splitter`](Tools/ple-splitter) to convert any upstream Hugging Face Gemma 4 or Gemma 3n checkpoint into the split zero-copy format (`model.safetensors` + `gemma4_ple_table.bin` + `ple_manifest.json`):

```bash
cd Tools/ple-splitter
pip install -e .
ple-split --model mlx-community/gemma-4-e4b-it-4bit --output-dir ./gemma-4-e4b-it-ple
```

See [`Tools/ple-splitter/README.md`](Tools/ple-splitter/README.md) for full usage, options, and Hugging Face Hub upload instructions.

---

## License

MIT License.

