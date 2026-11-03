# ple-splitter 🔪

**ple-splitter** is the model conversion tool for [**SwiftPLE**](https://github.com/naorbonomo/SwiftPLE). It extracts the massive ~1.5 GB Per-Layer Embedding (PLE) table from Hugging Face Gemma checkpoints into a **zero-copy memory-mapped binary file (`gemma4_ple_table.bin`)**, strips unused weights and multimodal towers from `model.safetensors`, generates `ple_manifest.json`, and verifies 100% bit-for-bit equivalence.

This allows 4B+ parameter models like **Gemma 4 E4B / E2B** and **Gemma 3n** to run on iOS within a **< 2.5 GB wired RAM budget**, completely avoiding iOS Jetsam crashes.

---

## Installation

```bash
cd Tools/ple-splitter
pip install -e .
```

Requires:
- Python $\ge$ 3.9
- `mlx` $\ge$ 0.21.0
- `numpy` $\ge$ 1.24.0
- `huggingface_hub` $\ge$ 0.20.0
- `safetensors` $\ge$ 0.4.0

---

## CLI Usage

### 1. Default Quickstart (Zero-Config)

By default, `ple-split` automatically targets the official **Gemma 4 E4B 4-bit** checkpoint (`mlx-community/gemma-4-e4b-it-4bit`) and splits it into `./gemma-4-e4b-it-ple`:

```bash
ple-split
```

Or customize output directory or model:

```bash
ple-split \
  --model mlx-community/gemma-4-e4b-it-4bit \
  --output-dir ./gemma-4-e4b-it-ple
```

This single command:
1. Downloads `config.json`, tokenizer configs, and `model.safetensors` from Hugging Face.
2. Extracts and interleaves `[packed_uint32_weights | bf16_scales | bf16_biases]` token-by-token into `gemma4_ple_table.bin` (using vectorized NumPy operations in ~1.5 seconds).
3. Strips `embed_tokens_per_layer.*` and optional vision/audio multimodal towers, saving a pruned `model.safetensors` (~2.49 GB).
4. Generates `ple_manifest.json` describing the row offsets, group sizes, and dimensions.
5. Runs an automated **bit-for-bit verification pass**, asserting zero difference (`max_diff == 0.0`) against the original weights.

### 2. Manual Upload to Hugging Face Hub

When you're ready to publish the converted model to Hugging Face, you can push it manually at any time without re-splitting:

#### Option A: Using `ple-upload` (Includes auto-generated Model Card)
```bash
ple-upload \
  --dir ./gemma-4-e4b-it-ple \
  --repo your-hf-username/gemma-4-e4b-it-ple \
  --token $HF_TOKEN
```

#### Option B: Using standard `huggingface-cli`
```bash
huggingface-cli upload your-hf-username/gemma-4-e4b-it-ple ./gemma-4-e4b-it-ple
```

*(Note: automatic pushing during splitting is also supported via `ple-split --push-to-hub <repo-id>`, but is completely optional).*

---

## Output Directory Layout

```text
gemma-4-e4b-it-ple/
├── config.json                     # Model hyperparameters
├── ple_manifest.json               # Row byte layout & quantization metadata
├── tokenizer.json                  # Gemma tokenizer
├── tokenizer_config.json
├── model.safetensors               # Pruned transformer weights (~2.5 GB)
└── gemma4_ple_table.bin            # Interleaved PLE table for mmap (~1.5 GB)
```

---

## SwiftPLE Integration

Once split and hosted on Hugging Face, any iOS app using **SwiftPLE** can stream the model in 3 lines:

```swift
import SwiftPLE

let engine = PLEModelClient(modelRepo: "your-hf-username/gemma-4-e4b-it-ple")
try await engine.prepare()
let response = try await engine.generate(prompt: "Explain how mmap PLE works.")
```

---

## License

MIT License.

