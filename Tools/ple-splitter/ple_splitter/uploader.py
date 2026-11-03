"""
uploader.py
Publishes split model checkpoints and ple_manifest.json to Hugging Face Hub.
"""

from pathlib import Path
from huggingface_hub import HfApi


def create_model_card(output_dir: Path, source_model: str, repo_id: str):
    card_path = output_dir / "README.md"
    content = f"""---
library_name: swift-ple
tags:
- swift
- mlx
- gemma
- on-device
- apple-silicon
- ios
---

# {repo_id}

This model is an **out-of-core Per-Layer Embedding (PLE) split** of [`{source_model}`](https://huggingface.co/{source_model}), formatted for zero-copy memory-mapped inference on iOS / macOS via [**SwiftPLE**](https://github.com/swift-ple/SwiftPLE).

## Architecture & Memory Layout
- **`model.safetensors`**: Core transformer weights (~2.5 GB) loaded into GPU memory.
- **`gemma4_ple_table.bin`**: Per-Layer Embedding table (~1.5 GB) memory-mapped out-of-core via POSIX `mmap` (`MADV_RANDOM`), keeping dirty RAM below iOS Jetsam limits.
- **`ple_manifest.json`**: Self-describing byte offset and quantization schema.

## Usage in Swift

```swift
import SwiftPLE

let engine = PLEModelClient(modelRepo: "{repo_id}")
try await engine.prepare()

let response = try await engine.generate(prompt: "Hello world!")
print(response.text)
```
"""
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[✓] Generated model card at {card_path}")


def upload_to_hub(output_dir: Path, repo_id: str, token: str = None, private: bool = False):
    print(f"[*] Uploading converted model from {output_dir} to Hugging Face: {repo_id} (private={private})...")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        repo_type="model"
    )
    print(f"[✓] Successfully pushed model to https://huggingface.co/{repo_id}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        prog="ple-upload",
        description="Upload an existing split PLE model directory to Hugging Face Hub."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="./gemma-4-e4b-it-ple",
        help="Directory containing the split model"
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="Target Hugging Face repo ID (e.g. username/gemma-4-e4b-it-ple)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repository as private"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face auth token (or uses logged-in huggingface-cli session)"
    )
    parser.add_argument(
        "--source-model",
        type=str,
        default="mlx-community/gemma-4-e4b-it-4bit",
        help="Source model name for model card"
    )

    args = parser.parse_args()
    model_dir = Path(args.dir).resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    create_model_card(model_dir, args.source_model, args.repo)
    upload_to_hub(model_dir, args.repo, token=args.token, private=args.private)


if __name__ == "__main__":
    main()
