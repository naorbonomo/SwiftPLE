"""
cli.py
CLI entry point for ple-splitter.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download
import mlx.core as mx

from .inspector import inspect_model_directory
from .packer import pack_ple_table
from .pruner import prune_and_save_safetensors
from .verifier import verify_ple_table
from .uploader import create_model_card, upload_to_hub

DEFAULT_MODEL = "mlx-community/gemma-4-e4b-it-4bit"
DEFAULT_OUTPUT_DIR = "./gemma-4-e4b-it-ple"
DEFAULT_BIN_NAME = "gemma4_ple_table.bin"

METADATA_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "chat_template.jinja",
    "special_tokens_map.json"
]


def download_hf_metadata(repo_id: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Downloading tokenizer and config files from {repo_id}...")
    for f in METADATA_FILES:
        try:
            hf_hub_download(repo_id, f, local_dir=str(output_dir))
            print(f"  Downloaded: {f}")
        except Exception:
            pass


def run_split(args):
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_input = args.model

    local_model_path = Path(model_input)
    is_local = local_model_path.exists() and local_model_path.is_dir()

    if is_local:
        print(f"[*] Reading local model from {local_model_path}...")
        source_safetensors = local_model_path / "model.safetensors"
        if not source_safetensors.exists():
            raise FileNotFoundError(f"model.safetensors not found in {local_model_path}")
        for f in METADATA_FILES:
            src = local_model_path / f
            if src.exists():
                import shutil
                shutil.copy2(src, output_dir / f)
    else:
        print(f"[*] Fetching remote checkpoint from Hugging Face: {model_input}...")
        download_hf_metadata(model_input, output_dir)
        print(f"[*] Resolving / downloading model.safetensors from {model_input}...")
        cached_file = hf_hub_download(model_input, "model.safetensors")
        source_safetensors = Path(cached_file)

    print(f"[*] Loading safetensors with MLX from {source_safetensors}...")
    weights = mx.load(str(source_safetensors))

    spec, _ = inspect_model_directory(output_dir, weights)
    print(f"[*] Detected Architecture: {spec.num_layers} layers, vocab {spec.vocab_size}, PLE dim {spec.total_ple_dim}")

    bin_path = output_dir / args.bin_name
    bin_size, sha256_hash = pack_ple_table(weights, spec, bin_path)

    # Save manifest
    manifest_dict = spec.to_manifest_dict(
        bin_filename=args.bin_name,
        bin_size_bytes=bin_size,
        source_repo=model_input if not is_local else None,
        ple_sha256=sha256_hash
    )
    manifest_path = output_dir / "ple_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_dict, f, indent=2)
    print(f"[✓] Saved manifest: {manifest_path}")

    # Prune safetensors
    target_safetensors = output_dir / "model.safetensors"
    if source_safetensors.resolve() == target_safetensors.resolve():
        temp_safetensors = output_dir / "model_stripped.safetensors"
        prune_and_save_safetensors(weights, temp_safetensors, keep_multimodal=args.keep_multimodal)
        del weights
        import gc
        gc.collect()
        os.remove(target_safetensors)
        os.rename(str(temp_safetensors), str(target_safetensors))
    else:
        prune_and_save_safetensors(weights, target_safetensors, keep_multimodal=args.keep_multimodal)
        del weights
        import gc
        gc.collect()

    # Verify bit-for-bit equivalence
    if not args.skip_verification:
        source_weights = mx.load(str(source_safetensors))
        verify_ple_table(bin_path, source_weights, spec)
        del source_weights
        import gc
        gc.collect()

    print("\n" + "=" * 60)
    print("PLE SPLIT SUCCESSFUL")
    print(f"Output directory: {output_dir}")
    print(f"  - Core weights:  {os.path.getsize(target_safetensors) / (1024 * 1024):.1f} MB")
    print(f"  - PLE table:     {bin_size / (1024 * 1024):.1f} MB")
    print(f"  - Checksum:      {sha256_hash}")
    print("=" * 60)

    # Optional Push to Hub
    if args.push_to_hub:
        create_model_card(output_dir, model_input, args.push_to_hub)
        upload_to_hub(output_dir, args.push_to_hub, token=args.token, private=args.private)


def main():
    parser = argparse.ArgumentParser(
        prog="ple-split",
        description="Split Per-Layer Embedding (PLE) models into out-of-core mmap binary tables for SwiftPLE."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Hugging Face repo ID or local model directory (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Destination directory for split model (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--bin-name",
        type=str,
        default=DEFAULT_BIN_NAME,
        help=f"Output PLE binary filename (default: {DEFAULT_BIN_NAME})"
    )
    parser.add_argument("--push-to-hub", type=str, default=None, help="Hugging Face repository ID to publish to")
    parser.add_argument("--private", action="store_true", help="Create the Hugging Face repository as private")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face auth token")
    parser.add_argument("--skip-verification", action="store_true", help="Skip bit-for-bit verification pass")
    parser.add_argument("--keep-multimodal", action="store_true", help="Retain vision/audio multimodal towers in safetensors")

    args = parser.parse_args()
    run_split(args)


if __name__ == "__main__":
    main()
