"""
pruner.py
Strips externalized PLE tables and multimodal towers to produce lean, GPU-wired safetensors checkpoints.
"""

import os
from pathlib import Path
from typing import Dict, Any
import mlx.core as mx


def prune_and_save_safetensors(
    weights: Dict[str, Any],
    dest_path: Path,
    keep_multimodal: bool = False
) -> int:
    print(f"[*] Pruning weights for iOS deployment -> {dest_path.name}...")
    stripped = {}
    pruned_ple = 0
    pruned_multimodal = 0

    multimodal_keywords = (
        "vision_tower",
        "audio_tower",
        "multi_modal_projector",
        "embed_audio",
        "embed_vision",
    )

    for k, v in weights.items():
        if "embed_tokens_per_layer" in k:
            pruned_ple += 1
            continue
        if not keep_multimodal and any(m in k for m in multimodal_keywords):
            pruned_multimodal += 1
            continue
        stripped[k] = v

    print(f"  Removed {pruned_ple} PLE tensors and {pruned_multimodal} multimodal tensors.")
    print(f"  Saving {len(stripped)} active transformer tensors...")

    mx.save_safetensors(str(dest_path), stripped)
    file_size = os.path.getsize(dest_path)
    print(f"[✓] Saved pruned weights: {file_size / (1024 * 1024):.1f} MB")
    return file_size

