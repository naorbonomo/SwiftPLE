"""
packer.py
High-speed vectorized row-major binary packer for Per-Layer Embeddings.
Interleaves [packed_uint32_weights | bf16_scales | bf16_biases] token-by-token for zero-copy mmap.
"""

import os
import hashlib
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import mlx.core as mx

from .inspector import PLEModelSpec


def pack_ple_table(weights: Dict[str, Any], spec: PLEModelSpec, dest_path: Path) -> Tuple[int, str]:
    w_key = f"{spec.key_prefix}embed_tokens_per_layer.weight"
    s_key = f"{spec.key_prefix}embed_tokens_per_layer.scales"
    b_key = f"{spec.key_prefix}embed_tokens_per_layer.biases"

    if w_key not in weights or s_key not in weights or b_key not in weights:
        raise KeyError(f"Missing one or more required PLE keys: {w_key}, {s_key}, {b_key}")

    print(f"[*] Extracting PLE tensors ({spec.vocab_size} tokens, {spec.total_ple_dim} dims/token)...")
    ple_w = np.array(weights[w_key])
    ple_s = np.array(weights[s_key].view(mx.uint16))
    ple_b = np.array(weights[b_key].view(mx.uint16))

    assert ple_w.shape == (spec.vocab_size, spec.num_u32_per_row), f"Unexpected weights shape: {ple_w.shape}"
    assert ple_s.shape == (spec.vocab_size, spec.num_groups), f"Unexpected scales shape: {ple_s.shape}"
    assert ple_b.shape == (spec.vocab_size, spec.num_groups), f"Unexpected biases shape: {ple_b.shape}"

    print(f"[*] Interleaving rows into contiguous array (vectorized)...")
    # Cast to uint8 byte views
    w_bytes = ple_w.view(np.uint8).reshape(spec.vocab_size, spec.bytes_weights)
    s_bytes = ple_s.view(np.uint8).reshape(spec.vocab_size, spec.bytes_scales)
    b_bytes = ple_b.view(np.uint8).reshape(spec.vocab_size, spec.bytes_biases)

    # Interleave side-by-side into a single contiguous [vocab_size, bytes_per_token] array
    packed = np.hstack([w_bytes, s_bytes, b_bytes])
    assert packed.shape == (spec.vocab_size, spec.bytes_per_token)

    print(f"[*] Calculating SHA256 checksum...")
    sha256_hash = hashlib.sha256(packed).hexdigest()

    print(f"[*] Writing {dest_path.name} ({packed.nbytes / (1024 * 1024):.1f} MB)...")
    packed.tofile(str(dest_path))

    file_size = os.path.getsize(dest_path)
    print(f"[✓] Successfully wrote {dest_path.name}: {file_size / (1024 * 1024):.2f} MB")
    print(f"    SHA256: {sha256_hash}")
    return file_size, sha256_hash

