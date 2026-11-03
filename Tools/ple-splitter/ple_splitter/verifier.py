"""
verifier.py
Automated bit-for-bit verification: checks that mmap row gather produces 100% identical
dequantized outputs compared to the original safetensors tensors.
"""

from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import mlx.core as mx

from .inspector import PLEModelSpec


def verify_ple_table(
    bin_path: Path,
    original_weights: Dict[str, Any],
    spec: PLEModelSpec,
    test_token_ids: List[int] = None
) -> bool:
    print(f"[*] Running bit-for-bit verification on {bin_path.name}...")
    if not bin_path.exists():
        raise FileNotFoundError(f"Binary file not found: {bin_path}")

    if test_token_ids is None:
        np.random.seed(42)
        random_ids = np.random.randint(0, spec.vocab_size, size=64).tolist()
        test_token_ids = [0, spec.vocab_size - 1, 1, 2, 106] + random_ids

    # Memory-map the generated binary file
    mmap_data = np.memmap(str(bin_path), dtype=np.uint8, mode="r")

    w_key = f"{spec.key_prefix}embed_tokens_per_layer.weight"
    s_key = f"{spec.key_prefix}embed_tokens_per_layer.scales"
    b_key = f"{spec.key_prefix}embed_tokens_per_layer.biases"

    orig_w = original_weights[w_key]
    orig_s = original_weights[s_key]
    orig_b = original_weights[b_key]

    for t in test_token_ids:
        offset = t * spec.bytes_per_token
        row = mmap_data[offset : offset + spec.bytes_per_token]

        w_bytes = row[0 : spec.bytes_weights]
        s_bytes = row[spec.bytes_weights : spec.bytes_weights + spec.bytes_scales]
        b_bytes = row[spec.bytes_weights + spec.bytes_scales : spec.bytes_per_token]

        gathered_w = np.frombuffer(w_bytes, dtype=np.uint32)
        gathered_s = np.frombuffer(s_bytes, dtype=np.uint16)
        gathered_b = np.frombuffer(b_bytes, dtype=np.uint16)

        # Compare directly with original slices
        ref_w = np.array(orig_w[t])
        ref_s = np.array(orig_s[t].view(mx.uint16))
        ref_b = np.array(orig_b[t].view(mx.uint16))

        assert np.array_equal(gathered_w, ref_w), f"Weight mismatch at token {t}"
        assert np.array_equal(gathered_s, ref_s), f"Scale mismatch at token {t}"
        assert np.array_equal(gathered_b, ref_b), f"Bias mismatch at token {t}"

    print(f"[✓] VERIFIED: 100% bit-for-bit equivalence confirmed across {len(test_token_ids)} test tokens!")
    return True

