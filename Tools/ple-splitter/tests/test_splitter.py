"""
test_splitter.py
Unit tests for ple_splitter packaging and verification.
"""

import unittest
import tempfile
from pathlib import Path
import numpy as np
import mlx.core as mx

from ple_splitter.inspector import PLEModelSpec
from ple_splitter.packer import pack_ple_table
from ple_splitter.verifier import verify_ple_table


class TestPLESplitter(unittest.TestCase):
    def test_spec_dimensions(self):
        spec = PLEModelSpec(vocab_size=1000, num_layers=4, hidden_per_layer=64, group_size=64, bits=4)
        self.assertEqual(spec.total_ple_dim, 256)
        self.assertEqual(spec.num_groups, 4)
        self.assertEqual(spec.num_u32_per_row, 32)
        self.assertEqual(spec.bytes_weights, 128)
        self.assertEqual(spec.bytes_scales, 8)
        self.assertEqual(spec.bytes_biases, 8)
        self.assertEqual(spec.bytes_per_token, 144)

    def test_vectorized_packing_and_verification(self):
        spec = PLEModelSpec(
            vocab_size=20,
            num_layers=4,
            hidden_per_layer=64,
            group_size=64,
            bits=4,
            key_prefix="model."
        )

        np.random.seed(123)
        dummy_w = mx.array(np.random.randint(0, 0xFFFFFFFF, size=(20, spec.num_u32_per_row), dtype=np.uint32))
        dummy_s = mx.array(np.random.randint(0, 0xFFFF, size=(20, spec.num_groups), dtype=np.uint16)).view(mx.bfloat16)
        dummy_b = mx.array(np.random.randint(0, 0xFFFF, size=(20, spec.num_groups), dtype=np.uint16)).view(mx.bfloat16)

        weights = {
            "model.embed_tokens_per_layer.weight": dummy_w,
            "model.embed_tokens_per_layer.scales": dummy_s,
            "model.embed_tokens_per_layer.biases": dummy_b,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            bin_path = Path(tmp_dir) / "test_ple.bin"
            file_size, sha256 = pack_ple_table(weights, spec, bin_path)

            self.assertEqual(file_size, 20 * spec.bytes_per_token)
            self.assertEqual(len(sha256), 64)
            # Verify bit-for-bit equivalence
            verified = verify_ple_table(bin_path, weights, spec, test_token_ids=[0, 5, 19])
            self.assertTrue(verified)


if __name__ == "__main__":
    unittest.main()
