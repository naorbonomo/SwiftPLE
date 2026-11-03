"""
inspector.py
Inspects model configuration and safetensors weights to detect PLE tensor dimensions and quantization schemes.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple


class PLEModelSpec:
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        hidden_per_layer: int,
        group_size: int = 64,
        bits: int = 4,
        key_prefix: str = "language_model.model.",
    ):
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.hidden_per_layer = hidden_per_layer
        self.group_size = group_size
        self.bits = bits
        self.key_prefix = key_prefix

        self.total_ple_dim = num_layers * hidden_per_layer
        self.num_groups = self.total_ple_dim // group_size
        self.num_u32_per_row = self.total_ple_dim // (32 // bits)

        self.bytes_weights = self.num_u32_per_row * 4
        self.bytes_scales = self.num_groups * 2  # bfloat16
        self.bytes_biases = self.num_groups * 2  # bfloat16
        self.bytes_per_token = self.bytes_weights + self.bytes_scales + self.bytes_biases

    def to_manifest_dict(
        self,
        bin_filename: str,
        bin_size_bytes: int,
        source_repo: str = None,
        ple_sha256: str = None
    ) -> Dict[str, Any]:
        return {
            "format_version": 1,
            "source_repo": source_repo,
            "ple_binary_file": bin_filename,
            "ple_binary_size_bytes": bin_size_bytes,
            "ple_sha256": ple_sha256,
            "vocab_size": self.vocab_size,
            "num_layers": self.num_layers,
            "ple_dim_per_layer": self.hidden_per_layer,
            "total_ple_dim": self.total_ple_dim,
            "quantization": {
                "bits": self.bits,
                "group_size": self.group_size,
                "scales_dtype": "bfloat16"
            },
            "row_layout": {
                "bytes_per_token": self.bytes_per_token,
                "weights_offset": 0,
                "weights_bytes": self.bytes_weights,
                "scales_offset": self.bytes_weights,
                "scales_bytes": self.bytes_scales,
                "biases_offset": self.bytes_weights + self.bytes_scales,
                "biases_bytes": self.bytes_biases
            }
        }


def inspect_model_directory(directory: Path, weights: Dict[str, Any] = None) -> Tuple[PLEModelSpec, str]:
    config_file = directory / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"config.json not found in {directory}")

    with open(config_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if "text_config" in cfg and isinstance(cfg["text_config"], dict):
        text_cfg = cfg["text_config"]
    else:
        text_cfg = cfg

    vocab_size = text_cfg.get("vocab_size", 262144)
    num_layers = text_cfg.get("num_hidden_layers", 42)
    hidden_per_layer = text_cfg.get("hidden_size_per_layer_input", 256)

    # Detect prefix from weights keys if provided
    key_prefix = "language_model.model."
    if weights:
        for k in weights.keys():
            if "embed_tokens_per_layer.weight" in k:
                key_prefix = k.rsplit("embed_tokens_per_layer.weight", 1)[0]
                break

    spec = PLEModelSpec(
        vocab_size=vocab_size,
        num_layers=num_layers,
        hidden_per_layer=hidden_per_layer,
        group_size=64,
        bits=4,
        key_prefix=key_prefix
    )
    return spec, key_prefix

