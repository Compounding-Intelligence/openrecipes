# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified for OpenRecipes, 2026-10-14: paths updated for this repository's layout (tests/, data/, tools/).
"""tools/active_params.py on synthetic safetensors headers: FP4 packing, embeddings, index tables, position embeddings."""

import importlib.util
import json
import struct
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("active_params", ROOT / "tools" / "active_params.py")
active_params = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(active_params)


def _checkpoint(tmp_path, config, shards):
    """A local checkpoint whose shards hold only headers; total_size is the stored bytes the headers describe."""
    weight_map, total = {}, 0
    for shard, tensors in shards.items():
        header = {name: {"dtype": dtype, "shape": shape, "data_offsets": [0, 0]} for name, (dtype, shape) in tensors.items()}
        raw = json.dumps(header).encode()
        (tmp_path / shard).write_bytes(struct.pack("<Q", len(raw)) + raw)
        for name, (dtype, shape) in tensors.items():
            weight_map[name] = shard
            n = 1
            for d in shape:
                n *= d
            total += n * active_params.BYTES[dtype]
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total},
                                                                       "weight_map": weight_map}))
    return str(tmp_path)


def _count(src, capsys):
    active_params.main([src])
    return json.loads(capsys.readouterr().out)


def test_fp4_experts_unpacked_and_non_matmul_tensors_excluded(tmp_path, capsys):
    src = _checkpoint(tmp_path, {"num_hidden_layers": 1, "num_experts_per_tok": 1, "n_routed_experts": 2}, {
        "a.safetensors": {
            "embed.weight": ("BF16", [10, 4]),                            # input embedding
            "layers.0.attn.wq.weight": ("BF16", [4, 4]),                  # 16
            "layers.0.attn.compressor.ape": ("F32", [2, 4]),              # position embedding
            "layers.0.ffn.gate.tid2eid": ("I64", [10, 1]),                # token-to-expert table
            "layers.0.ffn.experts.0.w1.weight": ("I8", [4, 16]),          # FP4: 4 x 32 values, scale in the other shard
            "layers.0.ffn.experts.1.w1.weight": ("I8", [4, 16]),
            "head.weight": ("BF16", [10, 4]),                             # 40
        },
        "b.safetensors": {
            "layers.0.ffn.experts.0.w1.scale": ("F8_E8M0", [4, 1]),
            "layers.0.ffn.experts.1.w1.scale": ("F8_E8M0", [4, 1]),
        },
    })
    out = _count(src, capsys)
    assert out["parts"]["routed_experts"] == 128   # 2 experts x 128 unpacked values x top-1 of 2
    assert out["parts"]["other_matmul"] == 16
    assert out["parts"]["output"] == 40
    assert out["active_params"] == 184


def test_mxfp4_blocks_with_scales_and_expert_biases(tmp_path, capsys):
    src = _checkpoint(tmp_path, {"num_hidden_layers": 1, "experts_per_token": 1, "num_local_experts": 2}, {
        "model.safetensors": {
            "model.layers.0.mlp.experts.gate_up_proj_blocks": ("U8", [2, 4, 3, 16]),   # 32 values per block
            "model.layers.0.mlp.experts.gate_up_proj_scales": ("U8", [2, 4, 3]),
            "model.layers.0.mlp.experts.gate_up_proj_bias": ("BF16", [2, 4]),
            "model.layers.0.mlp.router.weight": ("BF16", [2, 4]),
        },
    })
    out = _count(src, capsys)
    assert out["parts"]["routed_experts"] == 384   # 2 x 4 x 3 x 32 values, top-1 of 2
    assert out["active_params"] == 392


def test_int8_weight_with_scale_counts_one_value_per_byte(tmp_path, capsys):
    src = _checkpoint(tmp_path, {"num_hidden_layers": 1}, {"model.safetensors": {
        "model.layers.0.mlp.up_proj.weight": ("I8", [4, 8]),             # 2 x 8 == 16: only the scale dtype tells
        "model.layers.0.mlp.up_proj.weight_scale": ("BF16", [4, 1]),
    }})
    assert _count(src, capsys)["active_params"] == 32


@pytest.mark.parametrize("tensors, config", [
    ({"layers.0.mlp.table.weight": ("U8", [4, 4])}, {"num_hidden_layers": 1}),
    ({"layers.0.mlp.experts.0.up.weight": ("BF16", [4, 4])}, {"num_hidden_layers": 1}),
])
def test_unknown_integer_tensor_or_missing_expert_count_stops(tmp_path, capsys, tensors, config):
    src = _checkpoint(tmp_path, config, {"model.safetensors": tensors})
    with pytest.raises(SystemExit):
        active_params.main([src])


def test_tied_embed_tokens_is_the_output_projection(tmp_path, capsys):
    src = _checkpoint(tmp_path, {"num_hidden_layers": 1, "tie_word_embeddings": True}, {"model.safetensors": {
        "model.embed_tokens.weight": ("BF16", [10, 4]),
        "model.layers.0.mlp.up.weight": ("BF16", [4, 4]),
        "model.visual.pos_embed.weight": ("BF16", [6, 4]),
    }})
    out = _count(src, capsys)
    assert out["parts"]["output"] == 40
    assert out["active_params"] == 56


def test_audio_tower_and_vision_adapter_are_not_text_matmuls(tmp_path, capsys):
    src = _checkpoint(tmp_path, {"text_config": {"num_hidden_layers": 1}}, {"model.safetensors": {
        "model.language_model.layers.0.self_attn.q_proj.weight": ("BF16", [4, 4]),   # 16
        "model.audio_tower.layers.0.self_attn.q_proj.weight": ("BF16", [8, 4]),
        "model.audio_tower.proj.weight": ("BF16", [4, 8]),
        "model.vision_tower.layers.0.mlp.fc1.weight": ("BF16", [8, 4]),
        "model.vision_adapter.fc1.weight": ("BF16", [8, 4]),
        "model.vision_projection.weight": ("BF16", [4, 8]),
    }})
    out = _count(src, capsys)
    assert out["parts"]["other_matmul"] == 16
    assert out["parts"]["excluded"] == 160
    assert out["active_params"] == 16
