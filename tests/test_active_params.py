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


def test_nemotron_h_backbone_embeddings_is_the_input_embedding(tmp_path, capsys):
    src = _checkpoint(tmp_path, {"num_hidden_layers": 1}, {"model.safetensors": {
        "backbone.embeddings.weight": ("BF16", [10, 4]),
        "backbone.layers.0.mixer.in_proj.weight": ("BF16", [8, 4]),       # 32
        "lm_head.weight": ("BF16", [10, 4]),                              # 40
    }})
    out = _count(src, capsys)
    assert out["parts"]["excluded"] == 40
    assert out["active_params"] == 72


def test_total_size_in_another_unit_passes_when_the_weight_map_matches(tmp_path, capsys):
    src = _checkpoint(tmp_path, {"num_hidden_layers": 1}, {"model.safetensors": {
        "model.layers.0.mlp.up.weight": ("BF16", [4, 4]),
    }})
    index_path = tmp_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["metadata"]["total_size"] = 16                                  # elements, not bytes
    index_path.write_text(json.dumps(index))
    assert _count(src, capsys)["active_params"] == 16
    index["weight_map"]["model.layers.0.mlp.down.weight"] = "model.safetensors"   # a tensor the headers lack
    index_path.write_text(json.dumps(index))
    with pytest.raises(SystemExit):
        active_params.main([src])


def test_gemma4_top_k_experts_and_vision_embedding_projection(tmp_path, capsys):
    config = {"text_config": {"num_hidden_layers": 1, "num_experts": 4, "top_k_experts": 2, "tie_word_embeddings": True}}
    src = _checkpoint(tmp_path, config, {"model.safetensors": {
        "model.language_model.embed_tokens.weight": ("BF16", [10, 4]),              # tied output projection: 40
        "model.language_model.layers.0.experts.gate_up_proj": ("BF16", [4, 8, 4]),  # 128, top-2 of 4: 64
        "model.language_model.layers.0.mlp.up_proj.weight": ("BF16", [8, 4]),       # 32
        "model.language_model.layers.0.router.proj.weight": ("BF16", [4, 4]),       # 16
        "model.embed_vision.embedding_projection.weight": ("BF16", [4, 6]),
    }})
    out = _count(src, capsys)
    assert out["experts_per_token"] == 2
    assert out["parts"]["routed_experts"] == 64
    assert out["parts"]["other_matmul"] == 48
    assert out["active_params"] == 152


def test_inkling_audio_and_vision_towers_and_mtp_layers_excluded(tmp_path, capsys):
    config = {"text_config": {"num_hidden_layers": 1, "n_routed_experts": 4, "num_experts_per_tok": 2}}
    src = _checkpoint(tmp_path, config, {"model.safetensors": {
        "model.llm.embed.weight": ("BF16", [10, 4]),
        "model.llm.layers.0.mlp.experts.w13_weight": ("BF16", [4, 8, 4]),               # 128, top-2 of 4: 64
        "model.llm.layers.0.mlp.shared_experts.shared_w13_weight": ("BF16", [8, 4]),    # 32
        "model.llm.layers.0.attn_sconv.weight": ("BF16", [4, 1, 4]),                    # 16
        "model.llm.unembed.weight": ("BF16", [10, 4]),                                  # 40
        "model.audio.encoder.weight": ("BF16", [6, 4]),
        "model.visual.layers.linear_0.weight": ("BF16", [6, 4]),
        "model.mtp.layers.0.input_proj.weight": ("BF16", [4, 8]),
    }})
    out = _count(src, capsys)
    assert out["parts"]["routed_experts"] == 64
    assert out["parts"]["other_matmul"] == 88
    assert out["active_params"] == 152


def test_kimi_pack_quantized_int4_experts_and_vision_projection(tmp_path, capsys):
    quant = {"format": "pack-quantized", "config_groups": {"group_0": {"weights": {"num_bits": 4, "group_size": 32}}}}
    config = {"text_config": {"num_hidden_layers": 1, "n_routed_experts": 4, "num_experts_per_tok": 2,
                              "quantization_config": quant}}
    src = _checkpoint(tmp_path, config, {"model.safetensors": {
        "language_model.model.embed_tokens.weight": ("BF16", [10, 4]),
        "language_model.model.layers.0.mlp.experts.0.up_proj.weight_packed": ("I32", [8, 4]),   # 8 x 32 values
        "language_model.model.layers.0.mlp.experts.0.up_proj.weight_scale": ("BF16", [8, 1]),
        "language_model.model.layers.0.mlp.experts.0.up_proj.weight_shape": ("I32", [2]),
        "language_model.model.layers.0.mlp.shared_experts.up_proj.weight": ("BF16", [8, 4]),    # 32
        "language_model.model.layers.0.self_attn.o_proj.weight": ("BF16", [4, 4]),              # 16
        "language_model.lm_head.weight": ("BF16", [10, 4]),                                     # 40
        "vision_tower.encoder.blocks.0.wqkv.weight": ("BF16", [12, 4]),
        "mm_projector.proj.0.weight": ("BF16", [16, 16]),
    }})
    out = _count(src, capsys)
    assert out["parts"]["routed_experts"] == 128   # 256 unpacked values x top-2 of 4
    assert out["parts"]["other_matmul"] == 48
    assert out["parts"]["output"] == 40
    assert out["active_params"] == 216


def test_pack_quantized_weight_without_a_pack_quantized_config_stops(tmp_path, capsys):
    src = _checkpoint(tmp_path, {"num_hidden_layers": 1}, {"model.safetensors": {
        "layers.0.mlp.up_proj.weight_packed": ("I32", [8, 4]),
        "layers.0.mlp.up_proj.weight_scale": ("BF16", [8, 1]),
    }})
    with pytest.raises(SystemExit):
        active_params.main([src])
