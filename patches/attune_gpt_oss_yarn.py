# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPT-OSS layer spec with HF-matching YaRN RoPE, for slime raw mode.

slime's GPT-OSS script builds plain RoPE with Megatron's Llama-3 scaling (`--use-rope-scaling`), and slime's
model provider does not forward `--rope-scaling-factor` (model_provider.py:202-217), so GPTModel uses its
default factor 8 with original length 8192 and no attention factor. HF GPT-OSS uses YaRN (config.json
`rope_scaling`). Megatron's GPTModel supports YaRN (gpt_model.py:178-197; attention mscale at attention.py:1106)
but argparse does not expose it for GPT models, so this spec sets the config fields and the embedding type
before slime constructs GPTModel (slime calls the spec at model_provider.py:150-160, before building kwargs).

Usage: --spec attune_gpt_oss_yarn get_gpt_oss_yarn_spec   (module on PYTHONPATH, e.g. /scripts)
Keep --no-rope-fusion: the fused THD RoPE path drops mscale (rope_utils.py:286-290).
"""

import json
import os

from slime_plugins.models.gpt_oss import get_gpt_oss_spec


def _hf_rope_scaling(args):
    with open(os.path.join(args.hf_checkpoint, "config.json")) as f:
        cfg = json.load(f)
    rs = cfg.get("rope_scaling") or cfg.get("rope_parameters") or {}
    assert rs.get("rope_type", rs.get("type")) == "yarn", f"expected yarn rope_scaling in HF config, got {rs}"
    assert float(cfg.get("rope_theta", rs.get("rope_theta", 0))) == float(args.rotary_base), "rotary_base != rope_theta"
    return rs


def get_gpt_oss_yarn_spec(args, config, vp_stage):
    rs = _hf_rope_scaling(args)
    config.yarn_rotary_scaling_factor = float(rs["factor"])
    config.yarn_original_max_position_embeddings = int(rs["original_max_position_embeddings"])
    config.yarn_beta_fast = float(rs.get("beta_fast", 32.0))
    config.yarn_beta_slow = float(rs.get("beta_slow", 1.0))
    # HF `truncate: false` keeps the correction range unrounded.
    config.yarn_correction_range_round_to_int = bool(rs.get("truncate", True))
    # None/None -> attention factor 0.1*ln(factor)+1, as HF computes when attention_factor/mscale are absent.
    config.yarn_mscale = rs.get("mscale")
    config.yarn_mscale_all_dim = rs.get("mscale_all_dim")
    assert not config.apply_rope_fusion, "use --no-rope-fusion: fused THD RoPE ignores the YaRN attention factor"
    args.position_embedding_type = "yarn"
    return get_gpt_oss_spec(args, config, vp_stage)
