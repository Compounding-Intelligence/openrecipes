# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified for OpenRecipes, 2026-10-14: the rollout-adapter case of ATTUNE's test_make_run_yaml_tokens.py, on data/.
"""data/pretokenized_sft_rollout.py: rows of exactly $AUTOTUNE_SEQ_LEN tokens, global dataset required."""
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_rollout_adapter_honours_seq_len(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "data"))
    import pretokenized_sft_rollout as adapter

    def buffer(n):
        rows = [types.SimpleNamespace(metadata={"tokens": list(range(n)), "loss_mask": [1] * (n - 1)}) for _ in range(2)]
        return types.SimpleNamespace(get_samples=lambda k: [[r] for r in rows[:k]])

    args = types.SimpleNamespace(rollout_batch_size=2, rollout_global_dataset=True)
    monkeypatch.setenv("AUTOTUNE_SEQ_LEN", "16")
    out = adapter.generate_rollout(args, 0, buffer(16))
    assert [len(s.tokens) for (s,) in out] == [16, 16] and out[0][0].response_length == 15
    with pytest.raises(ValueError, match="Expected 16 tokens"):
        adapter.generate_rollout(args, 0, buffer(8192))
    monkeypatch.delenv("AUTOTUNE_SEQ_LEN")
    assert len(adapter.generate_rollout(args, 0, buffer(8192))[0][0].tokens) == 8192
    with pytest.raises(ValueError, match="disable-rollout-global-dataset"):
        adapter.generate_rollout(types.SimpleNamespace(rollout_batch_size=2, rollout_global_dataset=False), 0, buffer(16))
