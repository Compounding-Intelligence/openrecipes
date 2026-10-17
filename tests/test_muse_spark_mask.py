# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified for OpenRecipes, 2026-10-17: paths updated for this repository's layout (tests/, data/).
"""muse_spark loss masks (data/muse_spark_mask.py): assistant content and end marker, header as context.

The always-on tests use a small template of the same shape as the released one and a byte-level BPE tokenizer
trained here. ATTUNE_MUSE_TEMPLATE (the checkpoint's chat_template.jinja) runs the released template on that
tokenizer; ATTUNE_MUSE_CHECKPOINT (a checkout with tokenizer.json and chat_template.jinja) runs the real tokenizer.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
_spec = importlib.util.spec_from_file_location("muse_spark_mask", DATA / "muse_spark_mask.py")
muse_spark_mask = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(muse_spark_mask)

SPECIAL = ["<|begin_of_text|>", "<|end_of_text|>", "<|start|>", "<|message|>", "<|eot|>", "<|eom|>"]
# Same shape as the released template: dated system turn, escaped text, one generation span per assistant message.
TEMPLATE = (
    "{{- bos_token -}}"
    "{%- set current_date = current_date | default(strftime_now('%Y-%m-%d') if strftime_now is defined else none) -%}"
    "{%- set reasoning_effort = reasoning_effort | default('xhigh', true) -%}"
    "{%- macro esc(v) -%}{{- v | string | replace('<|', '< |') -}}{%- endmacro -%}"
    "{%- set ns = namespace(sys='', rest=messages) -%}"
    "{%- if messages and messages[0]['role'] == 'system' -%}"
    "{%- set ns.sys = esc(messages[0]['content']) -%}{%- set ns.rest = messages[1:] -%}{%- endif -%}"
    "{{- '<|start|>system<|message|>Current date: ' ~ current_date ~ '.\\n' ~ ns.sys ~ '\\nReasoning: '"
    " ~ reasoning_effort ~ '<|eot|>' -}}"
    "{%- for m in ns.rest -%}"
    "{%- if m['role'] == 'user' -%}{{- '<|start|>user<|message|>' ~ esc(m['content']) ~ '<|eot|>' -}}"
    "{%- elif m['role'] == 'assistant' -%}{%- generation %}"
    "{{- '<|start|>assistant to=user<|message|>' ~ esc(m['content']) ~ '<|eot|>' -}}{%- endgeneration %}"
    "{%- endif -%}{%- endfor -%}"
)
CONVERSATION = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4."}, {"role": "user", "content": "And <|eot|> 3+3, café?"},
                {"role": "assistant", "content": "6, naïve <|start|>user"}]


@pytest.fixture(scope="module")
def tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    corpus = ["Current date: 2026-10-01. Reasoning: xhigh. assistant to=user user system What is 2+2? 4. café"] * 20
    tok.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=400, special_tokens=SPECIAL,
                                                        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<|begin_of_text|>", eos_token="<|end_of_text|>")
    fast.chat_template = TEMPLATE
    return fast


def _targets(tokenizer, tokens, mask):
    """Decoded supervised runs, in order."""
    runs, cur = [], []
    for t, m in zip(tokens, mask, strict=True):
        if m:
            cur.append(t)
        elif cur:
            runs.append(tokenizer.decode(cur))
            cur = []
    return runs + ([tokenizer.decode(cur)] if cur else [])


def test_generation_spans_are_the_targets(tokenizer):
    gen = muse_spark_mask.MuseSparkLossMaskGenerator(tokenizer)
    tokens, mask = gen.get_loss_mask(CONVERSATION)
    text = gen.render(CONVERSATION)
    assert "Current date: 2026-10-01." in text and "< |eot|>" in text and "< |start|>user" in text
    assert tokens == tokenizer(text, add_special_tokens=False)["input_ids"]
    assert _targets(tokenizer, tokens, mask) == ["4.<|eot|>", "6, naïve < |start|>user<|eot|>"]
    assert gen.manifest()["template_kwargs"] == {"current_date": "2026-10-01", "reasoning_effort": "xhigh"}


def test_headers_of_every_assistant_part_are_context(tokenizer):
    """One assistant rendering may hold several parts (reasoning, commentary, a tool call), each with its header."""
    piece = ("<|start|>assistant to=self<|message|>think<|eom|><|start|>assistant to=user<|message|>"
             "<|eot|><|start|>assistant to=f<|message|><atem:function_calls>x</atem:function_calls><|eom|>")
    text = "ctx" + piece
    spans = muse_spark_mask.content_spans(text, 3, len(text))
    assert [text[a:b] for a, b in spans] == ["think<|eom|>", "<|eot|>",
                                             "<atem:function_calls>x</atem:function_calls><|eom|>"]
    for bad in ("<|start|>user<|message|>x<|eot|>", "<|start|>assistant to=user x<|eot|>",
                "<|start|>assistant to=user<|message|>x"):
        with pytest.raises(ValueError):
            muse_spark_mask.content_spans(bad, 0, len(bad))


def test_rejects_tools_unknown_roles_non_text_and_missing_assistant(tokenizer):
    gen = muse_spark_mask.MuseSparkLossMaskGenerator(tokenizer)
    ok = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    for messages, tools in ((ok, [{"name": "f"}]), ([{"role": "tool", "content": "x"}] + ok[1:], None),
                            ([{"role": "user", "content": ["x"]}] + ok[1:], None), (ok[:1], None)):
        with pytest.raises(ValueError):
            gen.get_loss_mask(messages, tools=tools)


def test_unpinned_date_and_templates_without_or_with_narrower_spans_are_refused(tokenizer):
    with pytest.raises(ValueError, match="current_date"):
        muse_spark_mask.MuseSparkLossMaskGenerator(tokenizer, template_kwargs={}).get_loss_mask(CONVERSATION)
    narrow = TEMPLATE.replace("{%- generation %}{{- '<|start|>assistant to=user<|message|>' ~ esc(m['content'])",
                              "{{- '<|start|>assistant to=user<|message|>' -}}{%- generation %}{{- esc(m['content'])")
    assert narrow != TEMPLATE
    for template in (TEMPLATE.replace("{%- generation %}", "").replace("{%- endgeneration %}", ""), narrow):
        tokenizer.chat_template = template
        try:
            with pytest.raises(ValueError):
                muse_spark_mask.MuseSparkLossMaskGenerator(tokenizer).get_loss_mask(CONVERSATION)
        finally:
            tokenizer.chat_template = TEMPLATE


def test_builder_option_records_the_mask_in_the_manifest(tokenizer, tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    tokenizer.save_pretrained(tmp_path / "tok")
    rows = [{"messages": [{"role": "user", "content": "q%d" % i}, {"role": "assistant", "content": "a%d " % i * 9}],
             "source": "s"} for i in range(40)]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "src.parquet")
    monkeypatch.syspath_prepend(str(DATA))
    spec = importlib.util.spec_from_file_location("generate_fixed_sft", DATA / "generate_fixed_sft.py")
    gfs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gfs)
    monkeypatch.setattr(gfs, "validate_slime_loader", lambda *a, **k: {})
    out = tmp_path / "rows" / "train.parquet"
    monkeypatch.setattr(sys, "argv", ["generate_fixed_sft.py", "--input", str(tmp_path / "src.parquet"),
                                      "--output", str(out), "--tokenizer", str(tmp_path / "tok"),
                                      "--loss-mask-type", "muse_spark", "--sequence-length", "128", "--rows", "4",
                                      "--min-token-entropy-bits", "0"])
    gfs.main()
    manifest = json.loads(out.with_suffix(".parquet.manifest.json").read_text())
    assert manifest["tokenizer"]["loss_mask"]["type"] == "muse_spark"
    assert manifest["tokenizer"]["loss_mask"]["template_kwargs"]["current_date"] == "2026-10-01"
    assert manifest["generation"]["rows"] == 4 and manifest["statistics"]["supervised_tokens"] > 0


@pytest.mark.skipif(not os.environ.get("ATTUNE_MUSE_TEMPLATE"), reason="set ATTUNE_MUSE_TEMPLATE to the Muse Spark "
                    "checkpoint's chat_template.jinja")
def test_released_template(tokenizer):
    tokenizer.chat_template = Path(os.environ["ATTUNE_MUSE_TEMPLATE"]).read_text()
    try:
        gen = muse_spark_mask.MuseSparkLossMaskGenerator(tokenizer)
        tokens, mask = gen.get_loss_mask(CONVERSATION)
        text = gen.render(CONVERSATION)
        assert "Current date: 2026-10-01." in text and "Reasoning strength: 512." in text
        assert _targets(tokenizer, tokens, mask) == ["4.<|eot|>", "6, naïve < |start|>user<|eot|>"]
    finally:
        tokenizer.chat_template = TEMPLATE


@pytest.mark.skipif(not os.environ.get("ATTUNE_MUSE_CHECKPOINT"), reason="set ATTUNE_MUSE_CHECKPOINT to a Muse Spark "
                    "checkout with tokenizer.json and chat_template.jinja")
def test_real_tokenizer():
    from transformers import AutoTokenizer

    path = os.environ["ATTUNE_MUSE_CHECKPOINT"]
    muse_spark_mask.register_config()
    tok = AutoTokenizer.from_pretrained(path, use_fast=True)
    gen = muse_spark_mask.MuseSparkLossMaskGenerator(tok, path)
    tokens, mask = gen.get_loss_mask(CONVERSATION)
    assert tok.decode(tokens) == gen.render(CONVERSATION)
    assert tokens[0] == tok.bos_token_id and gen.manifest()["chat_template_file_sha256"]
    assert _targets(tok, tokens, mask) == ["4.<|eot|>", "6, naïve < |start|>user<|eot|>"]
