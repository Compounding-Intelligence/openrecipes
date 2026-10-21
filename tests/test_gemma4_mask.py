# SPDX-License-Identifier: Apache-2.0
"""gemma4 loss masks (data/gemma4_mask.py): each model turn's content and closing marker, its header as context.

The always-on tests use a small template of the same shape as Gemma 4's (a system turn, `model` for the assistant role,
consecutive assistant messages merged into one turn) and a byte-level BPE tokenizer trained here.
OPENRECIPES_GEMMA4_CHECKPOINT (a checkout with tokenizer.json and chat_template.jinja) runs the released tokenizer and
template.
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
_spec = importlib.util.spec_from_file_location("gemma4_mask", DATA / "gemma4_mask.py")
gemma4_mask = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gemma4_mask)

SPECIAL = ["<bos>", "<eos>", "<|turn>", "<turn|>"]
# Same shape as the released template: trimmed content, `model` for assistant, no turn break between assistant messages.
TEMPLATE = (
    "{{- bos_token -}}"
    "{%- set ns = namespace(prev=none) -%}{%- set loop_messages = messages -%}"
    "{%- if messages and messages[0]['role'] == 'system' -%}"
    "{{- '<|turn>system\\n' + messages[0]['content'] | trim + '<turn|>\\n' -}}"
    "{%- set loop_messages = messages[1:] -%}{%- endif -%}"
    "{%- for m in loop_messages -%}"
    "{%- set role = 'model' if m['role'] == 'assistant' else m['role'] -%}"
    "{%- if not (role == 'model' and ns.prev == 'assistant') -%}{{- '<|turn>' + role + '\\n' -}}{%- endif -%}"
    "{{- m['content'] | trim -}}"
    "{%- if not (role == 'model' and not loop.last and loop_messages[loop.index0 + 1]['role'] == 'assistant') -%}"
    "{{- '<turn|>\\n' -}}{%- endif -%}"
    "{%- set ns.prev = m['role'] -%}{%- endfor -%}"
)
CONVERSATION = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4."}, {"role": "user", "content": "  And 3+3, café?  "},
                {"role": "assistant", "content": " 6, naïve model "}]
TARGETS = ["4.<turn|>\n", "6, naïve model<turn|>\n"]


@pytest.fixture(scope="module")
def tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    corpus = ["system user model Be brief. What is 2+2? 4. And 3+3, café? 6, naïve"] * 20
    tok.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=400, special_tokens=SPECIAL,
                                                        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<bos>", eos_token="<eos>")
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


def test_model_turn_content_and_closing_marker_are_the_targets(tokenizer):
    gen = gemma4_mask.Gemma4LossMaskGenerator(tokenizer)
    tokens, mask = gen.get_loss_mask(CONVERSATION)
    text = gen.render(CONVERSATION)
    assert text.startswith("<bos><|turn>system\nBe brief.<turn|>\n") and text.count("<|turn>model\n") == 2
    assert tokens == tokenizer(text, add_special_tokens=False)["input_ids"]
    assert _targets(tokenizer, tokens, mask) == TARGETS
    assert gen.manifest()["type"] == "gemma4"


def test_consecutive_assistant_messages_are_refused(tokenizer):
    """The template renders them as one model turn, so no turn belongs to the second message."""
    gen = gemma4_mask.Gemma4LossMaskGenerator(tokenizer)
    merged = CONVERSATION[:3] + [{"role": "assistant", "content": "Also 5."}]
    assert "4.Also 5.<turn|>" in gen.render(merged)
    with pytest.raises(ValueError, match="growing prefix"):
        gen.get_loss_mask(merged)


def test_rejects_tools_unknown_roles_non_text_and_missing_assistant(tokenizer):
    gen = gemma4_mask.Gemma4LossMaskGenerator(tokenizer)
    ok = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    for messages, tools in ((ok, [{"name": "f"}]), ([{"role": "tool", "content": "x"}] + ok[1:], None),
                            ([{"role": "user", "content": ["x"]}] + ok[1:], None), (ok[:1], None)):
        with pytest.raises(ValueError):
            gen.get_loss_mask(messages, tools=tools)


def test_tokens_straddling_a_span_and_other_templates_are_refused(tokenizer):
    assert gemma4_mask.token_mask([(0, 2), (2, 4), (4, 4), (4, 6)], [(2, 6)]) == [0, 1, 0, 1]
    with pytest.raises(ValueError, match="straddles"):
        gemma4_mask.token_mask([(0, 3), (3, 6)], [(2, 6)])
    tokenizer.chat_template = TEMPLATE.replace("<|turn>", "<|im_start|>").replace("<turn|>", "<|im_end|>")
    try:
        with pytest.raises(ValueError, match="Gemma 4 turns"):
            gemma4_mask.Gemma4LossMaskGenerator(tokenizer)
    finally:
        tokenizer.chat_template = TEMPLATE


def test_builder_option_records_the_mask_in_the_manifest(tokenizer, tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    tokenizer.save_pretrained(tmp_path / "tok")
    rows = [{"messages": [{"role": "user", "content": "q%d" % i}, {"role": "assistant", "content": "a%d " % i * 9}],
             "source": "s"} for i in range(40)]
    for row in rows[::2]:   # two assistant messages in a row: the builder skips these
        row["messages"].append({"role": "assistant", "content": "merged"})
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "src.parquet")
    monkeypatch.syspath_prepend(str(DATA))
    spec = importlib.util.spec_from_file_location("generate_fixed_sft", DATA / "generate_fixed_sft.py")
    gfs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gfs)
    monkeypatch.setattr(gfs, "validate_slime_loader", lambda *a, **k: {})
    out = tmp_path / "rows" / "train.parquet"
    monkeypatch.setattr(sys, "argv", ["generate_fixed_sft.py", "--input", str(tmp_path / "src.parquet"),
                                      "--output", str(out), "--tokenizer", str(tmp_path / "tok"),
                                      "--loss-mask-type", "gemma4", "--sequence-length", "128", "--rows", "4",
                                      "--min-token-entropy-bits", "0"])
    gfs.main()
    manifest = json.loads(out.with_suffix(".parquet.manifest.json").read_text())
    assert manifest["tokenizer"]["loss_mask"]["type"] == "gemma4"
    assert manifest["generation"]["rows"] == 4 and manifest["statistics"]["supervised_tokens"] > 0
    assert manifest["source"]["malformed_docs_skipped"] > 0


@pytest.mark.skipif(not os.environ.get("OPENRECIPES_GEMMA4_CHECKPOINT"),
                    reason="set OPENRECIPES_GEMMA4_CHECKPOINT to a Gemma 4 checkout with tokenizer.json and "
                    "chat_template.jinja")
def test_real_tokenizer():
    from transformers import AutoTokenizer

    path = os.environ["OPENRECIPES_GEMMA4_CHECKPOINT"]
    tok = AutoTokenizer.from_pretrained(path, use_fast=True)
    gen = gemma4_mask.Gemma4LossMaskGenerator(tok, path)
    tokens, mask = gen.get_loss_mask(CONVERSATION)
    assert tok.decode(tokens) == gen.render(CONVERSATION)
    assert tokens[0] == tok.bos_token_id and gen.manifest()["chat_template_file_sha256"]
    assert _targets(tok, tokens, mask) == TARGETS
    with pytest.raises(ValueError, match="growing prefix"):
        gen.get_loss_mask(CONVERSATION[:3] + [{"role": "assistant", "content": "Also 5."}])
