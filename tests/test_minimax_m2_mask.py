# SPDX-License-Identifier: Apache-2.0
"""minimax_m2 loss masks (data/minimax_m2_mask.py): each assistant turn's content and closing EOS, its header as
context.

The always-on tests use a small template of the same shape as MiniMax-M2's (a system turn that defaults to an identity
line, `ai` for the assistant role, `[e~[` and a newline closing every turn) and a byte-level BPE tokenizer trained here.
OPENRECIPES_MINIMAX_M2_CHECKPOINT (a checkout with tokenizer.json and chat_template.jinja) runs the released tokenizer
and template.
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
_spec = importlib.util.spec_from_file_location("minimax_m2_mask", DATA / "minimax_m2_mask.py")
minimax_m2_mask = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(minimax_m2_mask)

SPECIAL = ["]~!b[", "]~b]", "[e~["]
# Same shape as the released template: a system turn first (default identity without a system message), the first
# system message only, unstripped content, every turn closed by `[e~[` and a newline.
TEMPLATE = (
    "{%- set ns = namespace(sys='You are a helpful assistant.', msgs=messages) -%}"
    "{%- if messages and messages[0]['role'] == 'system' -%}"
    "{%- set ns.sys = messages[0]['content'] -%}{%- set ns.msgs = messages[1:] -%}{%- endif -%}"
    "{{- ']~!b[]~b]system\\n' + ns.sys + '[e~[\\n' -}}"
    "{%- for m in ns.msgs -%}"
    "{%- if m['role'] == 'assistant' -%}{{- ']~b]ai\\n' + m['content'] + '[e~[\\n' -}}"
    "{%- elif m['role'] == 'user' -%}{{- ']~b]user\\n' + m['content'] + '[e~[\\n' -}}{%- endif -%}"
    "{%- endfor -%}"
)
CONVERSATION = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4."}, {"role": "user", "content": "  And 3+3, café?  "},
                {"role": "assistant", "content": " 6, naïve model "}]
TARGETS = ["4.[e~[", " 6, naïve model [e~["]


@pytest.fixture(scope="module")
def tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    corpus = ["system user ai Be brief. What is 2+2? 4. And 3+3, café? 6, naïve model helpful assistant"] * 20
    tok.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=400, special_tokens=SPECIAL,
                                                        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="]~!b[", eos_token="[e~[")
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


def test_assistant_content_and_closing_eos_are_the_targets(tokenizer):
    gen = minimax_m2_mask.MiniMaxM2LossMaskGenerator(tokenizer)
    tokens, mask = gen.get_loss_mask(CONVERSATION)
    text = gen.render(CONVERSATION)
    assert text.startswith("]~!b[]~b]system\nBe brief.[e~[\n") and text.count("]~b]ai\n") == 2
    assert tokens == tokenizer(text, add_special_tokens=False)["input_ids"]
    assert _targets(tokenizer, tokens, mask) == TARGETS
    assert gen.manifest()["type"] == "minimax_m2"


def test_default_system_turn_and_separate_assistant_turns(tokenizer):
    """Without a system message the template's identity line is context; each assistant message is its own turn."""
    gen = minimax_m2_mask.MiniMaxM2LossMaskGenerator(tokenizer)
    messages = CONVERSATION[1:3] + [{"role": "assistant", "content": "Also 5."}]
    tokens, mask = gen.get_loss_mask(messages)
    assert gen.render(messages).startswith("]~!b[]~b]system\nYou are a helpful assistant.[e~[\n")
    assert _targets(tokenizer, tokens, mask) == ["4.[e~[", "Also 5.[e~["]


def test_rejects_tools_unknown_roles_non_text_and_missing_assistant(tokenizer):
    gen = minimax_m2_mask.MiniMaxM2LossMaskGenerator(tokenizer)
    ok = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    for messages, tools in ((ok, [{"name": "f"}]), ([{"role": "tool", "content": "x"}] + ok[1:], None),
                            ([{"role": "user", "content": ["x"]}] + ok[1:], None), (ok[:1], None)):
        with pytest.raises(ValueError):
            gen.get_loss_mask(messages, tools=tools)


def test_tokens_straddling_a_span_and_other_templates_are_refused(tokenizer):
    assert minimax_m2_mask.token_mask([(0, 2), (2, 4), (4, 4), (4, 6)], [(2, 6)]) == [0, 1, 0, 1]
    with pytest.raises(ValueError, match="straddles"):
        minimax_m2_mask.token_mask([(0, 3), (3, 6)], [(2, 6)])
    tokenizer.chat_template = TEMPLATE.replace("]~b]ai", "]~b]assistant")
    try:
        with pytest.raises(ValueError, match="MiniMax-M2 turns"):
            minimax_m2_mask.MiniMaxM2LossMaskGenerator(tokenizer)
    finally:
        tokenizer.chat_template = TEMPLATE


def test_builder_option_records_the_mask_in_the_manifest(tokenizer, tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    tokenizer.save_pretrained(tmp_path / "tok")
    rows = [{"messages": [{"role": "user", "content": "q%d" % i}, {"role": "assistant", "content": "a%d " % i * 9}],
             "source": "s"} for i in range(40)]
    for row in rows[::2]:   # a tool message: the builder skips these
        row["messages"].append({"role": "tool", "content": "out"})
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "src.parquet")
    monkeypatch.syspath_prepend(str(DATA))
    spec = importlib.util.spec_from_file_location("generate_fixed_sft", DATA / "generate_fixed_sft.py")
    gfs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gfs)
    monkeypatch.setattr(gfs, "validate_slime_loader", lambda *a, **k: {})
    out = tmp_path / "rows" / "train.parquet"
    monkeypatch.setattr(sys, "argv", ["generate_fixed_sft.py", "--input", str(tmp_path / "src.parquet"),
                                      "--output", str(out), "--tokenizer", str(tmp_path / "tok"),
                                      "--loss-mask-type", "minimax_m2", "--sequence-length", "128", "--rows", "4",
                                      "--min-token-entropy-bits", "0"])
    gfs.main()
    manifest = json.loads(out.with_suffix(".parquet.manifest.json").read_text())
    assert manifest["tokenizer"]["loss_mask"]["type"] == "minimax_m2"
    assert manifest["generation"]["rows"] == 4 and manifest["statistics"]["supervised_tokens"] > 0
    assert manifest["source"]["malformed_docs_skipped"] > 0


@pytest.mark.skipif(not os.environ.get("OPENRECIPES_MINIMAX_M2_CHECKPOINT"),
                    reason="set OPENRECIPES_MINIMAX_M2_CHECKPOINT to a MiniMax-M2 checkout with tokenizer.json and "
                    "chat_template.jinja")
def test_real_tokenizer():
    from transformers import AutoTokenizer

    path = os.environ["OPENRECIPES_MINIMAX_M2_CHECKPOINT"]
    tok = AutoTokenizer.from_pretrained(path, use_fast=True)
    gen = minimax_m2_mask.MiniMaxM2LossMaskGenerator(tok, path)
    tokens, mask = gen.get_loss_mask(CONVERSATION)
    assert tok.decode(tokens) == gen.render(CONVERSATION)
    assert tok.convert_ids_to_tokens(tokens[:2]) == ["]~!b[", "]~b]"] and gen.manifest()["chat_template_file_sha256"]
    assert _targets(tok, tokens, mask) == TARGETS
    with pytest.raises(ValueError, match="straddles"):   # a leading newline merges with the header's newline
        gen.get_loss_mask([{"role": "user", "content": "x"}, {"role": "assistant", "content": "\nHello"}])
