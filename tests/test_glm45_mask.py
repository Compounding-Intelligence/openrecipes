# SPDX-License-Identifier: Apache-2.0
"""glm45 loss masks (data/glm45_mask.py): each assistant turn after its header and the <|user|> that ends it.

The always-on tests use a small template of the same shape as GLM-4.5's (`[gMASK]<sop>`, role tokens without an
end-of-turn token, `<think></think>` before the stripped assistant content) and a byte-level BPE tokenizer trained
here. OPENRECIPES_GLM45_CHECKPOINT (a checkout with tokenizer.json and chat_template.jinja) runs the released tokenizer
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
_spec = importlib.util.spec_from_file_location("glm45_mask", DATA / "glm45_mask.py")
glm45_mask = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(glm45_mask)

SPECIAL = ["[gMASK]", "<sop>", "<|endoftext|>", "<|system|>", "<|user|>", "<|assistant|>", "<|observation|>",
           "<think>", "</think>"]
# Same shape as the released template: no end-of-turn token, an empty think block, stripped assistant content.
TEMPLATE = (
    "{{- '[gMASK]<sop>' -}}"
    "{%- for m in messages -%}"
    "{%- if m['role'] == 'user' -%}{{- '<|user|>\\n' + m['content'] -}}"
    "{%- elif m['role'] == 'assistant' -%}{{- '<|assistant|>\\n<think></think>' -}}"
    "{%- if m['content'].strip() -%}{{- '\\n' + m['content'].strip() -}}{%- endif -%}"
    "{%- elif m['role'] == 'system' -%}{{- '<|system|>\\n' + m['content'] -}}{%- endif -%}"
    "{%- endfor -%}"
)
CONVERSATION = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4."}, {"role": "user", "content": "  And 3+3, café?  "},
                {"role": "assistant", "content": " 6, naïve model "}]
TARGETS = ["\n<think></think>\n4.<|user|>", "\n<think></think>\n6, naïve model<|user|>"]


@pytest.fixture(scope="module")
def tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    corpus = ["system user assistant Be brief. What is 2+2? 4. And 3+3, café? 6, naïve model"] * 20
    tok.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=400, special_tokens=SPECIAL,
                                                        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|endoftext|>")
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


def test_turn_after_the_header_and_the_closing_user_token_are_the_targets(tokenizer):
    gen = glm45_mask.GLM45LossMaskGenerator(tokenizer)
    tokens, mask = gen.get_loss_mask(CONVERSATION)
    text = gen.render(CONVERSATION)
    assert text.startswith("[gMASK]<sop><|system|>\nBe brief.<|user|>\n") and text.count("<|assistant|>") == 2
    assert tokens == tokenizer(text, add_special_tokens=False)["input_ids"] + [tokenizer.convert_tokens_to_ids("<|user|>")]
    assert _targets(tokenizer, tokens, mask) == TARGETS
    assert gen.manifest()["type"] == "glm45"


def test_a_trailing_user_message_is_context(tokenizer):
    gen = glm45_mask.GLM45LossMaskGenerator(tokenizer)
    tokens, mask = gen.get_loss_mask(CONVERSATION[:4])
    assert tokens == tokenizer(gen.render(CONVERSATION[:4]), add_special_tokens=False)["input_ids"]
    assert _targets(tokenizer, tokens, mask) == TARGETS[:1]


def test_an_assistant_turn_must_end_with_a_user_message(tokenizer):
    """Two assistant messages in a row, or a system message after one, leave a turn without its <|user|>."""
    gen = glm45_mask.GLM45LossMaskGenerator(tokenizer)
    for follow in ({"role": "assistant", "content": "Also 5."}, {"role": "system", "content": "Now terse."}):
        with pytest.raises(ValueError, match="not followed by <\\|user\\|>"):
            gen.get_loss_mask(CONVERSATION[:3] + [follow, CONVERSATION[3], CONVERSATION[4]])


def test_rejects_tools_unknown_roles_non_text_and_missing_assistant(tokenizer):
    gen = glm45_mask.GLM45LossMaskGenerator(tokenizer)
    ok = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    for messages, tools in ((ok, [{"name": "f"}]), ([{"role": "tool", "content": "x"}] + ok[1:], None),
                            ([{"role": "user", "content": ["x"]}] + ok[1:], None), (ok[:1], None)):
        with pytest.raises(ValueError):
            gen.get_loss_mask(messages, tools=tools)


def test_tokens_straddling_a_span_and_other_templates_are_refused(tokenizer):
    assert glm45_mask.token_mask([(0, 2), (2, 4), (4, 4), (4, 6)], [(2, 6)]) == [0, 1, 0, 1]
    with pytest.raises(ValueError, match="straddles"):
        glm45_mask.token_mask([(0, 3), (3, 6)], [(2, 6)])
    tokenizer.chat_template = TEMPLATE.replace("<think></think>", "")
    try:
        with pytest.raises(ValueError, match="GLM-4.5 turns"):
            glm45_mask.GLM45LossMaskGenerator(tokenizer)
    finally:
        tokenizer.chat_template = TEMPLATE


def test_builder_option_records_the_mask_in_the_manifest(tokenizer, tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    tokenizer.save_pretrained(tmp_path / "tok")
    rows = [{"messages": [{"role": "user", "content": "q%d" % i}, {"role": "assistant", "content": "a%d " % i * 9}],
             "source": "s"} for i in range(40)]
    for row in rows[::2]:   # two assistant messages in a row: the builder skips these
        row["messages"].append({"role": "assistant", "content": "again"})
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "src.parquet")
    monkeypatch.syspath_prepend(str(DATA))
    spec = importlib.util.spec_from_file_location("generate_fixed_sft", DATA / "generate_fixed_sft.py")
    gfs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gfs)
    monkeypatch.setattr(gfs, "validate_slime_loader", lambda *a, **k: {})
    out = tmp_path / "rows" / "train.parquet"
    monkeypatch.setattr(sys, "argv", ["generate_fixed_sft.py", "--input", str(tmp_path / "src.parquet"),
                                      "--output", str(out), "--tokenizer", str(tmp_path / "tok"),
                                      "--loss-mask-type", "glm45", "--sequence-length", "128", "--rows", "4",
                                      "--min-token-entropy-bits", "0"])
    gfs.main()
    manifest = json.loads(out.with_suffix(".parquet.manifest.json").read_text())
    assert manifest["tokenizer"]["loss_mask"]["type"] == "glm45"
    assert manifest["generation"]["rows"] == 4 and manifest["statistics"]["supervised_tokens"] > 0
    assert manifest["source"]["malformed_docs_skipped"] > 0


@pytest.mark.skipif(not os.environ.get("OPENRECIPES_GLM45_CHECKPOINT"),
                    reason="set OPENRECIPES_GLM45_CHECKPOINT to a GLM-4.5 checkout with tokenizer.json and "
                    "chat_template.jinja")
def test_real_tokenizer():
    from transformers import AutoTokenizer

    path = os.environ["OPENRECIPES_GLM45_CHECKPOINT"]
    tok = AutoTokenizer.from_pretrained(path, use_fast=True)
    gen = glm45_mask.GLM45LossMaskGenerator(tok, path)
    tokens, mask = gen.get_loss_mask(CONVERSATION)
    assert tok.decode(tokens) == gen.render(CONVERSATION) + "<|user|>"
    assert tok.convert_ids_to_tokens(tokens[:2]) == ["[gMASK]", "<sop>"] and gen.manifest()["chat_template_file_sha256"]
    assert _targets(tok, tokens, mask) == TARGETS
    with pytest.raises(ValueError, match="not followed"):
        gen.get_loss_mask(CONVERSATION[:3] + [{"role": "assistant", "content": "Also 5."}])
