# SPDX-License-Identifier: Apache-2.0
"""kimi_k25 loss masks (data/kimi_k25_mask.py): each assistant turn's content and closing <|im_end|>, its header and
empty think block as context.

The always-on tests use a small template of the same shape as Kimi K2.6's (role tokens, `<|im_middle|>`, an empty
`<think></think>` before each assistant answer, a `<think>` generation prompt) and a byte-level BPE tokenizer trained
here. OPENRECIPES_KIMI_CHECKPOINT (a Kimi K2.5 or K2.6 checkout with tiktoken.model, tokenization_kimi.py and
chat_template.jinja) runs the released tokenizer, which has no offsets, and template; it needs tiktoken and runs the
checkout's tokenizer code.
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
_spec = importlib.util.spec_from_file_location("kimi_k25_mask", DATA / "kimi_k25_mask.py")
kimi_k25_mask = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kimi_k25_mask)

SPECIAL = ["[BOS]", "[EOS]", "<|im_end|>", "<|im_user|>", "<|im_assistant|>", "<|im_system|>", "<|im_middle|>",
           "<think>", "</think>"]
# Same shape as the released template: no default system turn, content not trimmed, an empty think block before each
# assistant answer, and the generation prompt `<|im_assistant|>assistant<|im_middle|><think>`.
TEMPLATE = (
    "{%- for m in messages -%}"
    "{%- if m['role'] == 'user' -%}{{- '<|im_user|>user<|im_middle|>' -}}"
    "{%- elif m['role'] == 'assistant' -%}{{- '<|im_assistant|>assistant<|im_middle|><think></think>' -}}"
    "{%- else -%}{{- '<|im_system|>' + m['role'] + '<|im_middle|>' -}}{%- endif -%}"
    "{{- m['content'] + '<|im_end|>' -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|im_assistant|>assistant<|im_middle|><think>' -}}{%- endif -%}"
)
CONVERSATION = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4."}, {"role": "user", "content": "  And 3+3, café?  "},
                {"role": "assistant", "content": " 6, naïve model "}]
TARGETS = ["4.<|im_end|>", " 6, naïve model <|im_end|>"]


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
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="[BOS]", eos_token="[EOS]")
    fast.chat_template = TEMPLATE
    return fast


def _runs(tokenizer, tokens, mask):
    """Decoded (supervised, text) runs, in order."""
    runs, cur, on = [], [], None
    for t, m in zip(tokens, mask, strict=True):
        if cur and m != on:
            runs.append((on, tokenizer.decode(cur)))
            cur = []
        on = m
        cur.append(t)
    return runs + [(on, tokenizer.decode(cur))]


def _targets(tokenizer, tokens, mask):
    return [text for on, text in _runs(tokenizer, tokens, mask) if on]


def test_assistant_content_and_im_end_are_the_targets(tokenizer):
    gen = kimi_k25_mask.KimiK25LossMaskGenerator(tokenizer)
    tokens, mask = gen.get_loss_mask(CONVERSATION)
    text = gen.render(CONVERSATION)
    assert text.startswith("<|im_system|>system<|im_middle|>Be brief.<|im_end|>") and text.endswith("model <|im_end|>")
    assert tokens == tokenizer(text, add_special_tokens=False)["input_ids"]
    assert _targets(tokenizer, tokens, mask) == TARGETS
    context = [text for on, text in _runs(tokenizer, tokens, mask) if not on]
    assert all(c.endswith("<|im_assistant|>assistant<|im_middle|><think></think>") for c in context)
    assert gen.manifest()["type"] == "kimi_k25"


def test_rejects_tools_unknown_roles_non_text_special_text_and_missing_assistant(tokenizer):
    gen = kimi_k25_mask.KimiK25LossMaskGenerator(tokenizer)
    ok = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    for messages, tools in ((ok, [{"name": "f"}]), ([{"role": "tool", "content": "x"}] + ok[1:], None),
                            ([{"role": "user", "content": ["x"]}] + ok[1:], None), (ok[:1], None),
                            (ok[:1] + [{"role": "assistant", "content": "a<|im_end|>b"}], None),
                            ([{"role": "user", "content": "x</think>"}] + ok[1:], None)):
        with pytest.raises(ValueError):
            gen.get_loss_mask(messages, tools=tools)


def test_empty_answer_consecutive_assistants_and_trailing_user(tokenizer):
    gen = kimi_k25_mask.KimiK25LossMaskGenerator(tokenizer)
    messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": ""},
                {"role": "assistant", "content": "a"}, {"role": "user", "content": "more?"}]
    tokens, mask = gen.get_loss_mask(messages)
    assert _targets(tokenizer, tokens, mask) == ["<|im_end|>", "a<|im_end|>"]
    assert _runs(tokenizer, tokens, mask)[-1] == (0, "<|im_user|>user<|im_middle|>more?<|im_end|>")

def test_other_templates_are_refused(tokenizer):
    for template in (TEMPLATE.replace("<think></think>", ""), TEMPLATE.replace("<|im_middle|>", "\n")):
        tokenizer.chat_template = template
        try:
            with pytest.raises(ValueError, match="Kimi K2.5/K2.6 turns"):
                kimi_k25_mask.KimiK25LossMaskGenerator(tokenizer)
        finally:
            tokenizer.chat_template = TEMPLATE


def _build(tok_dir, tmp_path, monkeypatch, extra=()):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rows = [{"messages": [{"role": "user", "content": "q%d" % i}, {"role": "assistant", "content": "a%d " % i * 9}],
             "source": "s"} for i in range(40)]
    for row in rows[::4]:   # a tool turn: the builder skips these
        row["messages"].append({"role": "tool", "content": "r"})
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "src.parquet")
    monkeypatch.syspath_prepend(str(DATA))
    spec = importlib.util.spec_from_file_location("generate_fixed_sft", DATA / "generate_fixed_sft.py")
    gfs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gfs)
    monkeypatch.setattr(gfs, "validate_slime_loader", lambda *a, **k: {})
    out = tmp_path / "rows" / "train.parquet"
    monkeypatch.setattr(sys, "argv", ["generate_fixed_sft.py", "--input", str(tmp_path / "src.parquet"),
                                      "--output", str(out), "--tokenizer", str(tok_dir),
                                      "--loss-mask-type", "kimi_k25", "--sequence-length", "128", "--rows", "4",
                                      "--min-token-entropy-bits", "0", *extra])
    gfs.main()
    return json.loads(out.with_suffix(".parquet.manifest.json").read_text())


def test_builder_option_records_the_mask_in_the_manifest(tokenizer, tmp_path, monkeypatch):
    tokenizer.save_pretrained(tmp_path / "tok")
    manifest = _build(tmp_path / "tok", tmp_path, monkeypatch)
    assert manifest["tokenizer"]["loss_mask"]["type"] == "kimi_k25"
    assert manifest["generation"]["rows"] == 4 and manifest["statistics"]["supervised_tokens"] > 0
    assert manifest["source"]["malformed_docs_skipped"] > 0


@pytest.mark.skipif(not os.environ.get("OPENRECIPES_KIMI_CHECKPOINT"),
                    reason="set OPENRECIPES_KIMI_CHECKPOINT to a Kimi K2.5 or K2.6 checkout with tiktoken.model, "
                    "tokenization_kimi.py and chat_template.jinja")
def test_real_tokenizer(tmp_path, monkeypatch):
    pytest.importorskip("tiktoken")
    from transformers import AutoTokenizer

    path = os.environ["OPENRECIPES_KIMI_CHECKPOINT"]
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    assert not tok.is_fast
    gen = kimi_k25_mask.KimiK25LossMaskGenerator(tok, path)
    tokens, mask = gen.get_loss_mask(CONVERSATION)
    assert tok.decode(tokens) == gen.render(CONVERSATION) and gen.manifest()["chat_template_file_sha256"]
    assert _targets(tok, tokens, mask) == TARGETS
    assert tok.convert_tokens_to_ids("<|im_end|>") == tokens[-1] and mask[-1] == 1
    with pytest.raises(ValueError):
        gen.get_loss_mask(CONVERSATION[:2] + [{"role": "assistant", "content": "<|reserved_token_163784|>"}])
    manifest = _build(path, tmp_path, monkeypatch, ["--trust-remote-code"])
    assert manifest["tokenizer"]["loss_mask"]["type"] == "kimi_k25" and "tiktoken.model" in manifest["tokenizer"]["files"]
