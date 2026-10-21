# SPDX-License-Identifier: Apache-2.0
"""inkling loss masks (data/inkling_mask.py): each assistant answer and its closing tokens, the header as context.

The always-on tests use a small template of the same shape as Inkling's (role and content-type tokens, a thinking-effort
system message before the first non-system message, `<|content_model_end_sampling|>` after each assistant message,
`<|message_model|>` as the generation prompt) and a byte-level BPE tokenizer trained here, with no EOS token, as
Inkling's. OPENRECIPES_INKLING_CHECKPOINT (a checkout with tokenizer.json and chat_template.jinja) runs the released
tokenizer and template.
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
_spec = importlib.util.spec_from_file_location("inkling_mask", DATA / "inkling_mask.py")
inkling_mask = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(inkling_mask)

SPECIAL = ["<|message_system|>", "<|message_user|>", "<|message_model|>", "<|content_text|>", "<|end_message|>",
           "<|content_model_end_sampling|>"]
# Same shape as the released template: the effort line before the first non-system message, no trimming.
TEMPLATE = (
    "{%- set effort_map = {'low': 0.2, 'high': 0.9} -%}"
    "{%- set role_token = {'user': '<|message_user|>', 'assistant': '<|message_model|>', "
    "'system': '<|message_system|>'} -%}"
    "{%- set eff = effort_map[reasoning_effort] if reasoning_effort is defined and reasoning_effort is not none "
    "else 0.9 -%}"
    "{%- set state = namespace(done=false) -%}"
    "{%- for m in messages -%}"
    "{%- if not state.done and m['role'] != 'system' -%}"
    "{{- '<|message_system|><|content_text|>Thinking effort level: ' ~ eff ~ '<|end_message|>' -}}"
    "{%- set state.done = true -%}{%- endif -%}"
    "{{- role_token[m['role']] ~ '<|content_text|>' ~ m['content'] ~ '<|end_message|>' -}}"
    "{%- if m['role'] == 'assistant' -%}{{- '<|content_model_end_sampling|>' -}}{%- endif -%}"
    "{%- endfor -%}"
    "{%- if not state.done -%}{{- '<|message_system|><|content_text|>Thinking effort level: ' ~ eff ~ "
    "'<|end_message|>' -}}{%- endif -%}"
    "{%- if add_generation_prompt -%}{{- '<|message_model|>' -}}{%- endif -%}"
)
CONVERSATION = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4."}, {"role": "user", "content": "  And 3+3, café?  "},
                {"role": "assistant", "content": " 6, naïve model \n"}]
CLOSE = "<|end_message|><|content_model_end_sampling|>"
TARGETS = ["<|content_text|>4." + CLOSE, "<|content_text|> 6, naïve model \n" + CLOSE]


@pytest.fixture(scope="module")
def tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    corpus = ["Thinking effort level: 0.9 Be brief. What is 2+2? 4. And 3+3, café? 6, naïve model ok bye"] * 20
    tok.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=400, special_tokens=SPECIAL,
                                                        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    fast = PreTrainedTokenizerFast(tokenizer_object=tok)
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


def test_answers_and_their_closing_tokens_are_the_targets(tokenizer):
    gen = inkling_mask.InklingLossMaskGenerator(tokenizer)
    tokens, mask = gen.get_loss_mask(CONVERSATION)
    text = gen.render(CONVERSATION)
    assert text.startswith("<|message_system|><|content_text|>Be brief.<|end_message|>" + inkling_mask.EFFORT_LINE)
    assert tokens == tokenizer(text, add_special_tokens=False)["input_ids"]
    assert _targets(tokenizer, tokens, mask) == TARGETS
    assert gen.manifest()["type"] == "inkling" and gen.manifest()["template_kwargs"] == {"reasoning_effort": "high"}


def test_a_trailing_user_turn_is_context(tokenizer):
    gen = inkling_mask.InklingLossMaskGenerator(tokenizer)
    tokens, mask = gen.get_loss_mask(CONVERSATION[1:3] + [{"role": "user", "content": "ok bye"}])
    assert _targets(tokenizer, tokens, mask) == TARGETS[:1]


def test_empty_answers_tools_unknown_roles_non_text_and_missing_assistant_are_refused(tokenizer):
    gen = inkling_mask.InklingLossMaskGenerator(tokenizer)
    ok = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    for messages, tools in ((ok[:1] + [{"role": "assistant", "content": ""}], None), (ok, [{"name": "f"}]),
                            ([{"role": "tool", "content": "x"}] + ok[1:], None),
                            ([{"role": "user", "content": ["x"]}] + ok[1:], None), (ok[:1], None)):
        with pytest.raises(ValueError):
            gen.get_loss_mask(messages, tools=tools)


def test_a_template_that_rewrites_earlier_answers_is_refused(tokenizer):
    """Each answer's context must be the prompt a model would see for it; a template that rewrites an earlier answer
    once a later message follows fails the growing-prefix check instead of producing a mask."""
    gen = inkling_mask.InklingLossMaskGenerator(tokenizer)
    tokenizer.chat_template = TEMPLATE.replace("~ m['content'] ~", "~ (m['content'] if loop.last else 'x') ~")
    try:
        with pytest.raises(ValueError, match="growing prefix"):
            gen.get_loss_mask(CONVERSATION)
    finally:
        tokenizer.chat_template = TEMPLATE


def test_tokens_straddling_a_span_and_other_templates_are_refused(tokenizer):
    assert inkling_mask.token_mask([(0, 2), (2, 4), (4, 4), (4, 6)], [(2, 6)]) == [0, 1, 0, 1]
    with pytest.raises(ValueError, match="straddles"):
        inkling_mask.token_mask([(0, 3), (3, 6)], [(2, 6)])
    tokenizer.chat_template = TEMPLATE.replace("Thinking effort level: ", "")
    try:
        with pytest.raises(ValueError, match="Inkling messages"):
            inkling_mask.InklingLossMaskGenerator(tokenizer)
    finally:
        tokenizer.chat_template = TEMPLATE


def test_builder_option_records_the_mask_in_the_manifest(tokenizer, tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    tokenizer.save_pretrained(tmp_path / "tok")
    rows = [{"messages": [{"role": "user", "content": "q%d" % i}, {"role": "assistant", "content": "a%d " % i * 9}],
             "source": "s"} for i in range(40)]
    for row in rows[::2]:   # an empty answer: the builder skips these
        row["messages"] += [{"role": "user", "content": "more"}, {"role": "assistant", "content": ""}]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "src.parquet")
    monkeypatch.syspath_prepend(str(DATA))
    spec = importlib.util.spec_from_file_location("generate_fixed_sft", DATA / "generate_fixed_sft.py")
    gfs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gfs)
    monkeypatch.setattr(gfs, "validate_slime_loader", lambda *a, **k: {})
    out = tmp_path / "rows" / "train.parquet"
    monkeypatch.setattr(sys, "argv", ["generate_fixed_sft.py", "--input", str(tmp_path / "src.parquet"),
                                      "--output", str(out), "--tokenizer", str(tmp_path / "tok"),
                                      "--loss-mask-type", "inkling", "--sequence-length", "128", "--rows", "4",
                                      "--min-token-entropy-bits", "0"])
    gfs.main()
    manifest = json.loads(out.with_suffix(".parquet.manifest.json").read_text())
    assert manifest["tokenizer"]["loss_mask"]["type"] == "inkling"
    assert manifest["generation"]["rows"] == 4 and manifest["statistics"]["supervised_tokens"] > 0
    assert manifest["source"]["malformed_docs_skipped"] > 0


@pytest.mark.skipif(not os.environ.get("OPENRECIPES_INKLING_CHECKPOINT"),
                    reason="set OPENRECIPES_INKLING_CHECKPOINT to an Inkling checkout with tokenizer.json and "
                    "chat_template.jinja")
def test_real_tokenizer():
    from transformers import AutoTokenizer

    path = os.environ["OPENRECIPES_INKLING_CHECKPOINT"]
    tok = AutoTokenizer.from_pretrained(path, use_fast=True)
    gen = inkling_mask.InklingLossMaskGenerator(tok, path)
    tokens, mask = gen.get_loss_mask(CONVERSATION)
    assert tok.decode(tokens) == gen.render(CONVERSATION)
    assert tok.convert_ids_to_tokens(tokens[-1]) == "<|content_model_end_sampling|>"
    assert gen.manifest()["chat_template_file_sha256"] == gen.template_sha256
    assert _targets(tok, tokens, mask) == TARGETS
    with pytest.raises(ValueError, match="empty assistant"):
        gen.get_loss_mask(CONVERSATION[:2] + [{"role": "assistant", "content": ""}])
