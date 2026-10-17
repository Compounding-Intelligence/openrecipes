# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Assistant-only loss masks for Muse Spark, from the `{% generation %}` spans of the checkpoint's chat template.

The template wraps each assistant message in `{% generation %}`; transformers' `return_assistant_tokens_mask` turns
those spans into a token mask, which must equal the character spans of the assistant messages found by rendering
growing prefixes of the conversation. Within each span the header `<|start|>assistant to=<recipient><|message|>` is
context, and the content with its end marker (`<|eot|>` or `<|eom|>`) is supervised, the convention of the gpt-oss
and Gemma 4 masks. The template stamps the current date and a reasoning-strength line into the system turn;
TEMPLATE_KWARGS pins both, so the rows do not depend on the build date. A conversation where the checks disagree
raises ValueError (the builder skips it).
"""

import hashlib
import re
from pathlib import Path

ROLES = ("system", "user", "assistant")
# current_date replaces the template's strftime_now() default; xhigh is the template's and the miles recipe's default
TEMPLATE_KWARGS = {"current_date": "2026-10-01", "reasoning_effort": "xhigh"}
ASSISTANT_OPEN = "<|start|>assistant"
HEADER_END = "<|message|>"
TURN_ENDS = ("<|eot|>", "<|eom|>")
_GENERATION_TAG = re.compile(r"\{%-?\s*generation\s*-?%\}")


def register_config() -> bool:
    """Register miles' muse_spark config with transformers, which needs it to load the checkpoint's tokenizer."""
    try:
        from miles.utils.hf_utils.config import register_hf_config_aliases
    except ImportError:
        return False
    register_hf_config_aliases()
    return True


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Content-plus-end-marker spans of one assistant rendering text[start:end], a sequence of
    `<|start|>assistant ...<|message|>` header, content, end marker (content holds no `<|`: the template escapes it)."""
    spans, pos = [], start
    while pos < end:
        if not text.startswith(ASSISTANT_OPEN, pos):
            raise ValueError(f"assistant part does not open with {ASSISTANT_OPEN}: {text[pos:pos + 40]!r}")
        body = text.find(HEADER_END, pos, end)
        if body < 0:
            raise ValueError("assistant header has no <|message|>")
        body += len(HEADER_END)
        stop = min((i for i in (text.find(m, body, end) for m in TURN_ENDS) if i >= 0), default=-1)
        if stop < 0:
            raise ValueError("assistant content has no end marker")
        stop += len(next(m for m in TURN_ENDS if text.startswith(m, stop)))
        spans.append((body, stop))
        pos = stop
    return spans


def _token_mask(offsets, spans, what) -> list[int]:
    """1 for each token inside one of `spans`; a token straddling a span boundary raises ValueError."""
    out = []
    for start, end in offsets:
        inside = any(s <= start and end <= e for s, e in spans)
        if not inside and any(s < end and start < e for s, e in spans):
            raise ValueError(f"a token straddles {what}")
        out.append(int(inside))
    return out


class MuseSparkLossMaskGenerator:
    def __init__(self, tokenizer, tokenizer_path: str | Path | None = None, template_kwargs: dict | None = None):
        template = getattr(tokenizer, "chat_template", None)
        if not isinstance(template, str) or not _GENERATION_TAG.search(template):
            raise ValueError("the tokenizer's chat template has no {% generation %} spans")
        self.tokenizer = tokenizer
        self.kwargs = dict(TEMPLATE_KWARGS if template_kwargs is None else template_kwargs)
        path = Path(tokenizer_path) / "chat_template.jinja" if tokenizer_path else None
        self.template_file_sha256 = _sha256(path.read_bytes()) if path and path.is_file() else None
        self.template_sha256 = _sha256(template.encode())

    def manifest(self) -> dict:
        return {"type": "muse_spark", "template_kwargs": self.kwargs, "chat_template_sha256": self.template_sha256,
                "chat_template_file_sha256": self.template_file_sha256,
                "supervised": "assistant content and its end marker; the <|start|>assistant ...<|message|> header "
                              "is context"}

    def render(self, messages: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=False, **self.kwargs)

    def assistant_spans(self, messages: list[dict], text: str) -> list[tuple[int, int]]:
        """Character spans of the assistant messages in `text`: what each one adds to the rendered prefix."""
        spans, prefixes = [], {len(messages): text}

        def prefix(n):
            if n not in prefixes:
                prefixes[n] = self.render(messages[:n])
            return prefixes[n]

        for index, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            before, through = prefix(index), prefix(index + 1)
            if not through.startswith(before) or not text.startswith(through):
                raise ValueError("the template does not render this conversation as a growing prefix")
            piece = through[len(before):]
            if not piece.startswith(ASSISTANT_OPEN) or not piece.endswith(TURN_ENDS):
                raise ValueError(f"unexpected assistant rendering {piece[:40]!r}...{piece[-12:]!r}")
            spans.append((len(before), len(through)))
        return spans

    def get_loss_mask(self, messages: list[dict], tools: list[dict] | None = None) -> tuple[list[int], list[int]]:
        if tools:
            raise ValueError("tool definitions are not supported by the muse_spark mask")
        messages = [{"role": message.get("role"), "content": message.get("content")} for message in messages]
        for message in messages:
            if message["role"] not in ROLES:
                raise ValueError(f"unsupported role {message['role']!r} (system, user and assistant only)")
            if not isinstance(message["content"], str):
                raise ValueError("message content must be a string")
        if not any(message["role"] == "assistant" for message in messages):
            raise ValueError("conversation has no assistant turn")

        text = self.render(messages)
        date = self.kwargs.get("current_date")
        if not date or f"Current date: {date}." not in text:
            raise ValueError("the pinned current_date did not reach the chat template")
        encoded = self.tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True,
                                                     return_assistant_tokens_mask=True, **self.kwargs)
        tokens, mask = list(encoded["input_ids"]), [int(v) for v in encoded["assistant_masks"]]
        plain = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        if list(plain["input_ids"]) != tokens:
            raise ValueError("tokenizing the rendered text differs from apply_chat_template(tokenize=True)")
        offsets = plain["offset_mapping"]
        spans = self.assistant_spans(messages, text)
        if _token_mask(offsets, spans, "an assistant message boundary") != mask:
            raise ValueError("the template's generation spans differ from the assistant messages")
        content = [c for s, e in spans for c in content_spans(text, s, e)]
        return tokens, _token_mask(offsets, content, "an assistant header boundary")
