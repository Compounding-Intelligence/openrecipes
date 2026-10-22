# SPDX-License-Identifier: Apache-2.0
"""Assistant-only loss masks for GLM-4.5, from the checkpoint's chat template.

GLM-4.5's template opens a conversation with `[gMASK]<sop>` and each message with its role token. It renders an
assistant message as `<|assistant|>`, a newline, a think block (`<think></think>` when the message has no reasoning), a
newline and the stripped content, and writes no end-of-turn token: the model ends a turn by generating the next role
token, `<|user|>` (GLM-4.5's generation config lists it among the EOS ids). This generator renders growing prefixes of
the conversation; what each assistant message adds must be one whole turn that starts with `<|assistant|>`. Everything
the turn renders after that header (the think block and the content) and the `<|user|>` that ends it are supervised:
the next message's `<|user|>`, or a `<|user|>` appended after the last message. The header, system and user messages
are context. A conversation that fails a check raises ValueError (the builder skips it); so does an assistant message
followed by anything other than a user message.
"""

import hashlib
from pathlib import Path

ROLES = ("system", "user", "assistant")
PREFIX = "[gMASK]<sop>"
HEADER = "<|assistant|>"
STOP = "<|user|>"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def token_mask(offsets, spans) -> list[int]:
    """1 for each token whose characters lie inside one of `spans`; a token straddling a span boundary raises
    ValueError. A token without characters is never a target."""
    out = []
    for start, end in offsets:
        if end <= start:
            out.append(0)
            continue
        inside = any(s <= start and end <= e for s, e in spans)
        if not inside and any(s < end and start < e for s, e in spans):
            raise ValueError("a token straddles a supervised span boundary")
        out.append(int(inside))
    return out


class GLM45LossMaskGenerator:
    def __init__(self, tokenizer, tokenizer_path: str | Path | None = None):
        template = getattr(tokenizer, "chat_template", None)
        if not isinstance(template, str):
            raise ValueError("the tokenizer has no chat template")
        self.tokenizer = tokenizer
        probe = self.render([{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}])
        if not (probe.startswith(PREFIX) and probe.endswith(STOP + "\nx" + HEADER + "\n<think></think>\ny")):
            raise ValueError("the tokenizer's chat template does not render GLM-4.5 turns")
        stop_ids = tokenizer(STOP, add_special_tokens=False)["input_ids"]
        if len(stop_ids) != 1:
            raise ValueError(f"{STOP} is not a single token")
        self.stop_id = stop_ids[0]
        path = Path(tokenizer_path) / "chat_template.jinja" if tokenizer_path else None
        self.template_file_sha256 = _sha256(path.read_bytes()) if path and path.is_file() else None
        self.template_sha256 = _sha256(template.encode())

    def manifest(self) -> dict:
        return {"type": "glm45", "chat_template_sha256": self.template_sha256,
                "chat_template_file_sha256": self.template_file_sha256,
                "supervised": "each assistant turn after its <|assistant|> header (the newline, the think block, the "
                              "content) and the <|user|> that ends it, appended after the last message; the header, "
                              "system and user messages are context"}

    def render(self, messages: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=False)

    def target_spans(self, messages: list[dict], text: str) -> list[tuple[int, int]]:
        """Character spans of the supervised part of each assistant turn in `text`, the rendered conversation with
        the closing `<|user|>` appended when the last message is an assistant message."""
        spans, prefixes = [], {}

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
            if not piece.startswith(HEADER) or HEADER in piece[len(HEADER):] or STOP in piece:
                raise ValueError(f"unexpected assistant rendering {piece[:40]!r}...{piece[-12:]!r}")
            if not text.startswith(STOP, len(through)):
                raise ValueError("an assistant turn is not followed by <|user|>")
            spans.append((len(before) + len(HEADER), len(through) + len(STOP)))
        return spans

    def get_loss_mask(self, messages: list[dict], tools: list[dict] | None = None) -> tuple[list[int], list[int]]:
        if tools:
            raise ValueError("tool definitions are not supported by the glm45 mask")
        messages = [{"role": message.get("role"), "content": message.get("content")} for message in messages]
        for message in messages:
            if message["role"] not in ROLES:
                raise ValueError(f"unsupported role {message['role']!r} (system, user and assistant only)")
            if not isinstance(message["content"], str):
                raise ValueError("message content must be a string")
        if not any(message["role"] == "assistant" for message in messages):
            raise ValueError("conversation has no assistant turn")

        rendered = self.render(messages)
        encoded = self.tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True)
        plain = self.tokenizer(rendered, add_special_tokens=False)
        if list(encoded["input_ids"]) != list(plain["input_ids"]):
            raise ValueError("tokenizing the rendered text differs from apply_chat_template(tokenize=True)")
        text = rendered + (STOP if messages[-1]["role"] == "assistant" else "")
        tokenized = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        tokens = list(tokenized["input_ids"])
        expected = list(plain["input_ids"]) + ([self.stop_id] if text != rendered else [])
        if tokens != expected:
            raise ValueError("appending <|user|> changed the conversation's tokens")
        return tokens, token_mask(tokenized["offset_mapping"], self.target_spans(messages, text))
