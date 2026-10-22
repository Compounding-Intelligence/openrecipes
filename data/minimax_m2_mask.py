# SPDX-License-Identifier: Apache-2.0
"""Assistant-only loss masks for MiniMax-M2, from the checkpoint's chat template.

MiniMax-M2's template opens the conversation with `]~!b[` and a system turn (the first message when it is a system
message, else the template's default identity), opens each turn with `]~b]` and the role (`ai` for the assistant), and
closes every turn with `[e~[`, the EOS token, and a newline. This generator renders growing prefixes of the
conversation: what each assistant message adds must be one whole turn, `]~b]ai` and a newline, the content, then
`[e~[` and a newline. The content and the closing `[e~[` are supervised; the header and the newline after `[e~[`,
which the model never generates (it stops at EOS), are context. A conversation that fails a check raises ValueError
(the builder skips it).
"""

import hashlib
from pathlib import Path

ROLES = ("system", "user", "assistant")
AI_OPEN = "]~b]ai\n"
TURN_CLOSE = "[e~["
NEWLINE = "\n"


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


class MiniMaxM2LossMaskGenerator:
    def __init__(self, tokenizer, tokenizer_path: str | Path | None = None):
        template = getattr(tokenizer, "chat_template", None)
        if not isinstance(template, str):
            raise ValueError("the tokenizer has no chat template")
        self.tokenizer = tokenizer
        probe = self.render([{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}])
        if not probe.endswith("]~b]user\nx" + TURN_CLOSE + NEWLINE + AI_OPEN + "y" + TURN_CLOSE + NEWLINE):
            raise ValueError("the tokenizer's chat template does not render MiniMax-M2 turns")
        if tokenizer.eos_token != TURN_CLOSE:
            raise ValueError(f"expected EOS {TURN_CLOSE!r}, got {tokenizer.eos_token!r}")
        path = Path(tokenizer_path) / "chat_template.jinja" if tokenizer_path else None
        self.template_file_sha256 = _sha256(path.read_bytes()) if path and path.is_file() else None
        self.template_sha256 = _sha256(template.encode())

    def manifest(self) -> dict:
        return {"type": "minimax_m2", "chat_template_sha256": self.template_sha256,
                "chat_template_file_sha256": self.template_file_sha256,
                "supervised": "assistant (ai) turn content and its closing [e~[ (EOS); the ]~b]ai header and the "
                              "newline after [e~[ are context"}

    def render(self, messages: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=False)

    def target_spans(self, messages: list[dict], text: str) -> list[tuple[int, int]]:
        """Character spans of the supervised part of each assistant turn in `text`, the rendered conversation."""
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
            close = TURN_CLOSE + NEWLINE
            if (not piece.startswith(AI_OPEN) or not piece.endswith(close)
                    or TURN_CLOSE in piece[len(AI_OPEN):-len(close)]):
                raise ValueError(f"unexpected assistant rendering {piece[:40]!r}...{piece[-12:]!r}")
            spans.append((len(before) + len(AI_OPEN), len(through) - len(NEWLINE)))
        return spans

    def get_loss_mask(self, messages: list[dict], tools: list[dict] | None = None) -> tuple[list[int], list[int]]:
        if tools:
            raise ValueError("tool definitions are not supported by the minimax_m2 mask")
        messages = [{"role": message.get("role"), "content": message.get("content")} for message in messages]
        for message in messages:
            if message["role"] not in ROLES:
                raise ValueError(f"unsupported role {message['role']!r} (system, user and assistant only)")
            if not isinstance(message["content"], str):
                raise ValueError("message content must be a string")
        if not any(message["role"] == "assistant" for message in messages):
            raise ValueError("conversation has no assistant turn")

        text = self.render(messages)
        plain = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        tokens = list(plain["input_ids"])
        encoded = self.tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True)
        if list(encoded["input_ids"]) != tokens:
            raise ValueError("tokenizing the rendered text differs from apply_chat_template(tokenize=True)")
        return tokens, token_mask(plain["offset_mapping"], self.target_spans(messages, text))
