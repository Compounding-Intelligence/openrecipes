# SPDX-License-Identifier: Apache-2.0
"""Assistant-only loss masks for Inkling, from the checkpoint's chat template.

Inkling's template renders every message as a role token, a content-type token, the content and `<|end_message|>`; an
assistant message uses the role token `<|message_model|>` and is followed by `<|content_model_end_sampling|>`, the
model's end-of-turn token (its config's `eos_token_id`). The generation prompt is `<|message_model|>` alone, so a
text answer is `<|content_text|>`, the content, `<|end_message|>` and `<|content_model_end_sampling|>`. This
generator renders growing prefixes of the conversation: the prefix before each assistant message, rendered with the
generation prompt, must be a prefix of the conversation through that message, and what the message adds after the
prompt must be exactly that answer. The answer is supervised; the `<|message_model|>` header, which the generation
prompt supplies, and every other message are context. The template inserts a system message with the thinking-effort
level before the first non-system message; TEMPLATE_KWARGS pins it to 0.9, the template's own default. The checkpoint's
template renders an assistant message with empty content as an empty answer, and miles' corrected copy of it
(`miles/utils/chat_template_utils/templates/inkling_fixed.jinja`) leaves the message out, so a conversation with one is
skipped rather than trained on an empty answer. A conversation that fails a check raises an error, and the builder skips
it.
"""

import hashlib
from pathlib import Path

ROLES = ("system", "user", "assistant")
MODEL_OPEN = "<|message_model|>"
ANSWER_OPEN = "<|content_text|>"
ANSWER_CLOSE = "<|end_message|><|content_model_end_sampling|>"
# "high" is 0.9 in the template's effort map, the value it uses when reasoning_effort is unset
TEMPLATE_KWARGS = {"reasoning_effort": "high"}
EFFORT_LINE = "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"


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


class InklingLossMaskGenerator:
    def __init__(self, tokenizer, tokenizer_path: str | Path | None = None):
        template = getattr(tokenizer, "chat_template", None)
        if not isinstance(template, str):
            raise ValueError("the tokenizer has no chat template")
        self.tokenizer = tokenizer
        self.kwargs = dict(TEMPLATE_KWARGS)
        probe = self.render([{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}])
        if probe != EFFORT_LINE + "<|message_user|><|content_text|>x<|end_message|>" + MODEL_OPEN + ANSWER_OPEN + "y" \
                + ANSWER_CLOSE:
            raise ValueError("the tokenizer's chat template does not render Inkling messages")
        path = Path(tokenizer_path) / "chat_template.jinja" if tokenizer_path else None
        self.template_file_sha256 = _sha256(path.read_bytes()) if path and path.is_file() else None
        self.template_sha256 = _sha256(template.encode())

    def manifest(self) -> dict:
        return {"type": "inkling", "chat_template_sha256": self.template_sha256,
                "chat_template_file_sha256": self.template_file_sha256, "template_kwargs": self.kwargs,
                "supervised": "each assistant answer after the <|message_model|> generation prompt: <|content_text|>, the "
                              "content, <|end_message|> and <|content_model_end_sampling|>; the header and the other "
                              "messages are context"}

    def render(self, messages: list[dict], add_generation_prompt: bool = False) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt,
                                                  **self.kwargs)

    def target_spans(self, messages: list[dict], text: str) -> list[tuple[int, int]]:
        """Character spans of the supervised answer of each assistant message in `text`, the rendered conversation."""
        spans = []
        for index, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            before = self.render(messages[:index], add_generation_prompt=True)
            through = self.render(messages[: index + 1])
            if not before.endswith(MODEL_OPEN) or not through.startswith(before) or not text.startswith(through):
                raise ValueError("the template does not render this conversation as a growing prefix")
            piece = through[len(before):]
            if piece != ANSWER_OPEN + message["content"] + ANSWER_CLOSE:
                raise ValueError(f"unexpected assistant rendering {piece[:40]!r}...{piece[-40:]!r}")
            spans.append((len(before), len(through)))
        return spans

    def get_loss_mask(self, messages: list[dict], tools: list[dict] | None = None) -> tuple[list[int], list[int]]:
        if tools:
            raise ValueError("tool definitions are not supported by the inkling mask")
        messages = [{"role": message.get("role"), "content": message.get("content")} for message in messages]
        for message in messages:
            if message["role"] not in ROLES:
                raise ValueError(f"unsupported role {message['role']!r} (system, user and assistant only)")
            if not isinstance(message["content"], str):
                raise ValueError("message content must be a string")
            if message["role"] == "assistant" and not message["content"]:
                raise ValueError("empty assistant message")
        if not any(message["role"] == "assistant" for message in messages):
            raise ValueError("conversation has no assistant turn")

        text = self.render(messages)
        plain = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        tokens = list(plain["input_ids"])
        encoded = self.tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True, **self.kwargs)
        if list(encoded["input_ids"]) != tokens:
            raise ValueError("tokenizing the rendered text differs from apply_chat_template(tokenize=True)")
        return tokens, token_mask(plain["offset_mapping"], self.target_spans(messages, text))
