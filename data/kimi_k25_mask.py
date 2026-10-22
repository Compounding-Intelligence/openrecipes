# SPDX-License-Identifier: Apache-2.0
"""Assistant-only loss masks for Kimi K2.5 and K2.6, from the checkpoint's chat template.

The Kimi K2.5/K2.6 template opens a turn with a role token (`<|im_user|>`, `<|im_assistant|>`, `<|im_system|>`), the
role name and `<|im_middle|>`, and closes it with `<|im_end|>`. An assistant message without reasoning renders as
`<|im_assistant|>assistant<|im_middle|><think></think>`, the content, then `<|im_end|>`; that header with its empty
think block is also the prompt the template builds in non-thinking mode (`thinking=False`), and `<|im_end|>` is the
checkpoint's end-of-generation token (K2.6 generation_config). So the content and the closing `<|im_end|>` are
supervised; the header and the empty think block are context.

This generator renders growing prefixes of the conversation (no generation prompt): what each assistant message adds
must be one whole turn of that shape. Kimi's tokenizer is a tiktoken one without offsets, so the token mask comes from
tokenizing the rendered text in pieces cut at the supervised span edges, which sit next to special tokens
(`</think>`, `<|im_end|>`); the pieces must concatenate to the tokens of the whole text and of
apply_chat_template(tokenize=True). Message content that holds special-token text is refused. A conversation that
fails a check raises ValueError (the builder skips it).
"""

import hashlib
from pathlib import Path

ROLES = ("system", "user", "assistant")
ASSISTANT_OPEN = "<|im_assistant|>assistant<|im_middle|><think></think>"
TURN_CLOSE = "<|im_end|>"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ids(encoded) -> list[int]:
    return list(encoded["input_ids"] if hasattr(encoded, "keys") else encoded)


class KimiK25LossMaskGenerator:
    def __init__(self, tokenizer, tokenizer_path: str | Path | None = None):
        template = getattr(tokenizer, "chat_template", None)
        if not isinstance(template, str):
            raise ValueError("the tokenizer has no chat template")
        self.tokenizer = tokenizer
        probe = self.render([{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}])
        if probe != "<|im_user|>user<|im_middle|>x" + TURN_CLOSE + ASSISTANT_OPEN + "y" + TURN_CLOSE:
            raise ValueError("the tokenizer's chat template does not render Kimi K2.5/K2.6 turns")
        special = {int(i) for i in getattr(tokenizer, "added_tokens_decoder", {})}
        tiktoken_specials = getattr(tokenizer, "special_tokens", None)   # Kimi: all 256 reserved ids, by name
        if isinstance(tiktoken_specials, dict):
            special |= {int(i) for i in tiktoken_specials.values() if isinstance(i, int)}
        self.special_ids = special
        self.close_ids = self.encode(TURN_CLOSE)
        if len(self.close_ids) != 1 or self.close_ids[0] not in special:
            raise ValueError("<|im_end|> is not a single special token")
        path = Path(tokenizer_path) / "chat_template.jinja" if tokenizer_path else None
        self.template_file_sha256 = _sha256(path.read_bytes()) if path and path.is_file() else None
        self.template_sha256 = _sha256(template.encode())

    def manifest(self) -> dict:
        return {"type": "kimi_k25", "chat_template_sha256": self.template_sha256,
                "chat_template_file_sha256": self.template_file_sha256,
                "supervised": "assistant content and its closing <|im_end|>; the <|im_assistant|>assistant<|im_middle|> "
                              "header and the empty <think></think> block are context"}

    def render(self, messages: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    def encode(self, text: str) -> list[int]:
        return _ids(self.tokenizer(text, add_special_tokens=False))

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
            if piece != ASSISTANT_OPEN + message["content"] + TURN_CLOSE:
                raise ValueError(f"unexpected assistant rendering {piece[:60]!r}...{piece[-16:]!r}")
            spans.append((len(before) + len(ASSISTANT_OPEN), len(through)))
        return spans

    def get_loss_mask(self, messages: list[dict], tools: list[dict] | None = None) -> tuple[list[int], list[int]]:
        if tools:
            raise ValueError("tool definitions are not supported by the kimi_k25 mask")
        messages = [{"role": message.get("role"), "content": message.get("content")} for message in messages]
        for message in messages:
            if message["role"] not in ROLES:
                raise ValueError(f"unsupported role {message['role']!r} (system, user and assistant only)")
            if not isinstance(message["content"], str):
                raise ValueError("message content must be a string")
            if self.special_ids.intersection(self.encode(message["content"])):
                raise ValueError("message content holds special-token text")
        if not any(message["role"] == "assistant" for message in messages):
            raise ValueError("conversation has no assistant turn")

        text = self.render(messages)
        spans = self.target_spans(messages, text)
        tokens, mask, cursor = [], [], 0
        for start, end in spans:
            context = self.encode(text[cursor:start])
            target = self.encode(text[start:end])
            if target[-1:] != self.close_ids:
                raise ValueError("an assistant turn does not end in <|im_end|>")
            tokens += context + target
            mask += [0] * len(context) + [1] * len(target)
            cursor = end
        tail = self.encode(text[cursor:])
        tokens += tail
        mask += [0] * len(tail)
        if tokens != self.encode(text):
            raise ValueError("tokenizing the rendered text in pieces differs from tokenizing it whole")
        encoded = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                                                     return_dict=True)
        if _ids(encoded) != tokens:
            raise ValueError("tokenizing the rendered text differs from apply_chat_template(tokenize=True)")
        return tokens, mask
