# Training rows

_Modified for OpenRecipes, 2026-10-14: from ATTUNE's `benchmark/data/README.md` (see `../NOTICE`),
with this repository's paths and names, a recipe YAML instead of an `entry.json` in step 4, and the build-time command
guard note removed._

The recipes in `../recipes/` train on the Tulu 3 SFT mixture, built once per tokenizer into rows of exactly 8,192
tokens. This directory holds what rebuilds and checks those rows.

| File | Purpose |
|---|---|
| `generate_fixed_sft.py` | builds the rows from the source Parquet shards and writes `<output>.manifest.json` beside them |
| `gemma4_mask.py` | the gemma4 loss mask the builder uses with `--loss-mask-type gemma4`, for the Gemma 4 recipes |
| `inkling_mask.py` | the inkling loss mask the builder uses with `--loss-mask-type inkling`, for the Inkling recipes |
| `patch_glm5_mask.py` | adds a glm5 loss mask to slime's `mask_utils.py`, in place; run it before a GLM-5.2 or GLM-4.7-Flash build |
| `patch_gptoss_mask.py` | adds a gpt_oss (harmony) loss mask to slime's `mask_utils.py`, in place; run it before a gpt-oss build |
| `pretokenized_sft_rollout.py` | the slime rollout adapter the recipes use; it feeds each row's stored tokens and mask unchanged |
| `verify_rows.py` | checks a built file against the dataset build a recipe records |

## What the builder does

The builder shuffles the source conversations with `random.Random(seed)`. It renders each one with the tokenizer's
chat template and slime's `MultiTurnLossMaskGenerator`, then concatenates them with an EOS token (mask 0) between
conversations. The stream is cut into consecutive rows of `--sequence-length` tokens. A row with no supervised target
is skipped, and the remainder after the last full row is dropped, so no row is padded. Each row stores its tokens, its
target mask (the token mask shifted by one, `sequence-length - 1` entries) and a `row_hash` in `metadata`. Its
`messages` field holds a fixed one-turn prompt that lets slime's loader initialize. The builder then re-reads the file,
validates every row, and writes the manifest with the source, tokenizer and output hashes.

## Requirements

- Python 3.10 or later, `pyarrow` and `transformers`.
- A slime checkout on `PYTHONPATH`. The builder imports slime's `Dataset`, `MultiTurnLossMaskGenerator` and
  `load_processor`. Upstream slime (<https://github.com/THUDM/slime>) has the `qwen3_5` loss mask; GLM-5.2 and
  GLM-4.7-Flash need `patch_glm5_mask.py` and gpt-oss `patch_gptoss_mask.py`. The `deepseek_v4`, `gemma4` and
  `inkling` masks need no slime generator; the loader check then uses miles when slime is absent. The Gemma 4 and
  Inkling loader checks load the checkpoint's processor, which needs `torch` and `torchvision`.
- The `muse_spark` mask needs a miles checkout at commit `bfdcff0e` and Megatron-LM on `PYTHONPATH`: it registers
  miles' Muse Spark config with transformers to load the checkpoint's tokenizer. That import loads Megatron and
  Transformer Engine, so the build needs a host with a CUDA driver; one GPU is enough.

## Build

Run the build once per tokenizer.

1. Download the source at the pinned revision:

   ```bash
   huggingface-cli download allenai/tulu-3-sft-mixture --repo-type dataset \
     --revision b14afda60f1bbebe55d5d2fa1e4df5042f97f8be --local-dir tulu-3-sft-mixture
   ```

   The six shards in `tulu-3-sft-mixture/data/` should have these sha256 values:

   | Shard | Rows | sha256 |
   |---|---:|---|
   | `train-00000-of-00006.parquet` | 156,558 | `41d4524345443ccca4202df2f01d84d84371afcde36f637178e0e2bcd14f79c7` |
   | `train-00001-of-00006.parquet` | 156,557 | `c612164eabfacb99e37de582bf91712f878602755cf87daaa514bdeebec71f56` |
   | `train-00002-of-00006.parquet` | 156,557 | `e15d19cb441578dacefa251e18023e2612497eaead1191c8b31acebf1a45260b` |
   | `train-00003-of-00006.parquet` | 156,557 | `b9e312d1a0e777a3a64450cb89d2e28116ef96b4f85875157deddc811599be79` |
   | `train-00004-of-00006.parquet` | 156,557 | `44ddad0bc45d272c4be32bd78a1473c0902e153d42178366414e0532ab29dd27` |
   | `train-00005-of-00006.parquet` | 156,557 | `244ec7ec0591fc67553ca0c9f42a0cba2b34a100438369bd323267a93b3b34ca` |

2. Use the model's Hugging Face checkpoint directory as the tokenizer. Its `tokenizer.json` sha256 should equal the
   one in the recipe's `workload.dataset.build.tokenizer`.

3. Build. All Qwen3.5 and Qwen3.6 recipes use one set, built with the Qwen3.5-122B-A10B tokenizer:

   ```bash
   export PYTHONPATH=<slime checkout>:$PYTHONPATH TOKENIZERS_PARALLELISM=false
   python3 -u generate_fixed_sft.py \
     --input tulu-3-sft-mixture/data \
     --output qwen35-tulu3-fixed8192-v1/train.parquet \
     --tokenizer <Qwen3.5-122B-A10B checkpoint> \
     --source-dataset allenai/tulu-3-sft-mixture \
     --source-revision b14afda60f1bbebe55d5d2fa1e4df5042f97f8be \
     --input-key messages --source-key source --loss-mask-type qwen3_5 \
     --sequence-length 8192 --rows 6400 --seed 42 \
     --source-batch-size 2048 --write-batch-size 64 --min-token-entropy-bits 8.0
   ```

   Qwen3.8's tokenizer adds 7 special tokens and its chat template a default system prompt, so the Qwen3.8 recipes
   use their own set: the same command with `--output qwen38-tulu3-fixed8192-v1/train.parquet` and
   `--tokenizer <Qwen3.8-27B checkpoint>`.

   Qwen3-32B has its own tokenizer and trains on its own set, built with slime's `qwen3` mask. The mask supervises
   each assistant turn from the empty `<think></think>` block the template puts before an answer without reasoning
   through its `<|im_end|>` and the newline after it. Use the checkpoint of `Qwen/Qwen3-32B` (revision
   `9216db5781bf21249d130ec9da846c4624c16137`):

   ```bash
   python3 -u generate_fixed_sft.py \
     --input tulu-3-sft-mixture/data \
     --output qwen3-tulu3-fixed8192-v1/train.parquet \
     --tokenizer <Qwen3-32B checkpoint> \
     --source-dataset allenai/tulu-3-sft-mixture \
     --source-revision b14afda60f1bbebe55d5d2fa1e4df5042f97f8be \
     --input-key messages --source-key source --loss-mask-type qwen3 \
     --sequence-length 8192 --rows 6400 --seed 42 \
     --source-batch-size 2048 --write-batch-size 64 --min-token-entropy-bits 8.0
   ```

   The Qwen3-30B-A3B recipes use this set too: at revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, its
   `tokenizer.json`, `tokenizer_config.json`, `vocab.json` and `merges.txt` are byte-identical to Qwen3-32B's.

   The GLM-5.2 recipes use a second set. Patch slime first; the patch edits `slime/utils/mask_utils.py` in the
   checkout on `PYTHONPATH` and does nothing if a glm5 generator is already there:

   ```bash
   python3 -u patch_glm5_mask.py
   python3 -u generate_fixed_sft.py \
     --input tulu-3-sft-mixture/data \
     --output glm52-tulu3-fixed8192-v1/train.parquet \
     --tokenizer <GLM-5.2 checkpoint> \
     --source-dataset allenai/tulu-3-sft-mixture \
     --source-revision b14afda60f1bbebe55d5d2fa1e4df5042f97f8be \
     --input-key messages --source-key source --loss-mask-type glm5 \
     --sequence-length 8192 --rows 12800 --seed 20260828 \
     --source-batch-size 2048 --write-batch-size 64 --min-token-entropy-bits 8.0
   ```

   GLM-4.7-Flash has GLM-5.2's `tokenizer.json`, but its chat template has no reasoning-effort system turn, so its
   recipes use their own set. Patch slime as for GLM-5.2, then build from the checkpoint of `zai-org/GLM-4.7-Flash`
   (revision `7dd20894a642a0aa287e9827cb1a1f7f91386b67`):

   ```bash
   python3 -u patch_glm5_mask.py
   python3 -u generate_fixed_sft.py \
     --input tulu-3-sft-mixture/data \
     --output glm47flash-tulu3-fixed8192-v1/train.parquet \
     --tokenizer <GLM-4.7-Flash checkpoint> \
     --source-dataset allenai/tulu-3-sft-mixture \
     --source-revision b14afda60f1bbebe55d5d2fa1e4df5042f97f8be \
     --input-key messages --source-key source --loss-mask-type glm5 \
     --sequence-length 8192 --rows 6400 --seed 42 \
     --source-batch-size 2048 --write-batch-size 64 --min-token-entropy-bits 8.0
   ```

   Nemotron-3-Nano-30B-A3B's chat template marks turns with `<|im_start|>` and `<|im_end|>`, so slime's `qwen3_5` mask
   applies: it supervises each assistant turn through its `<|im_end|>` and the newline after it, including the empty
   `<think></think>` block the template puts before an answer without reasoning. Use the checkpoint of
   `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (revision `bf77c3174f68ad409e1c2aa60daeb46e32d1c606`):

   ```bash
   python3 -u generate_fixed_sft.py \
     --input tulu-3-sft-mixture/data \
     --output nemotron3nano-tulu3-fixed8192-v1/train.parquet \
     --tokenizer <Nemotron-3-Nano-30B-A3B checkpoint> \
     --source-dataset allenai/tulu-3-sft-mixture \
     --source-revision b14afda60f1bbebe55d5d2fa1e4df5042f97f8be \
     --input-key messages --source-key source --loss-mask-type qwen3_5 \
     --sequence-length 8192 --rows 6400 --seed 42 \
     --source-batch-size 2048 --write-batch-size 64 --min-token-entropy-bits 8.0
   ```

   The Nemotron-3-Super-120B-A12B recipes use `nemotron3super-tulu3-fixed8192-v1`, the same command with
   `--output nemotron3super-tulu3-fixed8192-v1/train.parquet` and the checkpoint of
   `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` (revision `2dc98e2afe4face0e4ce40972a915c45368bd34a`) as
   `--tokenizer`. Its `tokenizer.json` parses to the same JSON as Nano's, and its chat template differs from Nano's
   only by an optional `low_effort` flag that is off by default, so the set is byte-identical to
   `nemotron3nano-tulu3-fixed8192-v1`.

   DeepSeek-V4 ships no chat template: `--loss-mask-type deepseek_v4` (`dsv4_mask.py`) renders each conversation with
   the checkout's official `encoding/encoding_dsv4.py` in chat mode and supervises the assistant turns. It runs that
   file, so it needs `--trust-remote-code`; use a checkout of `deepseek-ai/DeepSeek-V4-Flash` (revision
   `60d8d70770c6776ff598c94bb586a859a38244f1`), which carries `encoding/`:

   ```bash
   python3 -u generate_fixed_sft.py \
     --input tulu-3-sft-mixture/data \
     --output dsv4-tulu3-fixed8192-v1/train.parquet \
     --tokenizer <DeepSeek-V4-Flash checkout> \
     --source-dataset allenai/tulu-3-sft-mixture \
     --source-revision b14afda60f1bbebe55d5d2fa1e4df5042f97f8be \
     --loss-mask-type deepseek_v4 --trust-remote-code \
     --sequence-length 8192 --rows 6400 --seed 42
   ```

   The gpt-oss recipes use one set, built with the gpt-oss-20b tokenizer (revision
   `6cee5e81ee83917806bbde320786a8fb61efebee`), whose tokenizer files and chat template are byte-identical to
   gpt-oss-120b's. Patch slime first; the patch adds the `gpt_oss` mask, which supervises each assistant turn from
   after `<|start|>assistant` through its closing `<|end|>`, `<|return|>` or `<|call|>`, freezes the template's date to
   2026-10-01 and skips conversations the harmony template would not render faithfully:

   ```bash
   python3 -u patch_gptoss_mask.py
   python3 -u generate_fixed_sft.py \
     --input tulu-3-sft-mixture/data \
     --output gptoss-tulu3-fixed8192-v1/train.parquet \
     --tokenizer <gpt-oss-20b checkpoint> \
     --source-dataset allenai/tulu-3-sft-mixture \
     --source-revision b14afda60f1bbebe55d5d2fa1e4df5042f97f8be \
     --input-key messages --source-key source --loss-mask-type gpt_oss \
     --sequence-length 8192 --rows 6400 --seed 42 \
     --source-batch-size 2048 --write-batch-size 64 --min-token-entropy-bits 8.0
   ```

   Muse Spark renders conversations with the checkpoint's `chat_template.jinja`, which wraps each assistant message
   in `{% generation %}`. `--loss-mask-type muse_spark` (`muse_spark_mask.py`) takes the assistant messages from those
   spans, checks them against the rendering of growing prefixes of the conversation, and supervises each assistant
   part's content and its end marker (`<|eot|>` or `<|eom|>`); the `<|start|>assistant to=...<|message|>` header is
   context. It pins the template's current date to 2026-10-01 and its reasoning effort to `xhigh`, so the rows do not
   depend on the build date. Use the checkout of `meta-models/Muse-Spark-Open-816B-A42B` (revision
   `0fb02478b02565b2ce7fc57e6051c91172774ac9`):

   ```bash
   export PYTHONPATH=<miles checkout>:<Megatron-LM checkout>:$PYTHONPATH TOKENIZERS_PARALLELISM=false
   python3 -u generate_fixed_sft.py \
     --input tulu-3-sft-mixture/data \
     --output muse-spark-tulu3-fixed8192-v1/train.parquet \
     --tokenizer <Muse-Spark-Open-816B-A42B checkout> \
     --source-dataset allenai/tulu-3-sft-mixture \
     --source-revision b14afda60f1bbebe55d5d2fa1e4df5042f97f8be \
     --input-key messages --source-key source --loss-mask-type muse_spark \
     --sequence-length 8192 --rows 6400 --seed 42 \
     --source-batch-size 2048 --write-batch-size 64 --min-token-entropy-bits 8.0
   ```

   The Gemma 4 recipes use one set, built with the checkpoint of `google/gemma-4-26B-A4B-it` (revision
   `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`); Gemma-4-31B-it's `tokenizer.json`, `tokenizer_config.json` and
   `chat_template.jinja` are byte-identical at revision `842da3794eaa0b77d5f08bae87a17459d91ff475`. The template renders
   an assistant message as a `model` turn: `<|turn>model` and a newline, the content, then `<turn|>` and a newline.
   `--loss-mask-type gemma4` (`gemma4_mask.py`) finds each assistant message's turn by rendering growing prefixes of the
   conversation and supervises the content and the closing `<turn|>` with its newline; the header is context. The
   template merges consecutive assistant messages into one turn, so the mask skips such a conversation (one in this
   build):

   ```bash
   python3 -u generate_fixed_sft.py \
     --input tulu-3-sft-mixture/data \
     --output gemma4-tulu3-fixed8192-v1/train.parquet \
     --tokenizer <Gemma-4-26B-A4B-it checkpoint> \
     --source-dataset allenai/tulu-3-sft-mixture \
     --source-revision b14afda60f1bbebe55d5d2fa1e4df5042f97f8be \
     --input-key messages --source-key source --loss-mask-type gemma4 \
     --sequence-length 8192 --rows 6400 --seed 42 \
     --source-batch-size 2048 --write-batch-size 64 --min-token-entropy-bits 8.0
   ```

   The Inkling-Small recipes use `inkling-tulu3-fixed8192-v1`, built with the checkpoint of
   `thinkingmachines/Inkling-Small` (revision `8cc5877b44d343f88b92086aa1fb72897950f06a`). The template renders each
   message as a role token, a content-type token, the content and `<|end_message|>`, and closes an assistant message with
   `<|content_model_end_sampling|>`, the model's end-of-turn token. `--loss-mask-type inkling` (`inkling_mask.py`) finds
   each assistant answer by rendering growing prefixes of the conversation and supervises what follows the
   `<|message_model|>` generation prompt: `<|content_text|>`, the content, `<|end_message|>` and
   `<|content_model_end_sampling|>`; the header and the other messages are context. It pins the template's
   thinking-effort line to its default, 0.9, and skips a conversation with an empty assistant message (50 in this
   build). The tokenizer defines no EOS token, so the builder puts no separator between conversations; each answered
   conversation already ends with `<|content_model_end_sampling|>`:

   ```bash
   python3 -u generate_fixed_sft.py \
     --input tulu-3-sft-mixture/data \
     --output inkling-tulu3-fixed8192-v1/train.parquet \
     --tokenizer <Inkling-Small checkpoint> \
     --source-dataset allenai/tulu-3-sft-mixture \
     --source-revision b14afda60f1bbebe55d5d2fa1e4df5042f97f8be \
     --input-key messages --source-key source --loss-mask-type inkling \
     --sequence-length 8192 --rows 6400 --seed 42 \
     --source-batch-size 2048 --write-batch-size 64 --min-token-entropy-bits 8.0
   ```

4. Verify against a recipe (needs PyYAML):

   ```bash
   python3 verify_rows.py qwen35-tulu3-fixed8192-v1/train.parquet \
     --recipe ../recipes/qwen3.5-122b-a10b/h100/attune-tp4-pp6-ep8-flex-mt16k.yaml
   ```

   `rows_sha256` is the sha256 of the rows' `row_hash` values in file order, one per line. It matches when every row
   matches, whatever Parquet writer produced the file. `parquet_sha256` matches only a byte-identical file, which also
   needs the same `pyarrow` version. The script exits non-zero on a row error or a `rows_sha256` mismatch.

## Using the rows

Put this directory on `PYTHONPATH` and pass the flags the recipes' commands use:
`--prompt-data <train.parquet> --input-key messages --metadata-key metadata
--rollout-function-path pretokenized_sft_rollout.generate_rollout`. The adapter reads the rows through slime's
global dataset, so the command must not pass `--disable-rollout-global-dataset`. Every row must hold exactly `$AUTOTUNE_SEQ_LEN`
tokens (default 8,192); set it in the environment of every pod. The adapter refuses any other length.

## Limits

- The library versions of the original builds (transformers, tokenizers, pyarrow, slime) were not recorded. A
  different chat template or tokenizer version can change the rows; `rows_sha256` detects it.
- The GLM-5.2 and GLM-4.7-Flash sets were built with `patch_glm5_mask.py` applied to a slime that had no glm5
  generator. With a slime that ships its own, the patch does nothing and the rows can differ.
- The recorded builds ran with the tokenizer's remote code allowed. The Qwen3.5 tokenizer has no custom code (its
  `tokenizer_config.json` has no `auto_map`), so the Qwen3.5 command needs no `--trust-remote-code`. If the GLM-5.2
  checkpoint's `tokenizer_config.json` has an `auto_map`, add `--trust-remote-code` to the GLM-5.2 command;
  `verify_rows.py` shows whether the rows match.
- The gpt-oss set was built with transformers 5.8.1, tokenizers 0.22.2 and pyarrow 23.0.1; rebuilt on two more
  clusters from the same inputs, it matched byte for byte.
- The GLM-4.7-Flash and Nemotron-3-Nano-30B-A3B sets were built with transformers 5.8.1, tokenizers 0.22.2 and pyarrow
  23.0.1, without remote code: neither tokenizer has custom code.
- The Qwen3-32B set was built with transformers 5.8.1, tokenizers 0.22.2 and pyarrow 23.0.1, without remote code: the
  tokenizer has no custom code.
- The Muse Spark set was built with transformers 5.12.1, tokenizers 0.22.2 and pyarrow 25.0.1, the versions of the
  training image.
- The Gemma 4 set was built with transformers 5.8.1, tokenizers 0.22.2 and pyarrow 23.0.1, without remote code: the
  tokenizer has no custom code. Rebuilt with `gemma4_mask.py` from the same inputs and versions, it matched byte for
  byte.
- The Inkling set was built with transformers 5.12.1, tokenizers 0.22.2 and pyarrow 25.0.0, the versions of the
  training image, without remote code: the tokenizer has no custom code. Rebuilt with `inkling_mask.py` from the same
  inputs and versions, it matched byte for byte.
- `row_hash` hashes the tokens as native 32-bit integers, so it assumes a little-endian host (x86-64 or ARM64).
