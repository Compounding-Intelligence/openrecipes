# Patches

Code a recipe needs beyond the stock slime or miles stack. A recipe that depends on a file here lists it under
`patches:`; the rest of the recipe runs on the unmodified training image.

| File | Contents | Recipes |
|---|---|---|
| `attune_gpt_oss_yarn.py` | GPT-OSS layer spec with the Hugging Face model's YaRN RoPE, loaded with `--spec attune_gpt_oss_yarn get_gpt_oss_yarn_spec` from this directory on `PYTHONPATH` | gpt-oss-20b on H100, H200 and B200 |
| `fla_blackwell_guard.py` | Restricts FLA's gated-delta-rule kernels on Blackwell to the Triton autotune configs that FLA main keeps there (`chunk_gated_delta_rule_fwd_kernel_h_blockdim64` at 2 warps; below Triton 3.8, no 8-warp BK=BV=128 `chunk_fwd_kernel_o` and at most 4 warps for `chunk_bwd_kernel_dv_local`). Every training process on every node imports it before FLA: put this directory on `PYTHONPATH` in each container and Ray worker, with a `sitecustomize.py` on the same path that runs `import fla_blackwell_guard`. Each process that loads FLA's kernels prints a `[fla-guard]` line | Qwen3.5-122B-A10B and Qwen3.5-397B-A17B on B200 |

`attune_gpt_oss_yarn.py` comes from ATTUNE and keeps its copyright header (see `../NOTICE`).
