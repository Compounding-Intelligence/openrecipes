# Patches

Code a recipe needs beyond the stock slime or miles stack. A recipe that depends on a file here lists it under
`patches:`; the rest of the recipe runs on the unmodified training image.

| File | Contents | Recipes |
|---|---|---|
| `attune_gpt_oss_yarn.py` | GPT-OSS layer spec with the Hugging Face model's YaRN RoPE, loaded with `--spec attune_gpt_oss_yarn get_gpt_oss_yarn_spec` from this directory on `PYTHONPATH` | gpt-oss-20b on H100, H200 and B200 |

`attune_gpt_oss_yarn.py` comes from ATTUNE and keeps its copyright header (see `../NOTICE`).
