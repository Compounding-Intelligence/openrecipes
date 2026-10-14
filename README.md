# OpenRecipes

Tuned, verified LLM training recipes: <https://openrecipes.ai>

OpenRecipes publishes one YAML file per training recipe for a model and GPU type: the parallel layout, the command as
run, the measured tokens/s per GPU and cost, the framework versions and the correctness evidence. Every
recipe is measured one way (8,192-token SFT rows, global batch 128, BF16, steps 4-13 of a 20-step
run) and checked against the recipe it was tuned from, so recipes for the same model and GPU type compare directly.
The current recipes were found by ATTUNE ([paper](https://arxiv.org/abs/2610.17528)).

| Path | Contents |
|---|---|
| `recipes/<model-slug>/<gpu>/<recipe>.yaml` | the recipes; the schema is `schema/recipe.schema.yaml` |
| `models.yaml`, `gpus.yaml` | the model catalog, and each GPU type's peak and hourly rate |
| `data/` | builds and checks the training rows; the slime rollout adapter |
| `patches/` | code some recipes need beyond the stock slime stack |
| `tools/` | the validator, the site generator and the active-parameter counter |
| `site/` | the generated website (build output, committed) |

## Build

```bash
pip install -r requirements.txt     # PyYAML; the tests also need pytest
python3 tools/validate.py           # check every recipe
python3 tools/build.py              # write site/
python3 tools/build.py --check      # exit 1 if a recipe is invalid or site/ is stale
python3 -m pytest -q tests
python3 -m http.server -d site --bind 127.0.0.1 8000   # preview
```

Cloudflare serves the committed `site/` directory. With Workers, `wrangler.jsonc` points the static assets at
`site/`: leave the build command empty and deploy with `npx wrangler deploy`. With Pages, use output directory `site`
and an empty build command. To rebuild on Cloudflare instead, use `pip install -r requirements.txt && python3
tools/build.py` as the build command.

## Submit a recipe

Open a pull request with one recipe file and the rebuilt `site/`. <https://openrecipes.ai/submit.html> gives the
steps, the template and the fields.

## License

Apache-2.0 (`LICENSE`), Copyright 2026 Compounding Intelligence; see `NOTICE` and `THIRD_PARTY_NOTICES`.
