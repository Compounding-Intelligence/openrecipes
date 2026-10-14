#!/usr/bin/env python3
# Copyright 2026 Compounding Intelligence
# SPDX-License-Identifier: Apache-2.0
"""Validate OpenRecipes recipes against schema/recipe.schema.yaml and the cross-field rules it lists.

    python3 tools/validate.py                 # every recipe under recipes/, plus models.yaml and gpus.yaml
    python3 tools/validate.py recipes/x/h200/y.yaml ...

Prints one line per problem and exits 1 if there is any. Needs PyYAML only.
"""
import argparse
import datetime
import math
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---- a small JSON Schema interpreter: the keywords schema/recipe.schema.yaml uses ----------------------------------

def _is_type(value, name):
    if name == "null":
        return value is None
    if name == "boolean":
        return isinstance(value, bool)
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    if name == "string":
        return isinstance(value, str)
    if name == "object":
        return isinstance(value, dict)
    if name == "array":
        return isinstance(value, list)
    if name == "date":
        if isinstance(value, datetime.date) and not isinstance(value, datetime.datetime):
            return True
        if isinstance(value, str) and DATE.match(value):
            try:
                datetime.date.fromisoformat(value)
                return True
            except ValueError:
                return False
        return False
    raise ValueError("schema uses unknown type %r" % name)


def check_schema(value, schema, where="$"):
    """Errors of value against schema, as 'where: message' strings."""
    errs = []
    if "const" in schema and value != schema["const"]:
        return ["%s: must be %r" % (where, schema["const"])]
    if "enum" in schema and value not in schema["enum"]:
        return ["%s: %r is not one of %s" % (where, value, schema["enum"])]
    if "type" in schema:
        types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_is_type(value, t) for t in types):
            return ["%s: expected %s, got %s" % (where, " or ".join(types), type(value).__name__)]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errs.append("%s: %r is below the minimum %r" % (where, value, schema["minimum"]))
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            errs.append("%s: %r must be above %r" % (where, value, schema["exclusiveMinimum"]))
        if "maximum" in schema and value > schema["maximum"]:
            errs.append("%s: %r is above the maximum %r" % (where, value, schema["maximum"]))
    if isinstance(value, str):
        if "minLength" in schema and len(value.strip()) < schema["minLength"]:
            errs.append("%s: must not be empty" % where)
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errs.append("%s: %r does not match %s" % (where, value, schema["pattern"]))
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errs.append("%s: needs at least %d items" % (where, schema["minItems"]))
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errs.append("%s: allows at most %d items" % (where, schema["maxItems"]))
        if "items" in schema:
            for i, item in enumerate(value):
                errs += check_schema(item, schema["items"], "%s[%d]" % (where, i))
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                errs.append("%s: missing required field %r" % (where, key))
        extra = schema.get("additionalProperties", True)
        for key, item in value.items():
            if not isinstance(key, str):
                errs.append("%s: key %r is not a string" % (where, key))
            elif key in props:
                errs += check_schema(item, props[key], "%s.%s" % (where, key))
            elif extra is False:
                errs.append("%s: unknown field %r" % (where, key))
            elif isinstance(extra, dict):
                errs += check_schema(item, extra, "%s.%s" % (where, key))
    return errs


# ---- catalog files ----------------------------------------------------------------------------------------------

def load_catalog(root=ROOT):
    """(models, gpus, rates, errors): models by name, gpus by type, the rate basis, and problems in the two files."""
    errs = []
    models, slugs = {}, set()
    for i, m in enumerate((load_yaml(Path(root) / "models.yaml") or {}).get("models") or []):
        where = "models.yaml: models[%d]" % i
        if not isinstance(m, dict) or not {"family", "model", "slug", "hf"} <= set(m) <= {"family", "model", "slug", "hf", "notes"}:
            errs.append("%s: needs family, model, slug and hf, and optionally notes" % where)
            continue
        if not all(isinstance(m[k], str) and m[k] for k in m):
            errs.append("%s: every field is a non-empty string" % where)
            continue
        if m["slug"] != m["model"].lower() or not re.match(r"^[a-z0-9][a-z0-9.-]*$", m["slug"]):
            errs.append("%s: slug %r must be the lower-cased model name, of [a-z0-9.-]" % (where, m["slug"]))
        if m["model"] in models or m["slug"] in slugs:
            errs.append("%s: %s is listed twice" % (where, m["model"]))
        models[m["model"]] = m
        slugs.add(m["slug"])
    if not models:
        errs.append("models.yaml: no models")
    data = load_yaml(Path(root) / "gpus.yaml") or {}
    gpus = {}
    need = {"type", "name", "memory_gb", "peak_bf16_tflops", "instance", "usd_per_gpu_hour", "rate_source"}
    for i, g in enumerate(data.get("gpus") or []):
        if not isinstance(g, dict) or set(g) != need:
            errs.append("gpus.yaml: gpus[%d] needs exactly %s" % (i, ", ".join(sorted(need))))
            continue
        for k in ("memory_gb", "peak_bf16_tflops", "usd_per_gpu_hour"):
            if not _is_type(g[k], "number") or g[k] <= 0:
                errs.append("gpus.yaml: gpus[%d].%s must be a positive number" % (i, k))
        if not (isinstance(g["rate_source"], str) and g["rate_source"].startswith("https://")):
            errs.append("gpus.yaml: gpus[%d].rate_source must be an https:// link" % i)
        gpus[g["type"]] = g
    rates = data.get("rates") or {}
    if not (isinstance(rates, dict) and rates.get("basis") and _is_type(rates.get("fetched"), "date")):
        errs.append("gpus.yaml: rates needs basis and a fetched date")
    return models, gpus, rates, errs


# ---- one recipe -------------------------------------------------------------------------------------------------

def usd_per_million(rate, tps):
    return rate / (tps * 3600) * 1e6


def mfu_of(active, tps, peak_tflops):
    return 6 * active * tps / (peak_tflops * 1e12)


def author_handle(author):
    return (author.get("github") or author.get("name") or "").lower()


def semantic_errors(rel, r, models, gpus, root=ROOT):
    """Cross-field checks for a recipe that already matches the schema."""
    errs = []
    parts = Path(rel).parts
    m = models.get(r["model"])
    if m is None:
        errs.append("model %r is not in models.yaml" % r["model"])
    if len(parts) != 4 or parts[0] != "recipes" or not parts[3].endswith(".yaml"):
        errs.append("path must be recipes/<model-slug>/<gpu>/<recipe-slug>.yaml")
    else:
        if m is not None and parts[1] != m["slug"]:
            errs.append("directory %r must be the model's slug %r" % (parts[1], m["slug"]))
        if parts[2] != r["gpu"]["type"].lower():
            errs.append("directory %r must be the GPU type %r" % (parts[2], r["gpu"]["type"].lower()))
        stem = parts[3][:-5]
        if not re.match(r"^[a-z0-9][a-z0-9.-]*$", stem):
            errs.append("recipe slug %r must be of [a-z0-9.-]" % stem)
        handle = author_handle(r["author"])
        if handle and not stem.startswith(handle + "-"):
            errs.append("recipe slug %r must start with the author's handle %r" % (stem, handle + "-"))
    a = r["author"]
    if not (a.get("github") or (a.get("name") and a.get("url"))):
        errs.append("author needs github, or name and url")
    g = gpus.get(r["gpu"]["type"])
    lay, res, wl = r["layout"], r["result"], r["workload"]
    n = r["gpu"]["count"]
    if g is None:
        errs.append("gpu.type %r is not in gpus.yaml" % r["gpu"]["type"])
    elif r["gpu"]["memory_gb"] != g["memory_gb"]:
        errs.append("gpu.memory_gb %r differs from gpus.yaml (%r)" % (r["gpu"]["memory_gb"], g["memory_gb"]))
    model_parallel = lay["tp"] * lay["pp"] * lay["cp"]
    if n % model_parallel or lay["dp"] != n // model_parallel:
        errs.append("layout.dp %d must equal gpu.count / (tp x pp x cp) = %d / %d" % (lay["dp"], n, model_parallel))
    if (n // lay["pp"]) % (lay["ep"] * lay["etp"]):
        errs.append("GPUs per pipeline stage (%d) must be a multiple of ep x etp (%d)"
                    % (n // lay["pp"], lay["ep"] * lay["etp"]))
    if wl["global_batch"] % lay["dp"]:
        errs.append("global_batch %d must divide by dp %d" % (wl["global_batch"], lay["dp"]))
    if lay["sequence_parallel"] and lay["tp"] == 1:
        errs.append("layout.sequence_parallel is true at tp 1")
    if lay["max_tokens_per_gpu"] is not None:
        want = lay["max_tokens_per_gpu"] * lay["cp"] // wl["row_tokens"]
        if lay["rows_per_microbatch"] != want:
            errs.append("layout.rows_per_microbatch %d must be floor(max_tokens_per_gpu x cp / row_tokens) = %d"
                        % (lay["rows_per_microbatch"], want))
    build = wl["dataset"]["build"]
    if build["row_tokens"] != wl["row_tokens"]:
        errs.append("workload.dataset.build.row_tokens differs from workload.row_tokens")
    tps = res["tokens_per_s_per_gpu"]
    if g is not None:
        usd = usd_per_million(g["usd_per_gpu_hour"], tps)
        if abs(round(usd, 2) - res["usd_per_million_tokens"]) > 1e-9:
            errs.append("result.usd_per_million_tokens %r must be %.2f (rate %r / (tokens/s/GPU x 3600) x 1e6)"
                        % (res["usd_per_million_tokens"], usd, g["usd_per_gpu_hour"]))
        count = res["active_params"]["count"]
        if count is None:
            if res["mfu"] is not None or not res["mfu_note"]:
                errs.append("result.mfu must be null, with mfu_note saying why, when active_params.count is null")
        else:
            mfu = mfu_of(count, tps, g["peak_bf16_tflops"])
            if res["mfu"] is None or abs(res["mfu"] - mfu) > 1e-4:
                errs.append("result.mfu %r must be %.4f (6 x N x tokens/s/GPU / peak)" % (res["mfu"], mfu))
    w0, w1 = res["window"]
    if not w0 < w1 < res["steps_per_run"]:
        errs.append("result.window %r must lie inside the %d-step run" % (res["window"], res["steps_per_run"]))
    want = (w1 - w0 + 1) * wl["global_batch"] * wl["row_tokens"]
    if res["window_tokens"] != want:
        errs.append("result.window_tokens %d must be %d" % (res["window_tokens"], want))
    for p in r["patches"]:
        if ".." in Path(p).parts or not (Path(root) / p).is_file():
            errs.append("patches: %s does not exist" % p)
    c = r["correctness"]
    if r["status"] == "verified":
        failed = [k for k, v in (("result", c["result"]), ("configuration", c["configuration"]),
                                 ("convergence", c["convergence"]["result"])) if v != "pass"]
        failed += ["component test %r" % t["test"] for t in c["component_tests"] if t["result"] != "pass"]
        if failed:
            errs.append("status verified needs every check to pass; failed: %s" % ", ".join(failed))
    return errs


def validate_recipe(path, models, gpus, schema, root=ROOT):
    """(recipe or None, errors) for one file."""
    path = Path(path)
    try:
        rel = path.resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        rel = path.as_posix()
    try:
        r = load_yaml(path)
    except yaml.YAMLError as e:
        return None, ["%s: not valid YAML: %s" % (rel, str(e).splitlines()[0])]
    errs = check_schema(r, schema)
    if not errs:
        errs = semantic_errors(rel, r, models, gpus, root)
    return (r if not errs else None), ["%s: %s" % (rel, e) for e in errs]


def recipe_paths(root=ROOT):
    return sorted(Path(root).glob("recipes/**/*.yaml")) + sorted(Path(root).glob("recipes/**/*.yml"))


def validate_all(root=ROOT, paths=None):
    """({relative path: recipe}, models, gpus, rates, errors) for every recipe (or the given paths)."""
    root = Path(root)
    models, gpus, rates, errs = load_catalog(root)
    schema = load_yaml(root / "schema" / "recipe.schema.yaml")
    recipes = {}
    for p in paths or recipe_paths(root):
        p = Path(p)
        if p.suffix == ".yml":
            errs.append("%s: use the .yaml extension" % p)
            continue
        r, e = validate_recipe(p, models, gpus, schema, root)
        errs += e
        if r is not None:
            recipes[p.resolve().relative_to(root.resolve()).as_posix()] = r
    if paths is None and not recipes and not errs:
        errs.append("recipes/: no recipes")
    return recipes, models, gpus, rates, errs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="*", help="recipe files (default: every recipe under recipes/)")
    a = ap.parse_args(argv)
    recipes, _, _, _, errs = validate_all(ROOT, [Path(p) for p in a.paths] or None)
    for e in errs:
        print(e)
    print("%d recipe(s) valid, %d problem(s)" % (len(recipes), len(errs)), file=sys.stderr)
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
