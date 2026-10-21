# SPDX-License-Identifier: Apache-2.0
"""tools/validate.py: the schema interpreter, the cross-field rules, and every recipe in the repository."""
import copy
import datetime
import shutil
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import validate  # noqa: E402

SAMPLE = "recipes/qwen3.5-122b-a10b/h200/attune-tp1-pp4-ep8-rc2.yaml"


def test_every_recipe_in_the_repository_is_valid():
    recipes, models, gpus, rates, errs = validate.validate_all(ROOT)
    assert errs == []
    assert len(recipes) == 56
    assert len({(r["model"], r["gpu"]["type"]) for r in recipes.values()}) == 32
    assert len(models) == 40 and set(gpus) == {"H100", "H200", "B200"}


@pytest.mark.parametrize("value,schema,needle", [
    (None, {"type": "object", "required": ["a"]}, "expected object"),
    ({}, {"type": "object", "required": ["a"]}, "missing required field 'a'"),
    ({"a": 1, "b": 2}, {"type": "object", "properties": {"a": {}}, "additionalProperties": False}, "unknown field 'b'"),
    (True, {"type": "integer"}, "expected integer"),
    (1.5, {"type": "integer"}, "expected integer"),
    (0, {"type": "number", "exclusiveMinimum": 0}, "must be above 0"),
    ("x", {"enum": ["a", "b"]}, "is not one of"),
    ("abc", {"type": "string", "pattern": "^[0-9]+$"}, "does not match"),
    ("  ", {"type": "string", "minLength": 1}, "must not be empty"),
    ([1], {"type": "array", "minItems": 2}, "at least 2 items"),
    ("2026-13-01", {"type": "date"}, "expected date"),
    (2, {"const": 1}, "must be 1"),
])
def test_schema_interpreter_rejects(value, schema, needle):
    errs = validate.check_schema(value, schema)
    assert errs and needle in errs[0], errs


def test_schema_interpreter_accepts_dates_nulls_and_open_maps():
    assert validate.check_schema(datetime.date(2026, 10, 13), {"type": "date"}) == []
    assert validate.check_schema("2026-10-13", {"type": "date"}) == []
    assert validate.check_schema(None, {"type": ["integer", "null"], "minimum": 1}) == []
    assert validate.check_schema({"X": "1"}, {"type": "object", "additionalProperties": {"type": "string"}}) == []
    assert validate.check_schema({"X": 1}, {"type": "object", "additionalProperties": {"type": "string"}})


@pytest.fixture
def repo(tmp_path):
    """A minimal repository: the catalog files, the schema, patches/ and one recipe."""
    for name in ("models.yaml", "gpus.yaml"):
        shutil.copy(ROOT / name, tmp_path / name)
    shutil.copytree(ROOT / "schema", tmp_path / "schema")
    shutil.copytree(ROOT / "patches", tmp_path / "patches")
    dst = tmp_path / SAMPLE
    dst.parent.mkdir(parents=True)
    shutil.copy(ROOT / SAMPLE, dst)
    return tmp_path


def write(root, rel, recipe):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(recipe, sort_keys=False))
    return p


def errors_for(root, mutate, rel=SAMPLE):
    r = copy.deepcopy(validate.load_yaml(ROOT / SAMPLE))
    mutate(r)
    shutil.rmtree(root / "recipes")
    write(root, rel, r)
    return validate.validate_all(root)[4]


def test_fixture_recipe_is_valid(repo):
    recipes, _, _, _, errs = validate.validate_all(repo)
    assert errs == [] and list(recipes) == [SAMPLE]


@pytest.mark.parametrize("mutate,needle", [
    (lambda r: r["layout"].update(dp=4), "layout.dp 4 must equal"),
    (lambda r: r["layout"].update(ep=64), "must be a multiple of ep x etp"),
    (lambda r: r["result"].update(usd_per_million_tokens=0.80), "usd_per_million_tokens 0.8 must be 0.90"),
    (lambda r: r["result"].update(mfu=0.2), "result.mfu 0.2 must be 0.1340"),
    (lambda r: r["result"]["active_params"].update(count=None), "result.mfu must be null"),
    (lambda r: r["result"].update(window_tokens=1), "window_tokens 1 must be 15728640"),
    (lambda r: r["result"].update(steps_per_run=13), "below the minimum 14"),
    (lambda r: r["layout"].update(rows_per_microbatch=2), "rows_per_microbatch 2 must be"),
    (lambda r: r["layout"].update(sequence_parallel=True), "sequence_parallel is true at tp 1"),
    (lambda r: r.update(patches=["patches/missing.py"]), "patches/missing.py does not exist"),
    (lambda r: r.update(model="Qwen9-1T"), "is not in models.yaml"),
    (lambda r: r["gpu"].update(memory_gb=80), "memory_gb 80 differs"),
    (lambda r: r["correctness"]["convergence"].update(result="fail"), "status verified needs every check to pass"),
    (lambda r: r.update(notes=""), "notes: must not be empty"),
    (lambda r: r.pop("run"), "missing required field 'run'"),
    (lambda r: r.update(extra=1), "unknown field 'extra'"),
    (lambda r: r.update(author={"name": "ATTUNE"}), "author needs github, or name and url"),
])
def test_cross_field_rules(repo, mutate, needle):
    errs = errors_for(repo, mutate)
    assert any(needle in e for e in errs), errs


def test_path_must_match_model_gpu_and_author(repo):
    errs = errors_for(repo, lambda r: None, rel="recipes/qwen3.5-122b-a10b/b200/attune-tp1-pp4-ep8-rc2.yaml")
    assert any("must be the GPU type 'h200'" in e for e in errs), errs
    errs = errors_for(repo, lambda r: None, rel="recipes/qwen3.5-122b/h200/attune-tp1-pp4-ep8-rc2.yaml")
    assert any("must be the model's slug" in e for e in errs), errs
    errs = errors_for(repo, lambda r: None, rel="recipes/qwen3.5-122b-a10b/h200/someone-tp1.yaml")
    assert any("must start with the author's handle 'attune-'" in e for e in errs), errs


def test_github_author_and_notes_are_allowed(repo):
    def mutate(r):
        r["author"] = {"github": "octo-cat"}
        r["notes"] = "Extra notes."
    errs = errors_for(repo, mutate, rel="recipes/qwen3.5-122b-a10b/h200/octo-cat-tp1-pp4-ep8.yaml")
    assert errs == []


def test_invalid_yaml_is_reported(repo):
    (repo / SAMPLE).write_text("model: [unclosed\n")
    errs = validate.validate_all(repo)[4]
    assert errs and "not valid YAML" in errs[0]


def test_catalog_problems_are_reported(repo):
    text = (repo / "models.yaml").read_text().replace("slug: gpt-oss-20b", "slug: GPT")
    (repo / "models.yaml").write_text(text)
    errs = validate.validate_all(repo)[4]
    assert any("slug 'GPT' must be the lower-cased model name" in e for e in errs), errs


def test_cli_exit_codes(repo, capsys):
    assert validate.main([str(ROOT / SAMPLE)]) == 0
    bad = write(repo, "recipes/qwen3.5-122b-a10b/h200/attune-bad.yaml", {"model": "x"})
    assert validate.main([str(bad)]) == 1
    assert "missing required field" in capsys.readouterr().out


def test_verify_rows_reads_the_dataset_build_from_a_recipe():
    sys.path.insert(0, str(ROOT / "data"))
    import verify_rows

    want = verify_rows.expected_build(ROOT / SAMPLE)
    assert want == {"rows": 6400, "row_tokens": 8192,
                    "parquet_sha256": "f215bdc6aba60d6d0dff4471eaef805765cb393e40b95762190535376491ab01",
                    "rows_sha256": "7a837641dcde9ebcac9d3ed5ed84ddd941fc9bfc173606cc701e9a3f51c307f1"}
