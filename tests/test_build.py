# Copyright 2026 Compounding Intelligence
# SPDX-License-Identifier: Apache-2.0
"""tools/build.py: the pages it writes, their content, determinism and the --check mode."""
import json
import re
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import build  # noqa: E402


@pytest.fixture(scope="module")
def files():
    out, errs = build.render(ROOT)
    assert errs == []
    return {k: v.decode("utf-8") if k.endswith((".html", ".json", ".xml", ".txt", ".css", ".js")) else v
            for k, v in out.items()}


def test_pages(files):
    pairs = [k for k in files if re.match(r"^[a-z0-9.-]+/(h100|h200|b200)/index\.html$", k)]
    assert len(pairs) == 19
    for page in ("index.html", "protocol.html", "submit.html", "404.html", "sitemap.xml",
                 "robots.txt", "data.json", "style.css", "app.js", "favicon.svg", "_headers", "LICENSE.txt", "NOTICE.txt",
                 "schema/recipe.schema.yaml"):
        assert page in files, page
    assert len([k for k in files if k.startswith("recipes/")]) == 42


def test_build_is_deterministic():
    a, _ = build.render(ROOT)
    b, _ = build.render(ROOT)
    assert a == b


def test_pair_page_follows_design_a(files):
    page = files["qwen3.5-122b-a10b/h200/index.html"]
    assert "<h1>Training Qwen3.5-122B-A10B on H200</h1>" in page
    assert '<span class="v hi">2,453.7</span><span class="gauge"' in page and page.count('class="gauge"') == 3
    assert "Correctness</th>" not in page and "<h2>Questions</h2>" not in page and 'class="back"' in page
    ranked = re.findall(r'<details class="card recipe" id="([^"]+)"( open)?>', page)
    assert ranked == [("attune-tp1-pp4-ep8-rc2", " open"), ("attune-tp1-pp4-ep8", ""), ("attune-tp1-pp4-ep8-flex", "")]
    assert page.count("<dt>Convergence check</dt>") == 3 and "baseline" not in page.lower()
    assert page.count('data-copy="file-') == 3 and page.count('data-copy="run-') == 3
    assert page.count('href="../../protocol#run">How to run</a>') == 3
    assert "Qwen3.5-122B-A10B on other GPUs" in page and 'href="../../qwen3.5-122b-a10b/b200/"' in page


def test_notes_appear_once(files):
    page = files["gpt-oss-20b/b200/index.html"]
    assert page.count("<h2>Notes for gpt-oss-20b</h2>") == 1 and page.count("patches/attune_gpt_oss_yarn.py</code>") >= 1
    assert "it was not tuned on B200" in page
    assert "<h2>Notes for" not in files["qwen3.5-122b-a10b/h200/index.html"]
    index = files["index.html"].split('id="models">')[1].split("</section>")[0]
    assert "gauge" not in index and "$" not in index   # the index lists GPU types only; numbers sit in the blocks
    assert files["index.html"].count('class="irow has"') == 9 and "models have recipes" not in files["index.html"]
    assert 'class="tg" aria-pressed="false" hidden>With recipes only</button>' in files["index.html"]
    assert files["index.html"].count('<section class="hud mb"') == 9 and 'data-gpu="H100"' in files["index.html"]


def test_missing_gpu_types_are_grey(files):
    page = files["deepseek-v4-flash/h200/index.html"]
    assert page.count('class="card gpu off"') == 2
    assert files["index.html"].count('<span class="off">') == 3 * 40 - 19


def test_protocol_and_submit(files):
    proto = files["protocol.html"]
    for anchor in ("workload", "measurement", "correctness", "cost", "mfu", "stack"):
        assert 'id="%s"' % anchor in proto
    assert "steps 4&ndash;13" in proto and "$6.88" in proto and "$7.912" in proto and "$14.2416" in proto
    assert "fetched 2026-09-18" in proto
    submit = files["submit.html"]
    assert "amazon-science/attune" not in proto + submit
    assert 'id="template"' in submit and "<code>notes</code>" in submit and 'id="run"' in proto


def test_no_external_resources(files):
    for name, text in files.items():
        if not name.endswith(".html"):
            continue
        for tag in re.findall(r"<(?:script|link|img)\b[^>]*>", text):
            assert not re.search(r'(?:src|href)="(?:https?:)?//', tag) or 'rel="canonical"' in tag, (name, tag)


def test_footer(files):
    assert "Amazon" not in files["index.html"].split("<footer>")[1]
    assert "Copyright Amazon.com, Inc. or its affiliates" in files["NOTICE.txt"]


def test_no_attune_repo_links(files):
    for name, text in files.items():
        if name.endswith((".html", ".json", ".yaml")):
            assert "github.com/amazon-science" not in (text if isinstance(text, str) else text.decode()), name


def test_brand_mark(files):
    for name in ("index.html", "qwen3.5-122b-a10b/h200/index.html", "404.html"):
        assert '<a class="brand" href' in files[name] and build.MARK in files[name], name
    assert b"A10 10 0 1 1 21 7.34" in files["favicon.svg"]


def test_internal_links_use_clean_urls(files):
    for name, text in files.items():
        if name.endswith(".html"):
            assert not re.search(r'href="(?!https?:)[^"]*\.html\b', text), name
    assert 'href="./#models"' in files["protocol.html"] and 'href="../../protocol#run"' in \
        files["qwen3.5-122b-a10b/h200/index.html"] and 'href="/#models"' in files["404.html"]
    assert '<link rel="canonical" href="https://openrecipes.ai/submit">' in files["submit.html"]
    assert "<loc>https://openrecipes.ai/protocol</loc>" in files["sitemap.xml"] and "rankings" not in files["sitemap.xml"]


def test_data_json_and_sitemap(files):
    data = json.loads(files["data.json"])
    assert len(data["recipes"]) == 42 and len(data["models"]) == 40
    best = [r for r in data["recipes"] if r["model"] == "Qwen3.5-122B-A10B" and r["gpu"]["type"] == "H200"]
    assert [r["rank"] for r in best] == [1, 2, 3] and best[0]["result"]["measured"] == "2026-10-13"
    assert files["sitemap.xml"].count("<url>") == 3 + 19
    assert "Sitemap: https://openrecipes.ai/sitemap.xml" in files["robots.txt"]


def test_render_text():
    out = build.render_text("First `a<b>` line\ncontinues.\n\n- one\n  more\n- two https://example.org/x.\n")
    assert out == ('<p>First <code>a&lt;b&gt;</code> line continues.</p>\n'
                   '<ul><li>one more</li><li>two <a href="https://example.org/x">https://example.org/x</a>.</li></ul>')


def test_repository_site_is_up_to_date():
    assert build.main(["--check"]) == 0


@pytest.fixture
def repo(tmp_path):
    for name in ("models.yaml", "gpus.yaml", "LICENSE", "NOTICE"):
        shutil.copy(ROOT / name, tmp_path / name)
    for d in ("schema", "patches", "recipes"):
        shutil.copytree(ROOT / d, tmp_path / d)
    return tmp_path


def test_check_detects_stale_extra_and_invalid(repo, capsys):
    assert build.main(["--root", str(repo), "--check"]) == 1   # no site/ yet
    assert build.main(["--root", str(repo)]) == 0
    assert build.main(["--root", str(repo), "--check"]) == 0
    (repo / "site" / "stray.html").write_text("x")
    assert build.main(["--root", str(repo), "--check"]) == 1
    assert "site/stray.html: not generated" in capsys.readouterr().out
    assert build.main(["--root", str(repo)]) == 0
    assert not (repo / "site" / "stray.html").exists()
    recipe = repo / "recipes/gpt-oss-20b/h100/attune-tp1-pp2-ep4.yaml"
    recipe.write_text(recipe.read_text().replace("tokens_per_s_per_gpu: 6744.63", "tokens_per_s_per_gpu: 6000"))
    assert build.main(["--root", str(repo), "--check"]) == 1
    assert "usd_per_million_tokens" in capsys.readouterr().out
