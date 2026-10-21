# SPDX-License-Identifier: Apache-2.0
"""tools/build.py: the pages it writes, their content, determinism and the --check mode."""
import json
import re
import shutil
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import build  # noqa: E402
ATTUNE = "https://arxiv.org/abs/2610.17528"


@pytest.fixture(scope="module")
def files():
    out, errs = build.render(ROOT)
    assert errs == []
    return {k: v.decode("utf-8") if k.endswith((".html", ".json", ".xml", ".txt", ".css", ".js")) else v
            for k, v in out.items()}


def test_pages(files):
    pairs = [k for k in files if re.match(r"^[a-z0-9.-]+/(h100|h200|b200)/index\.html$", k)]
    assert len(pairs) == 34
    for page in ("index.html", "protocol.html", "submit.html", "404.html", "sitemap.xml",
                 "robots.txt", "data.json", "style.css", "app.js", "favicon.svg", "_headers", "LICENSE.txt", "NOTICE.txt",
                 "schema/recipe.schema.yaml"):
        assert page in files, page
    assert len([k for k in files if k.startswith("recipes/")]) == 58


def test_build_is_deterministic():
    a, _ = build.render(ROOT)
    b, _ = build.render(ROOT)
    assert a == b


def test_pair_page_follows_design_a(files):
    page = files["qwen3.5-122b-a10b/h200/index.html"]
    assert "<h1>Training Qwen3.5-122B-A10B on H200</h1>" in page
    assert '<span class="v hi">2,451.4</span><span class="gauge"' in page and page.count('class="gauge"') == 3
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


def test_missing_gpu_types_are_grey(files):
    page = files["deepseek-v4-flash/h200/index.html"]
    assert page.count('class="card gpu off"') == 2
    assert files["index.html"].count('<span class="none">No recipe yet</span>') == 3 * 19 - 34   # open slots on the cards


@pytest.fixture(scope="module")
def catalog():
    recipes, models, gpus, _, errs = build.validate_all(ROOT)
    assert errs == []
    pairs = build.collect(recipes, models, gpus)
    tuned = [m for m in models.values() if any((m["model"], gt) in pairs for gt in gpus)]
    return {"recipes": recipes, "models": models, "gpus": list(gpus), "pairs": pairs, "tuned": tuned}


def section(page, start):
    return page.split(start, 1)[1].split("</section>", 1)[0]


def test_index_hero_counts_what_has_recipes(files, catalog):
    index = files["index.html"]
    hero = index.split('<div class="big">', 1)[1].split("</header>", 1)[0]
    assert re.findall(r'<div class="ml">([^<]+)</div><span class="num">(\d+)</span>', hero) == [
        ("Recipes", str(len(catalog["recipes"]))), ("Models", str(len(catalog["tuned"]))),
        ("Model&ndash;GPU pairs", str(len(catalog["pairs"])))]
    # no catalog denominator and no count of the models or GPU types still without a recipe
    text = " ".join(re.sub(r"<[^>]+>", " ", index.split("<main", 1)[1]).split())
    assert "catalog" not in text and not re.search(r"\bof %d\b" % len(catalog["models"]), text)
    assert not re.search(r"\d+ (?:\w+ )?(?:models?|GPU types?) (?:have|has|are|is) (?:no|still)", text)
    assert "With recipes only" not in index and 'class="hud idx"' not in index and 'class="seg"' not in index


def test_index_cards(files, catalog):
    index, pairs, gpus = files["index.html"], catalog["pairs"], catalog["gpus"]
    cards = re.findall(r'<article class="hud mc" id="m-([^"]+)">(.*?)</article>', index, re.S)
    assert [slug for slug, _ in cards] == [m["slug"] for m in catalog["tuned"]]   # catalog order, models with recipes
    for (_, card), m in zip(cards, catalog["tuned"]):
        mine = {gt: pairs[(m["model"], gt)] for gt in gpus if (m["model"], gt) in pairs}
        n = sum(len(v) for v in mine.values())
        assert '<span class="cnt">%d recipe%s</span>' % (n, "" if n == 1 else "s") in card
        rows = re.findall(r'<a class="gr" href="([^"]+)"[^>]*><span class="g">(\w+)</span><span class="gauge sm[^"]*"'
                          r'.*?<span class="t( hi)?">([\d,.]+)</span>', card)
        assert [(u, gt, t) for u, gt, _, t in rows] == [
            (build.pair_url(m, gt), gt, build.tps(v[0]["r"]["result"]["tokens_per_s_per_gpu"])) for gt, v in mine.items()]
        assert [hi for *_, hi, _ in rows].count(" hi") >= 1
        assert card.count("No recipe yet") == len(gpus) - len(mine)
        newest = max(build.iso(x["r"]["result"]["measured"]) for v in mine.values() for x in v)
        assert "Newest <b>%s</b>" % newest in card
    for url in re.findall(r'<a class="gr" href="([^"]+)"', index):
        assert url + "index.html" in files
    card = dict(cards)["qwen3.5-122b-a10b"]
    assert '<span class="t hi">2,475.8</span><span class="c">&times;16</span><span class="u">$1.60</span>' in card
    assert '122B parameters &middot; <abbr' in card and "N</abbr> 9.0B active" in card


def test_index_starting_point_gains(files, catalog):
    cards = dict(re.findall(r'<article class="hud mc" id="m-([^"]+)">(.*?)</article>', files["index.html"], re.S))
    assert ('<span class="ml">Starting point</span><span class="nw">3,142.9 &rarr; 6,178.5 <b>+97%</b></span>'
            in cards["glm-4.7-flash"])
    best = [v[0]["r"] for v in catalog["pairs"].values()]
    assert files["index.html"].count('<div class="gl"') == sum(1 for r in best if build.starting_gain(r)) > 0


def test_starting_gain_checks_the_numbers():
    r = {"gpu": {"type": "H200", "count": 8}, "result": {"tokens_per_s_per_gpu": 6178.46}}
    note = ("- First item.\n- Starting point: miles' launcher, `x.py`. Ported to this workload, it measured 3,142.9\n"
            "  tokens/s/GPU; ATTUNE tuned it to 6,178.5 (+97%).\n- Last item.\n")
    gain = {"source": "miles' launcher", "measured": 3142.9, "tuned": 6178.5, "percent": 97}
    assert build.starting_gain(dict(r, notes=note)) == gain
    assert build.starting_gain(dict(r, notes=note.replace("GPU;", "GPU on 8 H200;"))) == gain
    assert build.starting_gain(dict(r, notes=note.replace("GPU;", "GPU on 8 H100;"))) is None    # another GPU type
    assert build.starting_gain(dict(r, notes=note.replace("(+97%)", "(+90%)"))) is None          # percent disagrees
    assert build.starting_gain(dict(r, notes=note.replace("6,178.5", "6,200.0"))) is None        # not the recipe's result
    assert build.starting_gain(dict(r, notes=note.replace("Starting point:", "Before:"))) is None
    assert build.starting_gain(dict(r, notes=None)) is None


def test_index_latest_recipes(files, catalog):
    sec = section(files["index.html"], '<section class="hud latest" id="latest">')
    n = len(catalog["recipes"])
    rows = re.findall(r'<tr( class="x")?>(.*?)</tr>', sec.split("<tbody>", 1)[1], re.S)
    assert len(rows) == n and not re.search(r"<tr[^>]*hidden", sec)   # every row renders, so it works without JS
    assert [bool(x) for x, _ in rows] == [i >= build.LATEST for i in range(n)]
    dates = re.findall(r'data-label="Measured"><span class="cv">([\d-]+)</span>', sec)
    assert len(dates) == n and dates == sorted(dates, reverse=True)
    assert dates[0] == max(build.iso(r["result"]["measured"]) for r in catalog["recipes"].values())
    for url, slug in re.findall(r'<a href="([^"#]+)#([^"]+)">', sec):
        assert 'id="%s"' % slug in files[url + "index.html"]
    assert ('<button type="button" class="tg" aria-pressed="false" data-more="Show all %d recipes" data-less="Show the '
            'newest %d" hidden>Show all %d recipes</button>' % (n, build.LATEST, n)) in sec
    assert '".latest tr.x"' in files["app.js"] and "more.hidden = false;" in files["app.js"]


def test_index_wanted_strip(files, catalog):
    sec = section(files["index.html"], '<section class="hud wanted" id="wanted">')
    tuned = {m["model"] for m in catalog["tuned"]}
    assert re.findall(r'<span class="(?:end )?wp">([^<]+)</span>', sec) == [
        build.esc(m) for m in catalog["models"] if m not in tuned]
    head = sec.split("</div>", 1)[0]
    assert "<h2>Wanted: recipes for these models</h2>" in head and 'href="submit"' in head
    assert not re.search(r"\d", re.sub(r"<[^>]+>", "", head))   # an invitation, not a count


def test_index_without_open_models(repo):
    data = build.load_yaml(repo / "models.yaml")
    have = {p.parts[-3] for p in (repo / "recipes").glob("*/*/*.yaml")}
    data["models"] = [m for m in data["models"] if m["slug"] in have]
    (repo / "models.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    out, errs = build.render(repo)
    assert errs == []
    index = out["index.html"].decode("utf-8")
    assert 'id="wanted"' not in index and index.count('<article class="hud mc"') == len(data["models"])


def test_protocol_and_submit(files):
    proto = files["protocol.html"]
    for anchor in ("workload", "measurement", "correctness", "cost", "mfu", "stack"):
        assert 'id="%s"' % anchor in proto
    assert "steps 5&ndash;19" in proto and "steps 0&ndash;4" in proto.lower() and "$6.88" in proto and "$7.912" in proto and "$14.2416" in proto
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
    assert len(data["recipes"]) == 58 and len(data["models"]) == 40
    best = [r for r in data["recipes"] if r["model"] == "Qwen3.5-122B-A10B" and r["gpu"]["type"] == "H200"]
    assert [r["rank"] for r in best] == [1, 2, 3] and best[0]["result"]["measured"] == "2026-10-13"
    assert files["sitemap.xml"].count("<url>") == 3 + 34
    assert "Sitemap: https://openrecipes.ai/sitemap.xml" in files["robots.txt"]


def test_render_text():
    out = build.render_text("First `a<b>` line\ncontinues.\n\n- one\n  more\n- two https://example.org/x.\n")
    assert out == ('<p>First <code>a&lt;b&gt;</code> line continues.</p>\n'
                   '<ul><li>one more</li><li>two <a href="https://example.org/x">https://example.org/x</a>.</li></ul>')


def test_citation(files):
    cits = build.citations(ROOT)
    assert list(cits) == ["https://arxiv.org/abs/2610.17528"] and cits[ATTUNE].startswith("@article{fan2026attune,\n")
    cff = build.load_yaml(ROOT / "CITATION.cff")
    assert cff["title"] == "OpenRecipes" and "preferred-citation" not in cff   # the repository cites itself, not ATTUNE
    assert "@article" not in (ROOT / "README.md").read_text(encoding="utf-8")
    page = files["qwen3.5-122b-a10b/h200/index.html"]
    assert page.count('<h2 id="cite">Citation</h2>') == 1 and page.count("If you use a recipe by ATTUNE,") == 1
    assert page.count('(<a href="#cite">cite</a>)') == page.count('<dt>Author</dt><dd><a href="%s">ATTUNE</a>' % ATTUNE)
    assert '<pre id="bibtex-1">%s</pre>' % build.esc(cits[ATTUNE]) in page and 'data-copy="bibtex-1"' in page


def test_citation_only_for_its_author(repo):
    recipe = repo / "recipes/deepseek-v4-flash/h200/attune-tp2-pp4-ep8.yaml"
    text = recipe.read_text()
    assert "\n  name: ATTUNE\n  url: %s\n" % ATTUNE in text
    recipe.write_text(text.replace("\n  name: ATTUNE\n  url: %s\n" % ATTUNE, "\n  github: attune\n"))
    out, errs = build.render(repo)
    assert errs == []
    page = out["deepseek-v4-flash/h200/index.html"].decode("utf-8")
    assert "github.com/attune" in page and 'id="cite"' not in page and 'href="#cite"' not in page
    assert b'id="cite"' in out["gpt-oss-20b/h100/index.html"]


def test_citation_names_each_author(repo):
    # another author linking a paper of citations.yaml is named for that paper; its own paper gets its own entry
    recipe = repo / "recipes/deepseek-v4-flash/h200/attune-tp2-pp4-ep8.yaml"
    other = "https://example.org/other-paper"
    text = recipe.read_text().replace("attune-tp2-pp4-ep8", "someone-tp2-pp4-ep8")
    recipe.unlink()
    (recipe.parent / "someone-tp2-pp4-ep8.yaml").write_text(text.replace(
        "\n  name: ATTUNE\n  url: %s\n" % ATTUNE, "\n  name: Someone\n  url: https://example.org\n  paper: %s\n" % other))
    with open(repo / "citations.yaml", "a") as f:
        f.write('%s:\n  type: misc\n  title: "Other: a method"\n  authors:\n    - given-names: A\n'
                '      family-names: Person\n  year: 2026\n' % other)
    out, errs = build.render(repo)
    assert errs == []
    page = out["deepseek-v4-flash/h200/index.html"].decode("utf-8")
    assert "If you use a recipe by Someone, please cite" in page and "If you use a recipe by ATTUNE" not in page
    assert "@misc{person2026other," in page and 'href="#cite"' in page


def test_repository_site_is_up_to_date():
    assert build.main(["--check"]) == 0


@pytest.fixture
def repo(tmp_path):
    for name in ("models.yaml", "gpus.yaml", "LICENSE", "NOTICE", "CITATION.cff", "citations.yaml"):
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
    recipe.write_text(recipe.read_text().replace("tokens_per_s_per_gpu: 6736.89", "tokens_per_s_per_gpu: 6000"))
    assert build.main(["--root", str(repo), "--check"]) == 1
    assert "usd_per_million_tokens" in capsys.readouterr().out
