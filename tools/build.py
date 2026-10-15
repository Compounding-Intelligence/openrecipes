#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the OpenRecipes static site from recipes/, models.yaml and gpus.yaml into site/.

    python3 tools/build.py            # validate every recipe, then write site/
    python3 tools/build.py --check    # exit 1 if a recipe is invalid or site/ differs from a fresh build
    python3 tools/build.py --bibtex   # print the preferred citation of CITATION.cff as BibTeX

Needs the Python standard library and PyYAML. The output is deterministic: the same inputs give the same bytes.
"""
import argparse
import datetime
import html
import json
import re
import shutil
import sys
import tempfile
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate import ROOT, load_yaml, validate_all  # noqa: E402

SITE_URL = "https://openrecipes.ai"
REPO_URL = "https://github.com/Compounding-Intelligence/openrecipes"
ASSETS = Path(__file__).resolve().parent / "assets"
# The open-ring mark; assets/favicon.svg draws the same shapes on a white tile.
MARK = ('<svg class="mark" width="22" height="22" viewBox="0 0 32 32" aria-hidden="true">'
        '<path d="M25.85 14.26A10 10 0 1 1 21 7.34" fill="none" stroke="#0e7490" stroke-width="4" stroke-linecap="round"/>'
        '<circle cx="25.2" cy="7.6" r="2.6" fill="#0f172a"/></svg>')
NAV = [("models", "Models", "index.html#models"), ("protocol", "Protocol", "protocol.html"), ("submit", "Submit", "submit.html")]

esc = html.escape
# Internal page links, written as files (protocol.html) and served at clean URLs (/protocol) by Workers static assets.
PAGE_HREF = re.compile(r'href="((?:/|(?:\.\./)*))(index|protocol|submit)\.html(#[^"]*)?"')


def tps(v):
    return "{:,.1f}".format(v)


def usd(v):
    return "${:,.2f}".format(v)


def pct(v):
    return "{:.1f}%".format(v * 100)


def iso(d):
    return d.isoformat() if isinstance(d, datetime.date) else str(d)


def inline(text):
    """Escape text; `code` becomes <code>, https:// links become anchors outside code."""
    out = []
    for i, part in enumerate(re.split(r"`([^`]*)`", text)):
        if i % 2:
            out.append("<code>%s</code>" % esc(part))
            continue
        pos = 0
        for m in re.finditer(r"https://[^\s<>()]+[^\s<>().,;:]", part):
            out.append(esc(part[pos:m.start()]))
            out.append('<a href="%s">%s</a>' % (esc(m.group(0)), esc(m.group(0))))
            pos = m.end()
        out.append(esc(part[pos:]))
    return "".join(out)


def render_text(text):
    """Plain text to HTML: blank lines separate paragraphs, lines starting with "- " are list items."""
    blocks, html_out = [], []
    for raw in re.split(r"\n\s*\n", text.strip()):
        blocks.append(raw.splitlines())
    for lines in blocks:
        if lines and lines[0].startswith("- "):
            items = []
            for line in lines:
                if line.startswith("- "):
                    items.append(line[2:].strip())
                elif items:
                    items[-1] += " " + line.strip()
            html_out.append("<ul>%s</ul>" % "".join("<li>%s</li>" % inline(x) for x in items))
        else:
            html_out.append("<p>%s</p>" % inline(" ".join(x.strip() for x in lines)))
    return "\n".join(html_out)


# ---- data ---------------------------------------------------------------------------------------------------------

def collect(recipes, models, gpus):
    """Pairs keyed (model, gpu type): recipes ranked by tokens/s per GPU, best first."""
    pairs = {}
    for path, r in recipes.items():
        slug = Path(path).stem
        pairs.setdefault((r["model"], r["gpu"]["type"]), []).append({"path": path, "slug": slug, "r": r})
    for items in pairs.values():
        items.sort(key=lambda x: (-x["r"]["result"]["tokens_per_s_per_gpu"], x["slug"]))
        for i, x in enumerate(items, 1):
            x["rank"] = i
    return pairs


def citation(root):
    """CITATION.cff's preferred citation: (its url, the citation as BibTeX)."""
    c = load_yaml(Path(root) / "CITATION.cff")["preferred-citation"]
    kind = "article" if c["type"] == "article" else "misc"
    key = "%s%s%s" % (c["authors"][0]["family-names"], c["year"], re.match(r"\w+", c["title"]).group(0))
    # no-break spaces keep each name on one line when the author list wraps
    names = " and ".join("%s\xa0%s" % (a["given-names"], a["family-names"]) for a in c["authors"])
    fields = [("title", "{%s}" % c["title"]), ("author", names)]
    if kind == "article":
        fields.append(("journal", c["journal"]))
    fields += [("year", c["year"]), ("url", c["url"])]
    body = ",\n".join(textwrap.fill("  %-7s = {%s}" % (k, v), 100, subsequent_indent=" " * 13, break_long_words=False)
                      for k, v in fields)
    return c["url"], "@%s{%s,\n%s\n}\n" % (kind, key.lower(), body.replace("\xa0", " "))


def cites(recipe, cite):
    """Whether the recipe's author is the method the preferred citation describes."""
    return cite[0] in (recipe["author"].get("url"), recipe["author"].get("paper"))


def pair_url(model, gpu_type):
    return "%s/%s/" % (model["slug"], gpu_type.lower())


def families(models):
    out = {}
    for m in models.values():
        out.setdefault(m["family"], []).append(m)
    return out


# ---- page shell ---------------------------------------------------------------------------------------------------

def _strip_tags(s):
    return html.unescape(re.sub(r"<[^>]+>", "", s)).strip()


def stack_table(table):
    """Label every cell with its column header, so narrow screens can show each row as a list (style.css)."""
    head = re.search(r"<thead>(.*?)</thead>", table, re.S)
    labels = [_strip_tags(h) for h in re.findall(r"<th[^>]*>(.*?)</th>", head.group(1), re.S)] if head else []

    def row(m):
        cells = iter(range(10 ** 6))

        def cell(c):
            i = next(cells)
            label = labels[i] if i < len(labels) else ""
            return '<td%s data-label="%s"><span class="cv">%s</span></td>' % (c.group(1), esc(label), c.group(2))
        return re.sub(r"<td([^>]*)>(.*?)</td>", cell, m.group(0), flags=re.S)
    body = re.sub(r"<tbody>.*?</tbody>", lambda b: re.sub(r"<tr[^>]*>.*?</tr>", row, b.group(0), flags=re.S), table,
                  flags=re.S)
    return body.replace("<table>", '<table class="stack">', 1)


def clean_path(rel):
    return rel[:-len("index.html")] if rel.endswith("index.html") else rel[:-len(".html")] if rel.endswith(".html") else rel


def clean_hrefs(page):
    return PAGE_HREF.sub(lambda m: 'href="%s%s"' % ((m.group(1) + clean_path(m.group(2) + ".html")) or "./",
                                                    m.group(3) or ""), page)


def shell(rel, title, description, body, current=None, data_date=None, absolute=False, canonical=True):
    prefix = "/" if absolute else "../" * rel.count("/")
    url = SITE_URL + "/" + clean_path(rel)
    links = []
    for key, label, href in NAV:
        cur = ' aria-current="page"' if key == current else ""
        links.append('<a href="%s%s"%s>%s</a>' % (prefix, href, cur, label))
    links.append('<a href="%s">GitHub</a>' % REPO_URL)
    body = re.sub(r"<table>.*?</table>", lambda m: stack_table(m.group(0)), body, flags=re.S)
    footer = "OpenRecipes, Apache-2.0."
    if data_date:
        footer += " Measurements through %s." % esc(data_date)
    canonical = '<link rel="canonical" href="%s">\n' % esc(url) if canonical else ""
    return clean_hrefs("""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{desc}">
{canonical}<link rel="icon" href="{p}favicon.svg" type="image/svg+xml">
<link rel="stylesheet" href="{p}style.css">
<script src="{p}app.js" defer></script>
</head>
<body>
<header class="nav"><a class="brand" href="{p}index.html">{mark}OpenRecipes</a><nav class="links" aria-label="Site">{links}</nav></header>
<main class="wrap">
{body}
</main>
<footer><div class="wrap">{footer}</div></footer>
</body>
</html>
""".format(title=esc(title), desc=esc(description), canonical=canonical, p=prefix, mark=MARK,
           links="".join(links), body=body, footer=footer))


# ---- index --------------------------------------------------------------------------------------------------------

SEGS = 24   # segments in a throughput gauge


def gauge(share):
    k = max(1, round(share * SEGS))
    return '<span class="gauge" title="%d%% of the best">%s</span>' % (
        round(share * 100), "".join('<b class="on"></b>' if i < k else "<b></b>" for i in range(SEGS)))


def author_html(a):
    if a.get("github"):
        return '<a href="https://github.com/%s">%s</a>' % (esc(a["github"]), esc(a.get("name") or a["github"]))
    return '<a href="%s">%s</a>' % (esc(a["url"]), esc(a["name"]))


def index_page(pairs, models, gpus, date):
    groups = list(families(models).items())
    total, left, count = len(models), [], 0
    for fam, ms in groups:
        if count < total / 2:
            left.append((fam, ms))
            count += len(ms)
    columns = []
    for col in (left, groups[len(left):]):
        fams = []
        for fam, ms in col:
            rows = []
            for m in ms:
                ours = [gt for gt in gpus if (m["model"], gt) in pairs]
                name = ('<a href="#m-%s">%s</a>' % (esc(m["slug"]), esc(m["model"])) if ours else esc(m["model"]))
                chips = "".join('<a href="%s">%s</a>' % (pair_url(m, gt), gt) if gt in ours else '<span class="off">%s</span>' % gt
                                for gt in gpus)
                rows.append('<div class="irow%s"><span class="mname">%s</span><span class="lbs">%s</span></div>'
                            % (" has" if ours else "", name, chips))
            has = any((m["model"], gt) in pairs for m in ms for gt in gpus)
            fams.append('<div class="fam%s"><div class="fname">%s</div><div>%s</div></div>'
                        % (" has" if has else "", esc(fam), "".join(rows)))
        columns.append('<div><div class="ihead ml"><span>Family</span><span>Model</span><span>GPU</span></div>%s</div>' % "".join(fams))
    n_recipes = sum(len(v) for v in pairs.values())
    hero = ('<header class="hud hero"><h1>OpenRecipes</h1>'
            '<p class="tag">Tuned, verified LLM training recipes on one fixed SFT workload, each with its cost per million '
            'tokens and a passed convergence check.</p>'
            '<p class="spec"><span>8,192-token SFT rows</span> &middot; <span>global batch 128</span> &middot; <span>BF16</span>'
            ' &middot; <span>%d recipes on %d model and GPU pairs</span></p>'
            '<div class="acts"><a class="btn pri" href="submit.html">Submit a recipe</a><a class="btn" href="protocol.html">Protocol</a>'
            '<a class="btn" href="%s">GitHub</a></div></header>') % (n_recipes, len(pairs), REPO_URL)
    seg = ('<div class="seg" role="group" aria-label="GPU"><span class="ml">GPU</span><button type="button" class="on" '
           'data-gpu="" aria-pressed="true">All</button>%s</div>' % "".join(
               '<button type="button" data-gpu="%s" aria-pressed="false">%s</button>' % (gt, gt) for gt in gpus))
    blocks = []
    for m in models.values():
        rows = [(gt, pairs[(m["model"], gt)]) for gt in gpus if (m["model"], gt) in pairs]
        if not rows:
            continue
        top = max(items[0]["r"]["result"]["tokens_per_s_per_gpu"] for _, items in rows)
        low = min(items[0]["r"]["result"]["usd_per_million_tokens"] for _, items in rows)
        trs = []
        for gt, items in rows:
            r = items[0]["r"]
            t_, u = r["result"]["tokens_per_s_per_gpu"], r["result"]["usd_per_million_tokens"]
            trs.append('<tr data-gpu="{gt}"><td class="g">{gt}</td><td class="n">{count}</td><td class="tp"><span class="v{hi}">{tps}'
                       '</span>{gauge}</td><td class="n{lo}">{usd}</td><td>{label}<br><small>{hl}</small></td>'
                       '<td class="nw">{date}</td><td>{author}</td><td class="go"><a href="{url}">{more}&nbsp;&rsaquo;</a></td></tr>'.format(
                           gt=esc(gt), count=r["gpu"]["count"], hi=" hi" if t_ == top else "", tps=tps(t_), gauge=gauge(t_ / top),
                           lo=" hi" if u == low else "", usd=usd(u), label=esc(r["layout"]["label"]),
                           hl=esc(r["layout"]["highlights"]), date=esc(iso(r["result"]["measured"])), author=author_html(r["author"]),
                           url=pair_url(m, gt), more="All recipes" if len(items) > 1 else "Recipe"))
        blocks.append('<section class="hud mb" id="m-{slug}"><div class="ml">{fam}</div><h2>{model}</h2><div class="tablewrap">'
                      '<table><thead><tr><th>GPU</th><th>GPUs</th><th>Tokens/s per GPU</th><th>$ / 1M tokens</th><th>Layout</th>'
                      '<th>Measured</th><th>Author</th><th></th></tr></thead><tbody>{rows}</tbody></table></div></section>'.format(
                          slug=esc(m["slug"]), fam=esc(m["family"]), model=esc(m["model"]), rows="".join(trs)))
    bar = '<div class="ibar"><button type="button" class="tg" aria-pressed="false" hidden>With recipes only</button></div>'
    body = "\n".join([hero, '<section class="hud idx" id="models">%s%s</section>' % (bar, "".join(columns)), seg] + blocks)
    return shell("index.html", "OpenRecipes: tuned, verified LLM training recipes",
                 "Training recipes for LLMs on H100, H200 and B200, each measured one way and checked for correctness.",
                 body, "models", date)


# ---- pair page ----------------------------------------------------------------------------------------------------

def facts(rows):
    return '<dl class="facts">%s</dl>' % "".join("<dt>%s</dt><dd>%s</dd>" % (esc(k), v) for k, v in rows)


def correctness_rows(c):
    conv = c["convergence"]
    rows = [("Convergence check", "correlation %s, final loss difference %s, %d pointwise failures over %d steps"
             % (conv["correlation"], conv["final_loss_difference"], conv["pointwise_failures"], conv["steps"]))]
    for t in c["component_tests"]:
        rows.append(("Component test", "%s: %s. Shape: %s. %s%s" % (
            inline(t["test"]), esc("%s, %s" % (t["gpu"], iso(t["date"]))), inline(t["shape"]), inline(t["detail"]),
            (" " + inline(t["note"])) if t.get("note") else "")))
    return rows


def recipe_details(x, root, cite):
    r, slug = x["r"], x["slug"]
    res, lay, ds = r["result"], r["layout"], r["workload"]["dataset"]
    a = r["author"]
    links = ['<a href="%s">paper</a>' % esc(a["paper"])] if a.get("paper") else []
    if cites(r, cite):
        links.append('<a href="#cite">cite</a>')
    author = author_html(a) + (" (%s)" % ", ".join(links) if links else "")
    if res["mfu"] is None:
        mfu = "not reported: %s" % inline(res["mfu_note"] or "")
    else:
        mfu = "%s (N = %s active parameters per token)" % (pct(res["mfu"]), "{:,}".format(res["active_params"]["count"]))
    measured = iso(res["measured"]) + ("" if res["window"] == [5, 19] else ", %d-step run scored over steps %d&ndash;%d"
                                       % (res["steps_per_run"], res["window"][0], res["window"][1]))
    rows = [
        ("Author", author),
        ("GPUs", "%d x %s %d GB (%s, %d per node)" % (r["gpu"]["count"], esc(r["gpu"]["type"]), r["gpu"]["memory_gb"],
                                                        esc(r["gpu"]["instance"]), r["gpu"]["per_node"])),
        ("Layout", inline(lay["summary"])),
        ("Tokens/s per GPU", "{:,.2f}".format(res["tokens_per_s_per_gpu"])),
        ("$ / 1M tokens", usd(res["usd_per_million_tokens"])),
        ("MFU", mfu),
        ("Step-time CV", "%s%%" % res["step_time_cv_percent"]),
    ] + correctness_rows(r["correctness"]) + [
        ("Measured", measured),
        ("GPU-hours", str(res["gpu_hours"])),
        ("Rows", "%s (rows_sha256 %s)" % (inline("`%s`" % ds["build"]["name"]), inline("`%s`" % ds["rows_sha256"]))),
    ]
    stack_rows = [(k.replace("_", " "), inline(v)) for k, v in r["stack"].items() if k != "source"]
    env = ('<h3>Environment</h3><ul class="env">%s</ul>' % "".join(
        "<li><code>%s=%s</code></li>" % (esc(k), esc(v)) for k, v in r["env"].items()) if r["env"] else "")
    patches = ('<h3>Patches</h3><ul>%s</ul>' % "".join('<li><a href="%s/blob/main/%s"><code>%s</code></a></li>'
                                                     % (REPO_URL, esc(p), esc(p)) for p in r["patches"])
               if r["patches"] else "")
    notes = '<div class="prose">%s</div>' % render_text(r["notes"]) if r.get("notes") else ""
    raw = (Path(root) / x["path"]).read_text(encoding="utf-8")
    return """<details class="card recipe" id="{slug}"{open}>
<summary><span class="rank">#{rank}</span><span class="slug">{slug}</span><span class="tps">{tps} tokens/s/GPU</span><span class="muted small">{label}</span><span class="hl">{hl}</span></summary>
<div class="body">
{notes}{facts}
<div class="cols"><div><h3>Stack</h3>{stack}</div><div>{env}{patches}</div></div>
<div class="filebar"><span class="k">Command</span><span class="acts"><a class="btn o" href="../../protocol.html#run">How to run</a><button type="button" class="btn o" data-copy="run-{slug}">Copy</button></span></div>
<pre id="run-{slug}">{run_text}</pre>
<div class="filebar"><span class="k">{path}</span><span class="acts"><a class="btn o" href="../../{path}" download>Download</a><a class="btn o" href="{repo}/blob/main/{path}">GitHub</a><button type="button" class="btn o" data-copy="file-{slug}">Copy</button></span></div>
<pre class="file" id="file-{slug}">{raw}</pre>
</div>
</details>""".format(slug=esc(slug), open=" open" if x["rank"] == 1 else "", rank=x["rank"],
                     tps=tps(res["tokens_per_s_per_gpu"]), label=esc(lay["label"]), hl=esc(lay["highlights"]),
                     notes=notes, facts=facts(rows), stack=facts(stack_rows), env=env, patches=patches,
                     run_text=esc(r["run"]), path=esc(x["path"]), repo=REPO_URL, raw=esc(raw))


def pair_page(model, gpu, items, pairs, gpus, date, root, cite):
    gt = gpu["type"]
    best = items[0]
    br = best["r"]
    res = br["result"]
    wl = br["workload"]
    workload = "{:,}-token SFT rows &middot; global batch {} &middot; {}".format(
        wl["row_tokens"], wl["global_batch"], esc(wl["precision"].upper()))
    cheapest = min(x["r"]["result"]["usd_per_million_tokens"] for x in items)
    top = res["tokens_per_s_per_gpu"]
    rows = []
    for x in items:
        r = x["r"]
        t_, u = r["result"]["tokens_per_s_per_gpu"], r["result"]["usd_per_million_tokens"]
        rows.append('<tr><td class="rk">#{rank}</td><td class="tp"><span class="v{hi}">{tps}</span>{gauge}</td>'
                    '<td class="n{lo}">{usd}</td><td class="n">{gpus}</td><td>{label}<br><small>{hl}</small></td>'
                    '<td class="nw">{date}</td><td>{author}</td><td class="go"><a href="#{slug}">Recipe&nbsp;&rsaquo;</a></td></tr>'.format(
                        rank=x["rank"], hi=" hi" if x["rank"] == 1 else "", tps=tps(t_), gauge=gauge(t_ / top),
                        lo=" hi" if u == cheapest else "", usd=usd(u), gpus=r["gpu"]["count"], label=esc(r["layout"]["label"]),
                        hl=esc(r["layout"]["highlights"]), date=esc(iso(r["result"]["measured"])), author=author_html(r["author"]),
                        slug=esc(x["slug"])))
    table = ('<section class="hud rank"><div class="ml">{fam}</div><h1>Training {model} on {gpu}</h1>'
             '<div class="ml">Ranked by tokens/s per GPU &middot; {workload}</div><div class="tablewrap"><table><thead><tr>'
             '<th>#</th><th>Tokens/s per GPU</th><th>$ / 1M tokens</th><th>GPUs</th><th>Layout</th><th>Measured</th><th>Author</th>'
             '<th></th></tr></thead><tbody>{rows}</tbody></table></div></section>').format(
        fam=esc(model["family"]), model=esc(model["model"]), gpu=esc(gt), workload=workload, rows="".join(rows))
    others = []
    for ot, og in gpus.items():
        o = pairs.get((model["model"], ot))
        if not o:
            others.append('<div class="card gpu off"><div class="k">%s</div><div class="v">&ndash;</div>'
                          '<div class="small">No recipe yet</div></div>' % esc(ot))
            continue
        orr = o[0]["r"]
        inner = ('<div class="k">{gpu} &middot; {count} GPUs{here}</div><div class="v">{tps}</div>'
                 '<div class="small muted">tokens/s/GPU &middot; {usd} / 1M tokens</div>').format(
            gpu=esc(ot), count=orr["gpu"]["count"], here=" &middot; this page" if ot == gt else "",
            tps=tps(orr["result"]["tokens_per_s_per_gpu"]), usd=usd(orr["result"]["usd_per_million_tokens"]))
        if ot == gt:
            others.append('<div class="card gpu here">%s</div>' % inner)
        else:
            others.append('<a class="card gpu" href="../../%s">%s</a>' % (pair_url(model, ot), inner))
    notes = ""
    if model.get("notes"):
        notes = ('<h2>Notes for %s</h2><div class="card notes prose">%s<p class="muted small">The steps every recipe '
                 'shares: <a href="../../protocol.html#run">Running a recipe</a>.</p></div>'
                 % (esc(model["model"]), render_text(model["notes"])))
    cited = [x["r"]["author"] for x in items if cites(x["r"], cite)]
    citing = ""
    if cited:
        citing = ('<h2 id="cite">Citation</h2><div class="card notes"><p>If you use a recipe by %s, please cite '
                  '<a href="%s">the paper</a>.</p><div class="filebar"><span class="k">BibTeX</span><span class="acts">'
                  '<button type="button" class="btn o" data-copy="bibtex">Copy</button></span></div>'
                  '<pre id="bibtex">%s</pre></div>' % (esc(cited[0]["name"]), esc(cite[0]), esc(cite[1])))
    body = "\n".join([
        '<a class="back" href="../../index.html#models">&lsaquo; All models</a>', table, notes,
        "<h2>Recipes</h2>",
        "\n".join(recipe_details(x, root, cite) for x in items), citing,
        "<h2>%s on other GPUs</h2>" % esc(model["model"]),
        '<div class="others">%s</div>' % "".join(others)])
    desc = "Verified training recipes for %s on %s: best %s tokens/s per GPU on %d GPUs, %s per million tokens." % (
        model["model"], gt, tps(res["tokens_per_s_per_gpu"]), br["gpu"]["count"], usd(cheapest))
    return shell(pair_url(model, gt) + "index.html", "Training %s on %s | OpenRecipes" % (model["model"], gt),
                 desc, body, None, date)


# ---- protocol -----------------------------------------------------------------------------------------------------

def protocol_page(recipes, gpus, rates, date):
    rate_rows = "".join(
        '<tr><td>{t}</td><td>{name}</td><td>{inst}</td><td class="n">${rate}</td><td class="n">{peak:,}</td>'
        '<td><a href="{src}">source</a></td></tr>'.format(
            t=esc(g["type"]), name=esc(g["name"]), inst=esc(g["instance"]), rate=g["usd_per_gpu_hour"],
            peak=g["peak_bf16_tflops"], src=esc(g["rate_source"])) for g in gpus.values())
    early = [r for r in recipes.values() if r["result"]["steps_per_run"] < 20]
    short = ""
    if early:
        days = sorted({iso(r["result"]["measured"]) for r in early})
        lens = sorted({r["result"]["steps_per_run"] for r in early})
        short = ("%d recipes, measured on %s, ran %s steps under an earlier window and are scored over steps "
                 "4&ndash;13; they record <code>steps_per_run</code> and their window, and their convergence checks ran "
                 "separate 20-step arms. "
                 % (len(early), " and ".join(days), " or ".join(str(x) for x in lens)))
    counts = sorted(r["gpu"]["count"] for r in recipes.values())
    step_tokens = 128 * 8192
    spread = ("The recipes run it on %d to %d GPUs, which leaves each GPU between %dK and %dK tokens per step; the "
              "large mixture-of-experts recipes sit at the low end" % (
                  counts[0], counts[-1], step_tokens // counts[-1] // 1024, step_tokens // counts[0] // 1024))
    body = """<h1>Protocol</h1>
<p class="lede">Every recipe is measured one way, so recipes for the same model and GPU type, a <em>pair</em>, compare directly. A pair can hold recipes from several authors; they rank by tokens/s per GPU.</p>
<h2 id="workload">Workload</h2>
<p>The rows are SFT conversations, tokenized with the model's own tokenizer and packed to exactly 8,192 tokens each with no padding: a seeded permutation of the source conversations, joined with an EOS token between conversations and cut into consecutive rows. A step trains on a global batch of 128 rows in BF16, and every run of a model reads the same rows in the same order.</p>
<p>The recipes use the Tulu 3 SFT mixture (<code>allenai/tulu-3-sft-mixture</code>, revision <code>b14afda60f1bbebe55d5d2fa1e4df5042f97f8be</code>, ODC-By 1.0), built once per tokenizer. <a href="{repo}/tree/main/data"><code>data/</code></a> rebuilds the rows, and <code>data/verify_rows.py</code> checks a build against the <code>rows_sha256</code> a recipe records. Some subsets of the mixture carry their own licenses and terms, including non-commercial ones (see the dataset card). This repository does not redistribute the data.</p>
<p>A recipe may run on any number of GPUs of its type and records the count.</p>
<h2 id="measurement">Measurement</h2>
<p>A run lasts 20 optimizer steps (0&ndash;19) with the profiler off. Steps 0&ndash;4 absorb startup, compilation and allocator warm-up and are discarded; steps 5&ndash;19 form the measured window <em>W</em>. The score is the aggregate throughput over the window, summed before dividing:</p>
<pre>tokens/s/GPU = (sum over W of tokens) / (number of GPUs x sum over W of step time)</pre>
<p>A recipe is one run. It reports the score and the coefficient of variation of the step time over <em>W</em>. The same 20-step run is the recipe's arm of the convergence check below, so one run covers both. A run that also needs a profiler trace continues past step 19: step 20 sets the profiler up, step 21 warms it up and step 22 (up to 24) is recorded. Those steps are never scored, and a run that long is not a convergence arm, because the run length can change the learning-rate schedule. {short}</p>
<p>GPU-hours count GPUs times the time from scheduler admission to finish, including startup.</p>
<h2 id="correctness">Correctness</h2>
<p>A faster recipe is listed once it passes two checks against its <em>reference recipe</em>, the recipe it was tuned from. The <em>configuration check</em> diffs the recipe's resolved arguments against the reference recipe's and classes every changed argument: structural changes (<code>placement</code>, <code>packing</code>) and same-math swaps (<code>allowlisted</code>) are eligible, and every same-math swap needs a passed <em>component test</em> at the model's shape, which records what it compares, the layer shape, the bound and the worst error.</p>
<p>The <em>convergence check</em> compares the per-step loss of a 20-step arm of the recipe with a 20-step arm of the reference recipe on the same image and rows, with Megatron-Bridge's <code>validate_convergence</code> and its pipeline thresholds. A recipe records the verdict, the correlation of the two loss curves, the final loss difference and the number of pointwise failures. The check compares 20 steps, so it catches changes that move the loss curve early; a change whose effect shows only later in training can pass it.</p>
<p>The measured run serves as the recipe's arm when nothing in it adds work to the step. A reference arm serves every recipe of its pair that runs on the same image and rows. Publishing the reference recipe is optional; a recipe states it when it is public.</p>
<p><b>Objective conventions.</b> Qwen3.5 and Qwen3.6 recipes train with the load-balancing loss off (<code>--moe-aux-loss-coeff 0</code>), in the reference recipe and the recipe alike. Megatron computes that loss per microbatch, so with the loss on, the number of rows per microbatch would be part of the objective.</p>
<h2 id="cost">Cost</h2>
<p><code>$ / 1M tokens = per-GPU hourly rate / (tokens/s/GPU x 3,600) x 10^6</code>. The rates are {basis}, fetched {fetched}, from <a href="{repo}/blob/main/gpus.yaml"><code>gpus.yaml</code></a>. Prices change; recompute for your region and commitment. Within a pair every recipe uses the same rate, so the fastest recipe is also the cheapest; each model's pages compare its GPU types on both.</p>
<div class="card tablewrap"><table><thead><tr><th>Type</th><th>GPU</th><th>Instance</th><th>$ / GPU-hour</th><th>Dense BF16 peak, TFLOP/s</th><th></th></tr></thead><tbody>{rates}</tbody></table></div>
<h2 id="mfu">MFU</h2>
<p>MFU = 6 x N x tokens/s/GPU / peak. <em>N</em> is the model's active parameters per token: every weight the text model applies to each token in a matrix multiplication (attention, linear attention, sparse-attention indexer, dense and shared-expert MLPs, routers and the output projection), with routed experts counted at top-k of the expert count. N leaves out the input embedding (a lookup), norms and biases, position embeddings and integer index tables, multi-token prediction layers and vision towers; FP4 weights stored two per byte count at their unpacked size. <a href="{repo}/blob/main/tools/active_params.py"><code>tools/active_params.py</code></a> counts N from the checkpoint's safetensors headers, and a recipe records N with the checkpoint and its revision. The peak is the GPU's dense BF16 peak in the table above. Attention-score FLOPs and recomputation are not counted, so MFU compares models with softmax, linear and sparse attention on one basis and sits below the hardware's actual FLOP rate.</p>
<p>MFU on this workload is low for large mixture-of-experts models, and it reflects the workload more than the recipe. A step of 128 rows of 8,192 tokens is about 1.05 million tokens. {spread}, so pipeline bubbles, the expert all-to-all and, where memory forces it, the CPU optimizer step take a large share of every step. Compare MFU between recipes of one pair, not with large-batch pretraining.</p>
<h2 id="stack">Stack</h2>
<p>A recipe records the versions it ran on: the framework (slime or miles, with its commit), Megatron-LM, Transformer Engine, PyTorch, CUDA and NCCL, plus any communication library it uses (for example DeepEP), the GPUs per node and the inter-node network. A version the run's evidence does not show reads <code>not pinned</code>.</p>
<h2 id="run">Running a recipe</h2>
<p>A recipe's command is the training command as run, with paths replaced by variables: <code>$HF_CKPT</code> is the model's Hugging Face checkpoint, <code>$TORCH_DIST</code> the Megatron <code>torch_dist</code> weights the run loads and <code>$SFT_DATA</code> the rows. <code>&lt;slime entry module&gt;</code> stands for the training image's slime entry point, and <code>MODEL_ARGS</code> come from the framework's model script for the model (<code>scripts/models/</code> in slime) unless the model's notes say otherwise. A model's notes, on each of its pages, give what all of its recipes need.</p>
<p>Build the rows a recipe names with <a href="{repo}/tree/main/data"><code>data/</code></a> and check them with <code>python3 data/verify_rows.py &lt;rows.parquet&gt; --recipe &lt;recipe file&gt;</code>. Put <code>data/</code> on <code>PYTHONPATH</code> so <code>--rollout-function-path pretokenized_sft_rollout.generate_rollout</code> resolves, and <code>patches/</code> too when the recipe lists patches.</p>
<p>For another GPU count, change DP, the number of replicas; a replica takes TP x PP x CP GPUs. The global batch of 128 rows must divide by DP, and the GPUs per pipeline stage (GPU count / PP) must stay a multiple of EP x ETP. For another row length, set <code>AUTOTUNE_SEQ_LEN</code> to it and size <code>--max-tokens-per-gpu</code> x CP to hold at least one row. Re-measure after any change, and re-run the convergence check after any change beyond the parallel layout and packing.</p>
<h2 id="status">Status and authors</h2>
<p>A <em>verified</em> recipe was measured as above and passed both checks; OpenRecipes lists verified recipes only. Each recipe names its author: the GitHub handle of the person or team that submitted it, or a name and a link for a tool.</p>""".format(
        repo=REPO_URL, basis=esc(rates["basis"]), fetched=esc(iso(rates["fetched"])),
        rates=rate_rows, short=short, spread=spread)
    return shell("protocol.html", "Protocol | OpenRecipes",
                 "How OpenRecipes measures training throughput, checks correctness, and computes cost and MFU.",
                 body, "protocol", date)


# ---- submit -------------------------------------------------------------------------------------------------------

TEMPLATE = """schema_version: 1
model: Qwen3.5-122B-A10B          # as in models.yaml; add the model there if it is missing
gpu: {type: H200, memory_gb: 141, count: 32, per_node: 8, instance: AWS p5en.48xlarge}
author: {github: your-handle}     # or {name: <tool>, url: https://...}
status: verified
stack:                            # the versions the measured run used; "not pinned" when unknown
  framework: slime <version> (THUDM/slime <commit>)
  megatron_lm: <commit> (megatron-core <version>)
  transformer_engine: <version>
  pytorch: <version>
  cuda: <version>
  nccl: <version>
workload:
  row_tokens: 8192
  rows: SFT conversations
  global_batch: 128
  precision: bf16
  dataset:
    name: Tulu 3 SFT mixture
    source: allenai/tulu-3-sft-mixture
    revision: b14afda60f1bbebe55d5d2fa1e4df5042f97f8be
    license: ODC-By 1.0
    rows_sha256: <printed by data/verify_rows.py>
    build: {name: qwen35-tulu3-fixed8192-v1, tokenizer: <name and tokenizer.json sha256>, loss_mask: qwen3_5,
            rows: 6400, row_tokens: 8192, seed: 42, builder: data/generate_fixed_sft.py}
layout:
  label: TP1 PP4 EP8 DP8
  summary: <the full layout in words>
  highlights: <the settings that set the recipe apart>
  tp: 1
  pp: 4
  vpp: 1
  cp: 1
  ep: 8
  etp: 1
  dp: 8                           # gpu.count / (tp x pp x cp)
  sequence_parallel: false
  layer_split: null               # or {first: 11, last: 10}
  max_tokens_per_gpu: 8192
  rows_per_microbatch: 1
  recompute: {granularity: full, method: uniform, num_layers: 1}
  offload: null                   # or {optimizer_fraction: 0.9, overlap_d2h_h2d: true}
  dispatcher: alltoall
result:
  tokens_per_s_per_gpu: <score over steps 5-19>
  usd_per_million_tokens: <rate in gpus.yaml / (tokens/s/GPU x 3600) x 1e6, to the cent>
  step_time_cv_percent: <CV of the step time over the window>
  mfu: <6 x N x tokens/s/GPU / peak, or null>
  mfu_note: null                  # why, when mfu is null
  active_params: {count: <N from tools/active_params.py>, source: https://huggingface.co/<repo>, revision: <commit>}
  measured: 2026-10-14
  steps_per_run: 20
  window: [5, 19]
  window_tokens: 15728640         # 15 steps x 128 rows x 8,192 tokens
  runs: 1
  profiler: false
  gpu_hours: <GPUs x (admission to finish)>
  gpu_hours_note: GPUs x (scheduler admission to finish), including startup
correctness:
  result: pass
  summary: <one line>
  reference: <the reference recipe, or "Not published.">
  configuration: pass
  component_tests: []             # one entry per same-math swap: test, shape, gpu, date, result, detail
  convergence: {steps: 20, result: pass, correlation: <r>, final_loss_difference: <d>, pointwise_failures: 0}
  rules: <the rule lists the check used>
env: {CUDA_DEVICE_MAX_CONNECTIONS: "1"}
patches: []                       # files under patches/ the recipe needs
run: |
  <the training command as run, with paths replaced by variables>
notes: |                          # optional; plain text, blank lines separate paragraphs, "- " starts a list item
  <what a reader needs to run this recipe that the protocol and the model's notes do not say>
"""


def schema_rows(schema, prefix="", required=()):
    out = []
    for key, sub in (schema.get("properties") or {}).items():
        path = prefix + key
        types = sub.get("type")
        if types is None and "enum" in sub:
            types = "one of " + ", ".join("null" if v is None else str(v) for v in sub["enum"])
        elif types is None and "const" in sub:
            types = "always %s" % json.dumps(sub["const"])
        elif isinstance(types, list):
            types = " or ".join(types)
        elif "enum" in sub:
            types = "%s: %s" % (types, ", ".join(str(v) for v in sub["enum"]))
        out.append((path, types or "", key in required, sub.get("description", "")))
        if isinstance(sub.get("properties"), dict):
            out += schema_rows(sub, path + ".", sub.get("required", ()))
        elif isinstance(sub.get("items"), dict) and isinstance(sub["items"].get("properties"), dict):
            out += schema_rows(sub["items"], path + "[].", sub["items"].get("required", ()))
    return out


def submit_page(schema, date):
    rows = "".join('<tr><td><code>%s</code></td><td>%s</td><td>%s</td><td>%s</td></tr>'
                   % (esc(p), esc(t), "yes" if req else "", inline(d)) for p, t, req, d in
                   schema_rows(schema, "", schema.get("required", ())))
    body = """<h1>Submit a recipe</h1>
<p class="lede">OpenRecipes takes two kinds of contribution: a faster recipe for a model and GPU type, including a pair the catalog does not list yet, and a search procedure that finds recipes, such as a tuner, an agent or a set of rules. Submit a procedure through the recipes it finds; each is one recipe file, and its notes describe or link the procedure.</p>
<p>A recipe is one YAML file at <code>recipes/&lt;model-slug&gt;/&lt;gpu&gt;/&lt;handle&gt;-&lt;layout&gt;.yaml</code>: the layout, the command as run, the measured result, the stack it ran on, the correctness evidence and, where needed, notes. You run the commands below yourself and open a pull request with the file.</p>
<h2 id="steps">Steps</h2>
<ol class="prose">
<li><b>Pick the pair and the reference recipe.</b> Recipes rank within a model and GPU type, on any number of GPUs. The reference recipe is the one you tune from: a public recipe, the pair's current best or one you wrote. You need not publish it. Apply the <a href="protocol.html#correctness">objective conventions</a> to both.</li>
<li><b>Build the rows</b> with <a href="{repo}/tree/main/data"><code>data/</code></a> (its README gives the command per tokenizer) and check them with <code>python3 data/verify_rows.py &lt;rows.parquet&gt; --recipe &lt;a recipe of the same model&gt;</code>.</li>
<li><b>Measure.</b> Run the recipe once under the <a href="protocol.html#measurement">protocol</a>: 8,192-token rows, global batch 128, BF16, 20 steps with the profiler off, scored over steps 5&ndash;19. Record the score, the step-time CV, the GPU-hours and the framework versions.</li>
<li><b>Check correctness.</b> Run a 20-step arm of the reference recipe on the same image and rows, compare its per-step loss with the measured run's using Megatron-Bridge's <code>validate_convergence</code>, and class every argument that differs between the two. Each same-math swap needs a passed component test at the model's shape. OpenRecipes lists a recipe once all of them pass.</li>
<li><b>Write the recipe</b> from the template below. Count N with <code>python3 tools/active_params.py &lt;hf-repo&gt;</code>, take the rate from <a href="{repo}/blob/main/gpus.yaml"><code>gpus.yaml</code></a>, and add notes only for what is specific to the recipe: the steps every recipe shares are under <a href="protocol.html#run">Running a recipe</a>, and those a model's recipes share go in the model's <code>notes</code> in <code>models.yaml</code>. Add the model to <code>models.yaml</code> if it is missing, and put any code beyond the stock stack under <code>patches/</code>.</li>
<li><b>Validate and build.</b> <code>python3 tools/validate.py</code> checks the file against the schema and recomputes the cost, the MFU, the data-parallel size and the window; <code>python3 tools/build.py</code> rebuilds <code>site/</code>, and <code>python3 tools/build.py --check</code> and <code>python3 -m pytest -q tests</code> must pass. Commit the rebuilt <code>site/</code> with the recipe.</li>
<li><b>Open a pull request</b> with the recipe, any new patches and the rebuilt site.</li>
</ol>
<p>Leave out internal paths, image registry addresses, account or cluster names, and datasets you may not cite.</p>
<h2 id="template">The recipe file</h2>
<p>The full schema is <a href="schema/recipe.schema.yaml"><code>schema/recipe.schema.yaml</code></a>; every recipe in <a href="{repo}/tree/main/recipes"><code>recipes/</code></a> is a worked example.</p>
<div class="filebar"><span class="k">recipes/&lt;model-slug&gt;/&lt;gpu&gt;/&lt;handle&gt;-&lt;layout&gt;.yaml</span><span class="acts"><button type="button" class="btn o" data-copy="template">Copy</button></span></div>
<pre class="file" id="template">{template}</pre>
<h2 id="fields">Fields</h2>
<p class="muted small">Generated from the schema. tools/validate.py also checks the cross-field rules listed at the top of the schema file.</p>
<div class="card tablewrap"><table><thead><tr><th>Field</th><th>Type</th><th>Required</th><th>Meaning</th></tr></thead><tbody>{rows}</tbody></table></div>""".format(
        repo=REPO_URL, template=esc(TEMPLATE), rows=rows)
    return shell("submit.html", "Submit a recipe | OpenRecipes",
                 "How to submit a training recipe to OpenRecipes by pull request, and the recipe YAML schema.",
                 body, "submit", date)


def not_found_page(date):
    body = ('<h1>Page not found</h1><p class="lede">This page does not exist. The <a href="/index.html#models">model '
            'catalog</a> lists every model and GPU type with recipes.</p>')
    return shell("404.html", "Page not found | OpenRecipes", "Page not found.", body, None, date, absolute=True,
                 canonical=False)


# ---- build --------------------------------------------------------------------------------------------------------

def jsonable(v):
    if isinstance(v, dict):
        return {k: jsonable(x) for k, x in v.items()}
    if isinstance(v, list):
        return [jsonable(x) for x in v]
    if isinstance(v, datetime.date):
        return v.isoformat()
    return v


def render(root=ROOT):
    """(files {relative path: bytes}, errors)."""
    recipes, models, gpus, rates, errs = validate_all(root)
    if errs:
        return {}, errs
    pairs = collect(recipes, models, gpus)
    cite = citation(root)
    date = max(iso(r["result"]["measured"]) for r in recipes.values())
    files = {}

    def put(rel, text):
        files[rel] = text.encode("utf-8") if isinstance(text, str) else text

    put("index.html", index_page(pairs, models, gpus, date))
    for (mname, gt), items in sorted(pairs.items(), key=lambda kv: (kv[0][0], list(gpus).index(kv[0][1]))):
        m = models[mname]
        put(pair_url(m, gt) + "index.html", pair_page(m, gpus[gt], items, pairs, gpus, date, root, cite))
    put("protocol.html", protocol_page(recipes, gpus, rates, date))
    schema_path = Path(root) / "schema" / "recipe.schema.yaml"
    put("submit.html", submit_page(load_yaml(schema_path), date))
    put("404.html", not_found_page(date))
    for name in ("style.css", "app.js", "favicon.svg"):
        put(name, (ASSETS / name).read_bytes())
    for name in ("LICENSE", "NOTICE"):                   # the notices travel with the files the site serves
        put(name + ".txt", (Path(root) / name).read_bytes())
    put("schema/recipe.schema.yaml", schema_path.read_bytes())
    for path in sorted(recipes):
        put(path, (Path(root) / path).read_bytes())
    data = {
        "site": SITE_URL, "repository": REPO_URL,
        "rates": jsonable(rates), "gpus": jsonable(list(gpus.values())),
        "models": [dict(m, pairs=[{"gpu": gt, "url": SITE_URL + "/" + pair_url(m, gt),
                                   "recipes": [x["path"] for x in pairs[(m["model"], gt)]]}
                                  for gt in gpus if (m["model"], gt) in pairs]) for m in models.values()],
        "recipes": [dict(path=x["path"], slug=x["slug"], rank=x["rank"], **jsonable(x["r"]))
                    for key in sorted(pairs, key=lambda k: (k[0], list(gpus).index(k[1]))) for x in pairs[key]],
    }
    put("data.json", json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    pages = ["", "protocol", "submit"] + [
        pair_url(models[m], gt) for m, gt in sorted(pairs, key=lambda k: (k[0], list(gpus).index(k[1])))]
    put("sitemap.xml", '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + "".join("  <url><loc>%s/%s</loc></url>\n" % (SITE_URL, esc(p)) for p in pages) + "</urlset>\n")
    put("robots.txt", "User-agent: *\nAllow: /\nSitemap: %s/sitemap.xml\n" % SITE_URL)
    put("_headers", "/*\n  Content-Security-Policy: default-src 'self'; img-src 'self' data:; "
        "base-uri 'none'; form-action 'none'; frame-ancestors 'none'\n  X-Content-Type-Options: nosniff\n"
        "  Referrer-Policy: strict-origin-when-cross-origin\n")
    return files, []


def tree(path):
    path = Path(path)
    if not path.is_dir():
        return {}
    return {p.relative_to(path).as_posix(): p.read_bytes() for p in sorted(path.rglob("*")) if p.is_file()}


def write(files, out):
    out = Path(out)
    tmp = Path(tempfile.mkdtemp(prefix=".site-", dir=out.parent))
    for rel, data in files.items():
        p = tmp / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if a recipe is invalid or site/ is stale")
    ap.add_argument("--bibtex", action="store_true", help="print the preferred citation of CITATION.cff as BibTeX")
    ap.add_argument("--root", default=str(ROOT), help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    root = Path(a.root)
    if a.bibtex:
        sys.stdout.write(citation(root)[1])
        return 0
    files, errs = render(root)
    if errs:
        for e in errs:
            print(e)
        print("build stopped: %d problem(s)" % len(errs), file=sys.stderr)
        return 1
    site = root / "site"
    if a.check:
        have = tree(site)
        stale = sorted(k for k in set(files) | set(have) if files.get(k) != have.get(k))
        for k in stale:
            print("site/%s: %s" % (k, "missing" if k not in have else "not generated" if k not in files else "stale"))
        if stale:
            print("site/ is stale: run python3 tools/build.py", file=sys.stderr)
            return 1
        print("site/ is up to date (%d files)" % len(files), file=sys.stderr)
        return 0
    write(files, site)
    print("wrote site/ (%d files)" % len(files), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
