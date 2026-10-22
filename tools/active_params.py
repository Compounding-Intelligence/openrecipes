#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified for OpenRecipes, 2026-10-14: paths and the protocol reference updated for this repository.
"""Count a checkpoint's active matmul parameters per token, the N of the MFU on the protocol page (protocol.html).

    python3 tools/active_params.py <hf-repo-id | local checkpoint dir> [--revision R]

Reads only the safetensors headers (shapes and dtypes), fetched with HTTP range requests for a Hub repo. N counts every
weight with two or more dimensions that the text model applies to each token: attention, linear attention, indexer,
dense and shared-expert MLPs, routers and the output projection; routed experts count num_experts_per_tok (Gemma 4's
top_k_experts) of num_experts. It leaves out the input embedding (a lookup), one-dimensional tensors and biases,
position embeddings and 64-bit integer index tables (added or looked up, not multiplied), multi-token prediction layers,
vision and audio towers with their projections and quantization scales, none of which an entry's text-only SFT run
applies to its tokens as a matmul.

A quantized integer weight pairs with its scale tensor (X.weight with X.scale, X.weight_scale and the like; X_blocks
with X_scales). It holds FP4 values packed two per byte, and counts at twice its stored size, when the scale is one byte
per block of 16 or 32 values ([..., n, k] weight bytes with [..., 2k/B] or [..., n] scales); otherwise it holds one
value per byte. A compressed-tensors pack-quantized weight (X.weight_packed with X.weight_scale, as Kimi K2.5/K2.6's
INT4 experts) holds 32 / num_bits values per 32-bit integer, with num_bits from the config's quantization_config. The
count stops with an error on any other integer tensor of two or more dimensions, and on routed experts without an
expert count in the config. It also stops when the stored bytes differ from the index's total_size, unless the index's
weight map names exactly the tensors it read: some indexes record total_size another way (GLM-4.7-Flash's counts
elements). Prints JSON.
"""
import argparse
import json
import os
import re
import struct
import subprocess
import sys
import time

ROUTED = re.compile(r"\.experts\.")
SKIP = (re.compile(r"(^|\.)visual\."), re.compile(r"(^|\.)vision_tower\."), re.compile(r"(^|\.)mtp\."),
        re.compile(r"(^|\.)(audio_tower|vision_adapter|vision_projection)\."),   # Muse Spark's other modalities
        re.compile(r"(^|\.)embed_(vision|audio)\."),   # Gemma's projections of image and audio tokens
        re.compile(r"(^|\.)audio\."),   # Inkling's audio projection
        re.compile(r"(^|\.)mm_projector\."),   # Kimi K2.5/K2.6's vision projection
        re.compile(r"scales?(_inv)?$"))   # quantization scales of FP8/FP4/INT8 checkpoints
EMBED = re.compile(r"(^|\.)(embed|embed_tokens|embeddings|wte|word_embeddings|embed_in|tok_embeddings)\.weight$")
POSITION = re.compile(r"(^|\.)(ape|pos_embed|position_embeddings?|wpe)(\.weight)?$")   # added to activations
OUTPUT = re.compile(r"(^|\.)(lm_)?head\.weight$")
INTEGER = ("I8", "U8", "I16", "U16", "I32", "U32", "I64", "U64")
INDEX = ("I64",)   # 64-bit integers hold indices (token-to-expert tables), never packed weights
COMPANIONS = {".weight": (".scale", ".scale_inv", ".weight_scale", ".weight_scale_inv"), "_blocks": ("_scales",),
              ".weight_packed": (".weight_scale",)}
FP4_SCALES = ("F8_E8M0", "F8_E4M3", "U8")   # one-byte block scales (MXFP4, NVFP4; gpt-oss stores E8M0 as U8)
BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1, "I64": 8, "U64": 8,
         "I32": 4, "U32": 4, "I16": 2, "U16": 2, "I8": 1, "U8": 1, "BOOL": 1}


def _fetch(src, rev, name, rng=None):
    if os.path.isdir(src):
        with open(os.path.join(src, name), "rb") as f:
            if rng is None:
                return f.read()
            f.seek(rng[0])
            return f.read(rng[1] - rng[0] + 1)
    url = "https://huggingface.co/%s/resolve/%s/%s" % (src, rev, name)
    cmd = ["curl", "-sSL", "--max-time", "120", "-w", "%{stderr}%{http_code}", url] + (["-r", "%d-%d" % rng] if rng else [])
    for attempt in range(10):
        r = subprocess.run(cmd, capture_output=True)
        code = r.stderr.decode()[-3:]
        if r.returncode == 0 and code in ("200", "206"):
            return r.stdout
        if r.returncode == 0 and code not in ("429",) and not code.startswith("5"):
            raise FileNotFoundError("%s: HTTP %s" % (url, code))
        time.sleep(5 * (attempt + 1))   # the Hub answers 429 when its queue is full
    sys.exit("fetch failed: %s" % url)


def headers(src, rev):
    try:
        index = json.loads(_fetch(src, rev, "model.safetensors.index.json"))
    except FileNotFoundError:
        index = {"weight_map": {"": "model.safetensors"}}   # a single-file checkpoint
    for shard in sorted(set(index["weight_map"].values())):
        n = struct.unpack("<Q", _fetch(src, rev, shard, (0, 7)))[0]
        yield shard, json.loads(_fetch(src, rev, shard, (8, 7 + n)))
    yield None, index


def values_per_element(name, t, tensors, packed=None):
    """2 for an FP4 weight packed two per byte, `packed` for a pack-quantized one, 1 for another quantized integer
    weight, None without a scale tensor."""
    for suffix, scales in COMPANIONS.items():
        if not name.endswith(suffix):
            continue
        for sfx in scales:
            sc = tensors.get(name[:-len(suffix)] + sfx)
            if sc is None:
                continue
            if suffix == ".weight_packed":
                return packed if t["dtype"] == "I32" else None
            w, s = t["shape"], sc["shape"]
            blocks = s == w[:-1] and 2 * w[-1] in (16, 32)
            rows = len(s) == len(w) and s[:-1] == w[:-1] and any(2 * w[-1] == s[-1] * b for b in (16, 32))
            return 2 if t["dtype"] in ("I8", "U8") and sc["dtype"] in FP4_SCALES and (blocks or rows) else 1
    return None


def pack_factor(cfg, text):
    """Values per 32-bit integer in a compressed-tensors pack-quantized checkpoint (32 / num_bits), else None."""
    quant = text.get("quantization_config") or cfg.get("quantization_config") or {}
    bits = {(g.get("weights") or {}).get("num_bits") for g in (quant.get("config_groups") or {}).values()}
    if quant.get("format") != "pack-quantized" or len(bits) != 1:
        return None
    (n,) = bits
    return 32 // n if isinstance(n, int) and n > 0 else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src")
    ap.add_argument("--revision", default="main")
    a = ap.parse_args(argv)
    cfg = json.loads(_fetch(a.src, a.revision, "config.json"))
    text = cfg.get("text_config", cfg)
    layers = text["num_hidden_layers"]
    k = (text.get("num_experts_per_tok") or text.get("experts_per_token") or text.get("top_k_experts")
         or 1)   # dense models have no routed experts
    e = text.get("n_routed_experts") or text.get("num_experts") or text.get("num_local_experts") or 1
    packed = pack_factor(cfg, text)
    total = nbytes = active = 0
    parts = {"routed_experts": 0, "other_matmul": 0, "output": 0, "excluded": 0}
    embed = 0
    tensors = {}
    for shard, h in headers(a.src, a.revision):
        if shard is None:
            index = h
            continue
        tensors.update((name, t) for name, t in h.items() if name != "__metadata__")
    for name, t in tensors.items():
        numel = 1
        for d in t["shape"]:
            numel *= d
        nbytes += numel * BYTES[t["dtype"]]   # stored bytes, checked against the index's total_size
        m = re.search(r"\.layers\.(\d+)\.", name)
        skipped = any(p.search(name) for p in SKIP) or bool(m and int(m.group(1)) >= layers)
        index_table = False
        if t["dtype"] in INTEGER and len(t["shape"]) >= 2 and not skipped:
            per = values_per_element(name, t, tensors, packed)
            if per is None and t["dtype"] not in INDEX:
                sys.exit("%s: %s tensor %s has no scale tensor and is not an index table" % (name, t["dtype"], t["shape"]))
            index_table = per is None
            numel *= per or 1
        total += numel
        if EMBED.search(name) and not skipped:
            embed = numel
        if (len(t["shape"]) < 2 or skipped or EMBED.search(name) or POSITION.search(name) or name.endswith("bias")
                or index_table):
            parts["excluded"] += numel
        elif ROUTED.search(name):
            if e == 1:
                sys.exit("%s: routed experts, but the config gives no expert count" % name)
            parts["routed_experts"] += numel * k / e
        elif OUTPUT.search(name):
            parts["output"] += numel
        else:
            parts["other_matmul"] += numel
    if not parts["output"] and cfg.get("tie_word_embeddings", text.get("tie_word_embeddings")):
        parts["output"] = embed   # tied embeddings: the output projection reuses the embedding matrix
    active = parts["routed_experts"] + parts["other_matmul"] + parts["output"]
    want = index.get("metadata", {}).get("total_size")
    if want is not None and want != nbytes and set(index["weight_map"]) != set(tensors):
        sys.exit("header bytes %d != index total_size %d" % (nbytes, want))
    print(json.dumps({"source": a.src, "revision": a.revision, "total_params": total, "active_params": round(active),
                      "experts_per_token": k, "routed_experts": e,
                      "parts": {p: round(v) for p, v in parts.items()}}, indent=1))


if __name__ == "__main__":
    main()
