#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified for OpenRecipes, 2026-10-14: checks against a recipe YAML (--recipe) instead of an entry.json.
"""Check a built row file against the dataset build a recipe records.

    python3 verify_rows.py train.parquet --recipe ../recipes/qwen3.5-122b-a10b/h100/attune-tp4-pp6-ep8-flex-mt16k.yaml

Every row must hold exactly row_tokens tokens and row_tokens - 1 binary target-mask entries with at least one
supervised target, and its stored row_hash must equal the hash of its tokens and mask (generate_fixed_sft.py's
row_digest). rows_sha256 is the sha256 of the rows' row_hash values in file order, one per line: it matches when the
rows match, whatever Parquet writer produced the file. parquet_sha256 matches only a byte-identical file.
"""
import argparse
import hashlib
import json
import sys
from array import array


def row_digest(tokens, loss_mask):
    digest = hashlib.sha256()
    digest.update(array("i", tokens).tobytes())
    digest.update(bytes(loss_mask))
    return digest.hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def scan(path, row_tokens):
    import pyarrow.parquet as pq

    rows, errors, rows_digest = 0, [], hashlib.sha256()
    for batch in pq.ParquetFile(path).iter_batches(columns=["metadata"], batch_size=256):
        for md in batch.column(0).to_pylist():
            tokens, mask, stored = md["tokens"], md["loss_mask"], md["row_hash"]
            if len(tokens) != row_tokens or len(mask) != row_tokens - 1:
                errors.append(f"row {rows}: {len(tokens)} tokens, {len(mask)} mask entries")
            elif any(v not in (0, 1) for v in mask) or not any(mask):
                errors.append(f"row {rows}: mask is not binary or has no supervised target")
            elif row_digest(tokens, mask) != stored:
                errors.append(f"row {rows}: row_hash does not match its tokens and mask")
            rows_digest.update((stored + "\n").encode())
            rows += 1
    return rows, rows_digest.hexdigest(), errors


def expected_build(recipe_path):
    """The rows, rows_sha256, parquet_sha256 and row_tokens a recipe YAML records for its dataset (workload.dataset)."""
    import yaml

    with open(recipe_path) as f:
        dataset = yaml.safe_load(f)["workload"]["dataset"]
    build = dataset.get("build") or {}
    want = {k: build[k] for k in ("rows", "parquet_sha256", "row_tokens") if k in build}
    if "rows_sha256" in dataset:
        want["rows_sha256"] = dataset["rows_sha256"]
    return want


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("parquet")
    ap.add_argument("--recipe", help="recipe YAML whose workload.dataset to check against (needs PyYAML)")
    ap.add_argument("--row-tokens", type=int, default=8192)
    a = ap.parse_args()
    want = {}
    if a.recipe:
        want = expected_build(a.recipe)
        a.row_tokens = want.get("row_tokens", a.row_tokens)
    rows, rows_sha256, errors = scan(a.parquet, a.row_tokens)
    got = {"rows": rows, "rows_sha256": rows_sha256, "parquet_sha256": file_sha256(a.parquet)}
    print(json.dumps(dict(got, row_errors=len(errors)), indent=1))
    for e in errors[:10]:
        print("error:", e, file=sys.stderr)
    failed = bool(errors)
    for key in ("rows", "rows_sha256", "parquet_sha256"):
        if key in want:
            ok = want[key] == got[key]
            print(f"{key}: {'match' if ok else 'MISMATCH'}")
            # a parquet_sha256 mismatch alone means another writer produced the same rows
            failed |= not ok and key != "parquet_sha256"
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
