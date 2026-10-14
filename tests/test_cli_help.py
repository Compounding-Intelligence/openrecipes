# Copyright 2026 Compounding Intelligence
# SPDX-License-Identifier: Apache-2.0
"""Each command-line tool answers --help with its usage and exit status 0, with only the standard library and PyYAML."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLIS = ["data/generate_fixed_sft.py", "data/patch_glm5_mask.py", "data/patch_gptoss_mask.py",
        "data/verify_rows.py", "tools/active_params.py", "tools/build.py",
        "tools/validate.py"]


@pytest.mark.parametrize("cli", CLIS)
def test_help(cli):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([sys.executable, cli, "--help"], capture_output=True, text=True, cwd=ROOT, env=env, timeout=60)
    assert p.returncode == 0 and "Traceback" not in p.stderr, p.stderr
    assert "usage:" in p.stdout, p.stdout
