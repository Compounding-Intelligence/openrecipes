# SPDX-License-Identifier: Apache-2.0
"""Keep FLA's gated-delta-rule kernels off the Triton autotune configs that race on NVIDIA Blackwell.

With Triton below 3.8, Triton miscompiles a `tl.dot` recurrence on Blackwell, and some autotune configs of FLA's
gated delta rule return a different, wrong result on every call (triton-lang/triton#10590, fixed in Triton 3.8;
fla-org/flash-linear-attention#945, #1228, #1327). FLA 0.4.1 keeps those configs, and its autotuner picks them for
Qwen3.5 shapes on B200. Importing this module applies FLA main's restriction to the installed FLA when its kernel
modules load:

- `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`: 2 warps only (fla-org/flash-linear-attention#953);
- `chunk_fwd_kernel_o`: no BK=BV=128 config at 8 warps, below Triton 3.8 (#1229, #1399);
- `chunk_bwd_kernel_dv_local`: at most 4 warps, below Triton 3.8 (#1399).

Other GPUs are left unchanged. Every process that imports these FLA modules prints one `[fla-guard]` line per module.

Usage: import this module in every training process before FLA is imported, for example with this directory on
`PYTHONPATH` and `import fla_blackwell_guard` in a `sitecustomize.py` on the same path.
"""

import importlib.machinery
import os
import re
import socket
import sys

# module -> [(kernel, needs Triton < 3.8, keep(config))]
_RULES = {
    "fla.ops.common.chunk_delta_h": [
        ("chunk_gated_delta_rule_fwd_kernel_h_blockdim64", False, lambda c: c.num_warps == 2),
    ],
    "fla.ops.common.chunk_o": [
        ("chunk_fwd_kernel_o", True,
         lambda c: not (c.num_warps == 8 and c.kwargs.get("BK") == 128 and c.kwargs.get("BV") == 128)),
        ("chunk_bwd_kernel_dv_local", True, lambda c: c.num_warps <= 4),
    ],
}


def _log(msg):
    sys.stderr.write("[fla-guard] %s (host %s, pid %d)\n" % (msg, socket.gethostname(), os.getpid()))
    sys.stderr.flush()


def _autotuner(fn):
    from triton.runtime.autotuner import Autotuner

    while not isinstance(fn, Autotuner):
        fn = fn.fn
    return fn


def _apply(module):
    import torch
    import triton

    name = module.__name__
    if not torch.cuda.is_available():
        _log("%s unchanged: no CUDA device" % name)
        return
    major, minor = torch.cuda.get_device_capability()
    if major not in (10, 12):
        _log("%s unchanged: sm%d%d is not Blackwell" % (name, major, minor))
        return
    old_triton = tuple(int(x) for x in re.findall(r"\d+", triton.__version__)[:2]) < (3, 8)
    done = []
    for kernel, needs_old_triton, keep in _RULES[name]:
        if needs_old_triton and not old_triton:
            continue
        tuner = _autotuner(getattr(module, kernel))
        kept = [c for c in tuner.configs if keep(c)]
        if not kept:
            raise RuntimeError("fla-guard: no config of %s.%s left" % (name, kernel))
        done.append("%s %d->%d configs" % (kernel, len(tuner.configs), len(kept)))
        tuner.configs = kept
        tuner.cache.clear()
    _log("%s on sm%d%d, triton %s: %s" % (name, major, minor, triton.__version__, ", ".join(done) or "unchanged"))


class _Finder:
    """Runs _apply right after the target module's own code."""

    @staticmethod
    def find_spec(name, path=None, target=None):
        if name not in _RULES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return None
        exec_module = spec.loader.exec_module

        def run(module):
            exec_module(module)
            _apply(module)

        spec.loader.exec_module = run
        return spec


if not getattr(sys, "_fla_blackwell_guard", False):
    sys._fla_blackwell_guard = True
    sys.meta_path.insert(0, _Finder)
    for _name in _RULES:
        if _name in sys.modules:
            _apply(sys.modules[_name])
