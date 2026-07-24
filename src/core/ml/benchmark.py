"""Minimal benchmarking utilities for timed pipeline sections.

The key detail is that CUDA work is synchronized before and after a measured
block so reported timings reflect actual GPU execution rather than queued ops.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable, Optional, Dict, Any, Iterator

import torch


@contextmanager
def bench(
    enabled: bool,
    name: str,
    *,
    meta_fn: Optional[Callable[[], Dict[str, Any]]] = None,
) -> Iterator[None]:
    """Time a code block and print a compact benchmark line.

    When CUDA is available, the context synchronizes before and after the block
    so timings include completed GPU work. Returns a no-op context if disabled.
    """

    # Keep the disabled path effectively free apart from the guard itself.
    if not enabled:
        yield
        return

    # CUDA kernels are asynchronous, so wall-clock timing needs synchronization.
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    try:
        yield

    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        dt = time.perf_counter() - t0

        meta = ""

        if meta_fn is not None:
            try:
                m = meta_fn() or {}

                if m:
                    meta = " | " + " ".join(
                        f"{k}={v}" for k, v in m.items()
                    )

            except Exception:
                # Benchmarking should never interfere with the measured code path.
                meta = ""

        print(f"  [BENCH] {name}: {dt:.3f}s{meta}")
