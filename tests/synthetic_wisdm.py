"""A small synthetic dataset in the format of WISDM_at_v2.0_raw.txt, so the tests need no download.

    python tests/synthetic_wisdm.py OUT_FILE [--users 40] [--seconds 26]

Lines look like the real file: "user,activity,timestamp_ms,x,y,z;" at 20 Hz. Every user does
every activity for `seconds` seconds; the signal depends on the class (learnable).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

CLASSES = ["Walking", "Jogging", "Stairs", "Sitting", "Standing", "LyingDown"]
# per class: (frequency Hz, amplitude, gravity direction)
PROFILE = {0: (2.0, 4.0, (0, 9.8, 0)), 1: (3.0, 9.0, (0, 9.8, 0)), 2: (1.5, 3.0, (2, 9.5, 1)),
           3: (0.0, 0.2, (0, 4, 9)), 4: (0.0, 0.2, (0, 9.8, 0.5)), 5: (0.0, 0.2, (9, 1, 3))}


def make(path: Path, users: int = 40, seconds: int = 26, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    lines = []
    ts = 1_300_000_000_000
    n = seconds * 20
    for u in range(1, users + 1):
        for c in rng.permutation(len(CLASSES)):
            freq, amp, g = PROFILE[int(c)]
            t = np.arange(n) / 20
            phase = rng.uniform(0, 2 * np.pi)
            sig = np.stack([g[i] + amp * np.sin(2 * np.pi * freq * t + phase + i) for i in range(3)], 1)
            sig += rng.normal(0, 0.3 + 0.02 * u, size=sig.shape)  # users differ a little
            for k in range(n):
                lines.append(f"{u},{CLASSES[c]},{ts + 50 * k},{sig[k, 0]:.4f},{sig[k, 1]:.4f},{sig[k, 2]:.4f};")
            ts += 50 * n + 60_000  # a gap between activities: new segment
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--users", type=int, default=40)
    p.add_argument("--seconds", type=int, default=26)
    a = p.parse_args()
    make(a.out, a.users, a.seconds)
