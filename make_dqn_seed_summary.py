"""Create the independent DQN seed ledger used by the T6 sampling run.

The original ``seed_summary.csv`` is an observed sampling ledger.  The new
DQN pipeline has its own random stream, so it must be recorded separately.
This script derives one reproducible 500-seed sequence per T6 cell from the
fixed train-year-3 DQN seed for that market.  The generated CSV is an input
artifact, not a source of randomness at runtime.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


CODE_DIR = Path(__file__).resolve().parent
OUT = CODE_DIR / "temp" / "dqn_seed_summary.csv"


def main() -> None:
    rows: dict[str, list[int]] = {}
    # These are the calibrated, paper-compatible DQN training seeds.  T6 uses
    # the three-year ranker and therefore shares this base seed by market.
    for prefix, base in (("T6M", 40), ("T6C", 50)):
        for rate_digit in range(5, 10):
            rng = np.random.default_rng(base + rate_digit * 1000)
            values = rng.integers(1, 2_147_483_647, size=500, dtype=np.int64)
            rows[f"{prefix}{rate_digit}_DQN"] = [int(value) for value in values]

    columns = ["name"] + [f"seed_{index}" for index in range(1, 501)]
    frame = pd.DataFrame(
        [{"name": name, **{f"seed_{i}": value for i, value in enumerate(values, 1)}}
         for name, values in rows.items()],
        columns=columns,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False, encoding="utf-8-sig")
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    OUT.with_suffix(".json").write_text(
        json.dumps(
            {
                "file": str(OUT),
                "sha256": digest,
                "rows": len(frame),
                "seeds_per_row": 500,
                "base_seeds": {"Main": 40, "ChiNext": 50},
                "generation": "numpy.default_rng(base_seed + rate_digit * 1000)",
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    print(f"wrote {OUT} ({len(frame)} rows x {len(frame.columns)} columns)")


if __name__ == "__main__":
    main()
