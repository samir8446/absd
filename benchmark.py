"""Offline cross-cell benchmark: forecast-origin sweep x cells x paradigms.

Examples
  python benchmark.py --master data/master.parquet --imp data/impedance.parquet
  python benchmark.py --synthetic --fracs 0.3 0.5 --paradigms ML Twin --out results/demo.parquet

Writes <out> (one row per cell / origin / paradigm), <out stem>_residuals.parquet (one row per
held-out cycle, for coverage-vs-horizon), <out stem>_summary.csv and <out stem>_manifest.json.
The Streamlit app can load the first file directly.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import twin_engine as te


def _save(df, path: Path) -> Path:
    """Parquet when pyarrow is available, CSV otherwise (the app reads both)."""
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception:
        alt = path.with_suffix(".csv")
        df.to_csv(alt, index=False)
        return alt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--master", type=Path, help="master telemetry Parquet")
    src.add_argument("--synthetic", action="store_true", help="use the synthetic ground-truth cohort")
    ap.add_argument("--imp", type=Path, help="impedance Parquet (optional)")
    ap.add_argument("--out", type=Path, default=Path("results/benchmark.parquet"))
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5, 0.6])
    ap.add_argument("--paradigms", nargs="+", default=list(te.BENCH_PARADIGMS), choices=te.BENCH_PARADIGMS)
    ap.add_argument("--cells", nargs="*", help="subset of cells (default: all with enough cycles)")
    ap.add_argument("--ml-model", default="Gradient Boosting", choices=te.ML_MODELS)
    ap.add_argument("--ml-strategy", default="increment", choices=te.ML_STRATEGIES)
    ap.add_argument("--conformal-cells", type=int, default=4)
    ap.add_argument("--band-level", type=float, default=0.9)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--eol-ah", type=float, default=te.DEFAULT_EOL_AH)
    ap.add_argument("--pinn-epochs", type=int, default=1000)
    ap.add_argument("--pinn-seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--ea-mode", default="fixed", choices=te.EA_MODES)
    ap.add_argument("--use-capacity", action="store_true", help="dual twin also uses capacity measurements")
    ap.add_argument("--min-cycles", type=int, default=30)
    args = ap.parse_args(argv)

    t0 = time.time()
    if args.synthetic:
        master, imp, _ = te.make_synthetic_master(n_cells=6, n_cycles=120, ambients=(24, 34, 43, 24, 34, 43),
                                                  k_true=(4e-4, 5e-4, 3.5e-4, 6e-4, 4.5e-4, 5e-4), seed=1)
        store = te.ParquetStore.from_dataframe(master)
    else:
        store = te.ParquetStore(args.master)
        imp = te.load_impedance(args.imp) if args.imp else None

    def progress(frac: float, msg: str) -> None:
        print(f"\r[{100 * frac:5.1f}%] {msg:<60}", end="", file=sys.stderr, flush=True)

    errors: dict = {}
    ct = te.build_cycle_table(store, progress=progress, errors=errors)
    print(file=sys.stderr)
    for cell, why in errors.items():
        print(f"warning: skipped {cell}: {why}", file=sys.stderr)

    cfg = te.BenchmarkConfig(fracs=tuple(args.fracs), paradigms=tuple(args.paradigms), ml_model=args.ml_model,
                             ml_strategy=args.ml_strategy, conformal_cells=args.conformal_cells,
                             band_level=args.band_level, alpha=args.alpha, eol_ah=args.eol_ah,
                             pinn_epochs=args.pinn_epochs, pinn_seeds=tuple(args.pinn_seeds),
                             ea_mode=args.ea_mode, use_capacity=args.use_capacity, min_cycles=args.min_cycles)
    bench, resid = te.run_benchmark(store, ct, imp, cfg, cells=args.cells, progress=progress)
    print(file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    stem = args.out.with_suffix("")
    p_bench = _save(bench, args.out)
    p_res = _save(resid, Path(f"{stem}_residuals.parquet"))
    summary = te.benchmark_summary(bench, cfg.alpha)
    summary.to_csv(f"{stem}_summary.csv")
    manifest = te.run_manifest(asdict(cfg), store.key,
                               {"cells": sorted(bench["cell"].unique().tolist()) if len(bench) else [],
                                "skipped_cells": errors, "runtime_s": round(time.time() - t0, 1),
                                "outputs": [str(p_bench), str(p_res)]})
    Path(f"{stem}_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(summary.round(4).to_string())
    print(f"\nwrote {p_bench}, {p_res}, {stem}_summary.csv, {stem}_manifest.json "
          f"({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
