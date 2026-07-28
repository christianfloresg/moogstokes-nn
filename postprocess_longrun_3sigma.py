#!/usr/bin/env python
"""Regenerate final-stage MCMC summaries and corner plots with 3-sigma intervals.

This does not rerun any neural-net models or MCMC sampling. It only reads the
saved stage-2 .npz files from the long batch run.
"""

from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path

import numpy as np

from mcmc import (
    PARAM_ORDER,
    flat_samples_from_chain,
    load_mcmc_run,
    normalize_fixed_params,
    percentile_summary,
    summary_values_from_free,
)
from mcmc_plots import save_corner_plot


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_STATUS_CSV = PROJECT_ROOT / "mcmc_runs" / "batch_logs" / "values_longrun_batch_status.csv"
DEFAULT_SUMMARY_CSV = PROJECT_ROOT / "mcmc_runs" / "values_longrun_stage2_3sigma_summary.csv"


def read_stage2_run_files(status_csv: Path) -> list[dict]:
    rows = []
    with status_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "success":
                continue
            run_file = row.get("stage2_run_file", "").strip()
            if not run_file:
                continue
            rows.append({
                "filename": row.get("filename", ""),
                "include_reg0": row.get("include_reg0", ""),
                "vsini_rule": row.get("vsini_rule", ""),
                "stage1_vsini": row.get("stage1_vsini", ""),
                "stage2_regions": row.get("stage2_regions", ""),
                "stage2_fixed_params": row.get("stage2_fixed_params", ""),
                "stage2_run_file": run_file,
            })
    return rows


def write_summary(summary_csv: Path, rows: list[dict]) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "filename",
        "run_name",
        "stage",
        "include_reg0",
        "vsini_rule",
        "stage1_vsini",
        "stage2_regions",
        "stage2_fixed_params",
        "mcmc_run_file",
        "corner_plot",
        "nsteps",
        "nwalkers",
        "n_free_params",
        "n_flat_samples",
        "discard",
        "thin",
        "best_percentile",
        "lower_percentile",
        "upper_percentile",
    ]
    for param in PARAM_ORDER:
        fieldnames.extend([
            param,
            f"{param}_err_minus",
            f"{param}_err_plus",
            f"{param}_status",
        ])

    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def postprocess_one(row: dict, args: argparse.Namespace) -> dict:
    run_file = Path(row["stage2_run_file"])
    if not run_file.exists():
        raise FileNotFoundError(run_file)

    payload = load_mcmc_run(run_file)
    metadata = payload["metadata"]
    chain = payload["chain"]
    free_param_names = list(payload["free_param_names"])
    fixed_params = normalize_fixed_params(payload["fixed_params"])

    flat_samples = flat_samples_from_chain(chain, discard=args.discard, thin=args.thin)
    best_values, bounds = percentile_summary(
        flat_samples,
        best_percentile=args.best_percentile,
        uncertainty_percentiles=(args.lower_percentile, args.upper_percentile),
    )
    values = summary_values_from_free(best_values, bounds, free_param_names, fixed_params)

    plot_metadata = metadata.copy()
    plot_metadata.update({
        "discard": args.discard,
        "thin": args.thin,
        "best_percentile": args.best_percentile,
        "uncertainty_percentiles": [args.lower_percentile, args.upper_percentile],
    })

    corner_plot = save_corner_plot(
        flat_samples=flat_samples,
        param_names=free_param_names,
        outdir=args.figures_dir,
        run_name=metadata["run_name"],
        metadata=plot_metadata,
        discard=args.discard,
        thin=args.thin,
        timestamp=args.timestamp,
        quantiles=[
            args.lower_percentile / 100.0,
            args.best_percentile / 100.0,
            args.upper_percentile / 100.0,
        ],
    )

    out = {
        "filename": row["filename"],
        "run_name": metadata["run_name"],
        "stage": metadata.get("stage_label", "stage2"),
        "include_reg0": row["include_reg0"],
        "vsini_rule": row["vsini_rule"],
        "stage1_vsini": row["stage1_vsini"],
        "stage2_regions": row["stage2_regions"],
        "stage2_fixed_params": row["stage2_fixed_params"],
        "mcmc_run_file": str(run_file),
        "corner_plot": corner_plot,
        "nsteps": chain.shape[0],
        "nwalkers": chain.shape[1],
        "n_free_params": chain.shape[2],
        "n_flat_samples": flat_samples.shape[0],
        "discard": args.discard,
        "thin": args.thin,
        "best_percentile": args.best_percentile,
        "lower_percentile": args.lower_percentile,
        "upper_percentile": args.upper_percentile,
    }
    for param in PARAM_ORDER:
        out[param] = values[param]
        out[f"{param}_err_minus"] = values[f"{param}_err_minus"]
        out[f"{param}_err_plus"] = values[f"{param}_err_plus"]
        out[f"{param}_status"] = values[f"{param}_status"]
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-csv", default=str(DEFAULT_STATUS_CSV))
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY_CSV))
    parser.add_argument("--figures-dir", default=str(PROJECT_ROOT / "figures"))
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_3sigma"))
    parser.add_argument("--discard", type=int, default=1000)
    parser.add_argument("--thin", type=int, default=5)
    parser.add_argument("--best-percentile", type=float, default=50.0)
    parser.add_argument("--lower-percentile", type=float, default=0.135)
    parser.add_argument("--upper-percentile", type=float, default=99.865)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", nargs="*", default=None)
    return parser.parse_args()


def main() -> int:
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/moogstokes_mplcache")
    args = parse_args()
    args.status_csv = Path(args.status_csv).expanduser().resolve()
    args.summary_csv = Path(args.summary_csv).expanduser().resolve()
    args.figures_dir = str(Path(args.figures_dir).expanduser().resolve())

    rows = read_stage2_run_files(args.status_csv)
    if args.only:
        selected = set(args.only)
        rows = [row for row in rows if row["filename"] in selected]
    if args.limit is not None:
        rows = rows[: args.limit]

    print(f"Loaded {len(rows)} stage-2 saved run(s) from {args.status_csv}")
    print(
        f"Using median plus central {args.lower_percentile}-{args.upper_percentile} "
        f"percentile interval"
    )

    output_rows = []
    for index, row in enumerate(rows, start=1):
        print(f"[{index}/{len(rows)}] {row['filename']}")
        output_rows.append(postprocess_one(row, args))

    write_summary(args.summary_csv, output_rows)
    print(f"Wrote summary: {args.summary_csv}")
    print(f"Wrote {len(output_rows)} corner plot(s) to {args.figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
