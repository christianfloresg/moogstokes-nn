#!/usr/bin/env python
"""Batch runner for the Class III two-stage MCMC notebook workflow.

This script mirrors the long-run settings in run_two_stage_mcmc_classiii.ipynb
and varies only the stage-2 regions and vsini behavior from values_for_long_runs.
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

from nn_helpers import MoogStokesNN
from mcmc_pipeline import postprocess_stage_result, run_stage_raw


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TABLE = PROJECT_ROOT / "values_for_long_runs.numbers"
DEFAULT_DATA_PATH = "data/science/ClassIII-sources-spectra"
DEFAULT_BATCH_DIR = PROJECT_ROOT / "mcmc_runs" / "batch_logs"


def export_numbers_to_csv(numbers_path: Path) -> Path:
    """Export an Apple Numbers file to a temporary CSV using Numbers.app."""
    outdir = Path(tempfile.mkdtemp(prefix="values_for_long_runs_"))
    csv_path = outdir / "values_for_long_runs.csv"
    script = f'''
set inputPath to POSIX file "{numbers_path}"
set outputPath to POSIX file "{csv_path}"
tell application "Numbers"
    set theDoc to open inputPath
    export theDoc to outputPath as CSV
    close theDoc saving no
end tell
'''
    subprocess.run(["osascript", "-e", script], check=True)
    return csv_path


def table_to_csv_path(table_path: Path) -> Path:
    suffix = table_path.suffix.lower()
    if suffix == ".numbers":
        return export_numbers_to_csv(table_path)
    if suffix == ".csv":
        return table_path
    raise ValueError(f"Unsupported table format: {table_path}. Use .numbers or .csv.")


def read_batch_rows(table_path: Path) -> list[dict]:
    csv_path = table_to_csv_path(table_path)
    rows = []
    with csv_path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"No rows found in {csv_path}")
        for line_number, raw in enumerate(reader, start=2):
            first_three = [(raw[i].strip() if i < len(raw) else "") for i in range(3)]
            if not any(first_three):
                continue
            filename, include_reg0, vsini = first_three
            if not filename:
                continue
            include_reg0_norm = include_reg0.lower()
            if include_reg0_norm not in {"yes", "no"}:
                raise ValueError(
                    f"Line {line_number}: Include Reg 0 must be yes/no, got {include_reg0!r}"
                )
            vsini_norm = vsini.lower()
            if vsini_norm not in {"step1", "free"}:
                try:
                    float(vsini)
                except ValueError as exc:
                    raise ValueError(
                        f"Line {line_number}: vsini must be step1, free, or a number; got {vsini!r}"
                    ) from exc
            rows.append({
                "line_number": line_number,
                "filename": filename,
                "include_reg0": include_reg0_norm,
                "vsini": vsini,
                "stage2_regions": [0, 1, 2, 3, 4, 5]
                if include_reg0_norm == "yes"
                else [1, 2, 3, 4, 5],
            })
    return rows


def stage2_fixed_params(vsini_value: str, vsini_stage1: float) -> dict:
    value = vsini_value.strip()
    lower = value.lower()
    if lower == "free":
        return {}
    if lower == "step1":
        return {"vsini": float(vsini_stage1)}
    return {"vsini": float(value)}


def read_completed(status_csv: Path) -> set[str]:
    if not status_csv.exists():
        return set()
    completed = set()
    with status_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "success":
                completed.add(row.get("filename", ""))
    return completed


def append_status(status_csv: Path, row: dict) -> None:
    status_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "status",
        "filename",
        "include_reg0",
        "vsini_rule",
        "stage1_vsini",
        "stage2_regions",
        "stage2_fixed_params",
        "stage1_run_file",
        "stage2_run_file",
        "stage1_trace_plot",
        "stage2_trace_plot",
        "summary_csv",
        "error",
    ]
    file_exists = status_csv.exists()
    with status_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def run_one_source(row: dict, moognn: MoogStokesNN, args: argparse.Namespace) -> dict:
    basename = row["filename"]
    source_name = Path(basename).stem
    run_prefix = args.run_prefix

    stage1_label = "all regions"
    stage2_label = "stage 2 no-lit high vsini "

    stage1_out = run_stage_raw(
        basename=basename,
        moognn=moognn,
        data_path=args.data_path,
        regions=[0, 1, 2, 3, 4, 5, 6],
        fixed_params={},
        stage_label=stage1_label,
        run_name=f"{run_prefix}_{source_name}_stage1_all_regions",
        nwalkers=args.nwalkers,
        nsteps=args.nsteps,
        scale_discard=args.scale_discard,
        scale_thin=args.scale_thin,
        progress=args.progress,
        mcmc_runs_dir=args.mcmc_runs_dir,
        save_mcmc_run_file=True,
    )

    out1 = postprocess_stage_result(
        stage1_out,
        moognn=moognn,
        data_path=args.data_path,
        discard=args.discard,
        thin=args.thin,
        save_fits=False,
        save_plots=args.save_plots,
        mcmc_runs_dir=args.mcmc_runs_dir,
        figures_dir=args.figures_dir,
        best_percentile=args.best_percentile,
        uncertainty_percentiles=(args.lower_percentile, args.upper_percentile),
    )

    vsini_stage1 = out1["values"]["vsini"]
    fixed_params = stage2_fixed_params(row["vsini"], vsini_stage1)

    stage2_out = run_stage_raw(
        basename=basename,
        moognn=moognn,
        data_path=args.data_path,
        regions=row["stage2_regions"],
        fixed_params=fixed_params,
        stage_label=stage2_label,
        run_name=f"{run_prefix}_{source_name}_stage2",
        nwalkers=args.nwalkers,
        nsteps=args.nsteps,
        scale_discard=args.scale_discard,
        scale_thin=args.scale_thin,
        progress=args.progress,
        mcmc_runs_dir=args.mcmc_runs_dir,
        save_mcmc_run_file=True,
    )

    out2 = postprocess_stage_result(
        stage2_out,
        moognn=moognn,
        data_path=args.data_path,
        discard=args.discard,
        thin=args.thin,
        save_fits=False,
        save_plots=args.save_plots,
        mcmc_runs_dir=args.mcmc_runs_dir,
        figures_dir=args.figures_dir,
        best_percentile=args.best_percentile,
        uncertainty_percentiles=(args.lower_percentile, args.upper_percentile),
    )

    result = {
        "status": "success",
        "filename": basename,
        "include_reg0": row["include_reg0"],
        "vsini_rule": row["vsini"],
        "stage1_vsini": vsini_stage1,
        "stage2_regions": row["stage2_regions"],
        "stage2_fixed_params": fixed_params,
        "stage1_run_file": out1["mcmc_run_file"],
        "stage2_run_file": out2["mcmc_run_file"],
        "stage1_trace_plot": out1["trace_plot"],
        "stage2_trace_plot": out2["trace_plot"],
        "summary_csv": out2["summary_csv"],
    }

    del stage1_out, out1, stage2_out, out2
    gc.collect()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default=str(DEFAULT_TABLE), help="Path to .numbers or .csv batch table.")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--mcmc-runs-dir", default="mcmc_runs")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--batch-dir", default=str(DEFAULT_BATCH_DIR))
    parser.add_argument("--run-prefix", default="longrun")
    parser.add_argument("--nwalkers", type=int, default=64)
    parser.add_argument("--nsteps", type=int, default=2000)
    parser.add_argument("--scale-discard", type=int, default=1000)
    parser.add_argument("--scale-thin", type=int, default=5)
    parser.add_argument("--discard", type=int, default=1000)
    parser.add_argument("--thin", type=int, default=5)
    parser.add_argument("--best-percentile", type=float, default=50)
    parser.add_argument("--lower-percentile", type=float, default=16)
    parser.add_argument("--upper-percentile", type=float, default=84)
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N rows.")
    parser.add_argument("--only", nargs="*", default=None, help="Only run these .nspec filenames.")
    parser.add_argument("--resume", action="store_true", help="Skip filenames already marked success.")
    parser.add_argument("--dry-run", action="store_true", help="Print parsed run plan without running MCMC.")
    parser.add_argument("--progress", action="store_true", help="Show emcee progress bars.")
    parser.add_argument("--no-plots", dest="save_plots", action="store_false", help="Skip plot generation.")
    parser.set_defaults(save_plots=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    table_path = Path(args.table).expanduser().resolve()
    batch_dir = Path(args.batch_dir).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    status_csv = batch_dir / f"{args.run_prefix}_batch_status.csv"

    rows = read_batch_rows(table_path)
    if args.only:
        selected = set(args.only)
        rows = [row for row in rows if row["filename"] in selected]
    if args.limit is not None:
        rows = rows[: args.limit]

    completed = read_completed(status_csv) if args.resume else set()
    print(f"Loaded {len(rows)} planned run(s) from {table_path}")
    for idx, row in enumerate(rows, start=1):
        fixed_preview = row["vsini"] if row["vsini"].lower() != "step1" else "stage1 vsini"
        skip = row["filename"] in completed
        print(
            f"{idx:03d}: {row['filename']} | regions={row['stage2_regions']} | "
            f"stage2 vsini={fixed_preview} | {'SKIP' if skip else 'RUN'}"
        )

    if args.dry_run:
        return 0

    moognn = MoogStokesNN(nn_models_dir="data/", regions=range(7))
    for idx, row in enumerate(rows, start=1):
        if row["filename"] in completed:
            continue
        print(f"\n[{idx}/{len(rows)}] Starting {row['filename']} at {datetime.now().isoformat(timespec='seconds')}")
        status = {
            "timestamp": timestamp,
            "filename": row["filename"],
            "include_reg0": row["include_reg0"],
            "vsini_rule": row["vsini"],
        }
        try:
            status.update(run_one_source(row, moognn, args))
            print(f"[{idx}/{len(rows)}] Finished {row['filename']}")
        except Exception:
            err = traceback.format_exc()
            status.update({"status": "failed", "error": err})
            print(f"[{idx}/{len(rows)}] FAILED {row['filename']}", file=sys.stderr)
            print(err, file=sys.stderr)
        append_status(status_csv, status)

    print(f"Batch status written to {status_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
