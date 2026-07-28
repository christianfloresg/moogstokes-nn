#!/usr/bin/env python3
"""Plot weak-B-field spectra in manually chosen diagnostic windows.

The script reuses the saved stage-2 MCMC metadata to load the same preprocessed
science spectra, then keeps Teff, logg, rK, and vsini fixed at the summary
values while plotting the fitted-B model and a small comparison grid in B.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


PROJECT_ROOT = Path("/Users/christianflores/Documents/GitHub/moogstokes-nn")
SUMMARY_CSV = PROJECT_ROOT / "mcmc_runs" / "values_longrun_stage2_3sigma_summary.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures" / "weak_bfield_manual_windows"

WEAK_FIELD_FILENAMES = [
    "Spectrum_EPIC211002011.nspec",
    "Spectrum_HIP16563NW.nspec",
    "Spectrum_HIP82688.cleaned.nspec",
    "Spectrum_EPIC211068400.nspec",
    "Spectrum_EPIC211095259.nspec",
    "Spectrum_HIP12635.nspec",
    "Spectrum_HIP18859.nspec",
]

DISPLAY_NAMES = {
    "Spectrum_EPIC211068400.nspec": "EPIC 211068400",
    "Spectrum_EPIC211002011.nspec": "EPIC 211002011",
    "Spectrum_EPIC211095259.nspec": "EPIC 211095259",
    "Spectrum_HIP113579.cleaned.nspec": "HIP 113579",
    "Spectrum_HIP12635.nspec": "HIP 12635",
    "Spectrum_HIP16563NW.nspec": "HIP 16563 NW",
    "Spectrum_HIP18859.nspec": "HIP 18859",
    "Spectrum_HIP82688.cleaned.nspec": "HIP 82688",
}

MANUAL_WINDOWS_UM = {
    "Spectrum_HIP12635.nspec": [(0, 2.1170), (2, 2.1903)],
    "Spectrum_HIP18859.nspec": [(3, 2.2066), (4, 2.2265)],
    "Spectrum_HIP16563NW.nspec": [(0, 2.1170), (1, 2.1787)],
    "Spectrum_EPIC211002011.nspec": [(1, 2.1787), (5, 2.2630)],
    "Spectrum_HIP82688.cleaned.nspec": [(2, 2.1904), (4, 2.2264)],
    "Spectrum_EPIC211068400.nspec": [(0, 2.1098), (2, 2.1903)],
    "Spectrum_EPIC211095259.nspec": [(1, 2.1787), (2, 2.1902)],
}

COMPARISON_B_GRID = [0.5, 1.0, 1.5]
DEFAULT_WINDOW_FULL_WIDTH_A = 14.0
OBSERVED_ZORDER = 2
BEST_FIT_ZORDER = 3
B_FIELD_ZORDER = 5
RESIDUAL_HEIGHT_RATIO = 0.42
RESIDUAL_MIN_ABS_FLUX = 0.05
RESIDUAL_LIMIT_PERCENTILE = 98.0
B_STYLES = {
    "best": {"color": "#006D77", "lw": 2.5, "ls": "-", "alpha": 0.48},
    0.5: {"color": "#E9C46A", "lw": 1.2, "ls": "--", "alpha": 0.98},
    1.0: {"color": "#E76F51", "lw": 1.25, "ls": ":", "alpha": 0.98},
    1.5: {"color": "#9D174D", "lw": 1.25, "ls": "-.", "alpha": 0.98},
}


@dataclass
class SourceModels:
    filename: str
    display_name: str
    params: dict[str, float]
    metadata: dict
    regions: list[int]
    testdata: object
    models: dict[int, dict[float, np.ndarray]]


def safe_name(name: str) -> str:
    keep = []
    for char in name:
        if char.isalnum():
            keep.append(char)
        elif char in {" ", "-", "_"}:
            keep.append("_")
    return "".join(keep).strip("_")


def setup_project_imports(project_root: Path) -> None:
    sys.path.insert(0, str(project_root))
    os.chdir(project_root)


def load_project_helpers(project_root: Path):
    setup_project_imports(project_root)
    from mcmc import load_mcmc_run, retrieve_spectrum_preproc
    from nn_helpers import MoogStokesNN

    return load_mcmc_run, retrieve_spectrum_preproc, MoogStokesNN


def load_summary(summary_csv: Path) -> pd.DataFrame:
    summary = pd.read_csv(summary_csv)
    missing = [name for name in WEAK_FIELD_FILENAMES if name not in set(summary["filename"])]
    if missing:
        raise ValueError("Missing weak-field rows from summary CSV: " + ", ".join(missing))
    weak = (
        summary[summary["filename"].isin(WEAK_FIELD_FILENAMES)]
        .copy()
        .set_index("filename")
        .loc[WEAK_FIELD_FILENAMES]
        .reset_index()
    )
    return weak


def source_params_from_row(row: pd.Series) -> dict[str, float]:
    return {
        "Teff": float(row["Teff"]),
        "logg": float(row["logg"]),
        "rK": float(row["rK"]),
        "B": float(row["B"]),
        "vsini": float(row["vsini"]),
    }


def unique_float_values(values: list[float], atol: float = 1.0e-8) -> list[float]:
    unique: list[float] = []
    for value in values:
        value = float(value)
        if not any(math.isclose(value, prev, rel_tol=0.0, abs_tol=atol) for prev in unique):
            unique.append(value)
    return unique


def model_b_values_for_source(params: dict[str, float]) -> list[float]:
    # Include B=0 so the older automatic-window helper still works if reused.
    return unique_float_values([params["B"], *COMPARISON_B_GRID, 0.0])


def plot_b_values_for_source(source: SourceModels) -> list[float]:
    return unique_float_values([source.params["B"], *COMPARISON_B_GRID])


def requested_regions_for_filename(filename: str) -> list[int]:
    return [int(region) for region, _ in MANUAL_WINDOWS_UM.get(filename, [])]


def load_source_models(row, moognn, load_mcmc_run, retrieve_spectrum_preproc) -> SourceModels:
    run = load_mcmc_run(row["mcmc_run_file"])
    metadata = dict(run["metadata"])
    fit_regions = [int(r) for r in metadata.get("regions", [])]
    regions = sorted(set(fit_regions) | set(requested_regions_for_filename(row["filename"])))
    data_path = metadata.get("data_path", "data/science/ClassIII-sources-spectra")

    with contextlib.redirect_stdout(open(os.devnull, "w")):
        testdata, preproc_meta = retrieve_spectrum_preproc(
            row["filename"],
            data_path=data_path,
            regions_override=regions,
            return_metadata=True,
        )
        errscale = metadata.get("error_scale_factor")
        if errscale is not None and np.isfinite(float(errscale)):
            testdata.rescale_yerr(float(errscale))

    metadata = {**preproc_meta, **metadata}
    params = source_params_from_row(row)
    models: dict[int, dict[float, np.ndarray]] = {}

    for region in regions:
        x, _, _ = testdata.get_region(region)
        models[region] = {}
        for bfield in model_b_values_for_source(params):
            model = moognn.make_moogstokes_model(
                Teff=params["Teff"],
                logg=params["logg"],
                rK=params["rK"],
                B=bfield,
                vsini=params["vsini"],
                region=region,
            )
            if testdata.resolution is not None and testdata.kernel is not None:
                model.resolution_change(resolution=testdata.resolution, Kernel=testdata.kernel)
            models[region][bfield] = model.interpolate(x)

    return SourceModels(
        filename=row["filename"],
        display_name=DISPLAY_NAMES.get(row["filename"], row["filename"]),
        params=params,
        metadata=metadata,
        regions=regions,
        testdata=testdata,
        models=models,
    )


def smooth_boxcar(values: np.ndarray, width: int = 31) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    width = max(3, int(width))
    if width % 2 == 0:
        width += 1
    width = min(width, values.size if values.size % 2 == 1 else values.size - 1)
    if width < 3:
        return values
    kernel = np.ones(width, dtype=float)
    finite = np.isfinite(values)
    numerator = np.convolve(np.where(finite, values, 0.0), kernel, mode="same")
    denominator = np.convolve(finite.astype(float), kernel, mode="same")
    out = np.divide(numerator, denominator, out=np.full_like(values, np.nan), where=denominator > 0)
    return out


def select_diagnostic_windows(
    source: SourceModels,
    n_windows: int = 3,
    half_width: float = 2.2,
    smooth_points: int = 41,
) -> list[dict[str, float | int]]:
    candidates: list[dict[str, float | int]] = []

    for region in source.regions:
        x, _, yerr = source.testdata.get_region(region)
        y0 = source.models[region][0.0]
        y1 = source.models[region][1.0]
        finite = np.isfinite(x) & np.isfinite(y0) & np.isfinite(y1) & np.isfinite(yerr) & (yerr > 0)
        if np.count_nonzero(finite) < 10:
            continue

        noise_floor = np.nanmedian(yerr[finite])
        if not np.isfinite(noise_floor) or noise_floor <= 0:
            noise_floor = 1.0
        sensitivity = np.abs(y1 - y0) / noise_floor
        sensitivity = smooth_boxcar(np.where(finite, sensitivity, np.nan), width=smooth_points)

        order = np.argsort(np.nan_to_num(sensitivity, nan=-np.inf))[::-1]
        per_region = 0
        for idx in order:
            center = float(x[idx])
            if not np.isfinite(center):
                continue
            if center - half_width < float(np.nanmin(x)) or center + half_width > float(np.nanmax(x)):
                continue
            if any(
                int(c["region"]) == region and abs(float(c["center_A"]) - center) < 2.0 * half_width
                for c in candidates
            ):
                continue
            window = (x >= center - half_width) & (x <= center + half_width) & finite
            if np.count_nonzero(window) < 15:
                continue
            candidates.append(
                {
                    "region": region,
                    "center_A": center,
                    "xlo_A": center - half_width,
                    "xhi_A": center + half_width,
                    "score": float(np.nanmax(sensitivity[window])),
                }
            )
            per_region += 1
            if per_region >= 2:
                break

    candidates = sorted(candidates, key=lambda c: float(c["score"]), reverse=True)
    selected: list[dict[str, float | int]] = []
    for cand in candidates:
        region = int(cand["region"])
        center = float(cand["center_A"])
        if any(
            int(sel["region"]) == region and abs(float(sel["center_A"]) - center) < 2.0 * half_width
            for sel in selected
        ):
            continue
        selected.append(cand)
        if len(selected) >= n_windows:
            break

    return sorted(selected, key=lambda c: (int(c["region"]), float(c["center_A"])))


def manual_windows_for_source(source: SourceModels, half_width: float) -> list[dict[str, float | int]]:
    specs = MANUAL_WINDOWS_UM.get(source.filename)
    if not specs:
        raise ValueError(f"No manual diagnostic windows were specified for {source.display_name}")

    windows: list[dict[str, float | int]] = []
    for region, center_um in specs:
        region = int(region)
        if region not in source.models:
            raise ValueError(
                f"Requested region R{region} is unavailable for {source.display_name}; "
                f"loaded regions are {source.regions}"
            )
        center = float(center_um) * 1.0e4
        x, _, _ = source.testdata.get_region(region)
        xlo = center - half_width
        xhi = center + half_width
        if np.count_nonzero((x >= xlo) & (x <= xhi) & np.isfinite(x)) < 10:
            raise ValueError(
                f"Manual window R{region} centered at {center:.1f} A for {source.display_name} "
                f"has too few pixels in the loaded data span "
                f"{float(np.nanmin(x)):.1f}-{float(np.nanmax(x)):.1f} A"
            )
        windows.append(
            {
                "region": region,
                "center_A": center,
                "center_um": float(center_um),
                "xlo_A": xlo,
                "xhi_A": xhi,
                "score": np.nan,
            }
        )
    return windows


def y_limits_for_window(source: SourceModels, region: int, xlo: float, xhi: float) -> tuple[float, float]:
    x, y, yerr = source.testdata.get_region(region)
    window = (x >= xlo) & (x <= xhi)
    pieces = [y[window]]
    for bfield in plot_b_values_for_source(source):
        pieces.append(source.models[region][bfield][window])
    values = np.concatenate([p[np.isfinite(p)] for p in pieces if p.size])
    if values.size == 0:
        return 0.75, 1.05
    lo, hi = np.nanpercentile(values, [1, 99])
    pad = max(0.015, 0.18 * (hi - lo))
    return max(0.0, lo - pad), min(1.18, hi + pad)


def residual_limits_for_window(source: SourceModels, region: int, xlo: float, xhi: float) -> tuple[float, float]:
    x, y, yerr = source.testdata.get_region(region)
    window = (x >= xlo) & (x <= xhi)
    pieces = []
    for bfield in plot_b_values_for_source(source):
        model = source.models[region][bfield]
        residual = y[window] - model[window]
        pieces.append(np.abs(residual[np.isfinite(residual)]))
    pieces.append(np.abs(yerr[window][np.isfinite(yerr[window])]))
    values = np.concatenate([p for p in pieces if p.size])
    if values.size == 0:
        return -0.05, 0.05
    limit = np.nanpercentile(values, RESIDUAL_LIMIT_PERCENTILE)
    limit = max(RESIDUAL_MIN_ABS_FLUX, 1.25 * float(limit))
    return -limit, limit


def draw_window_panel(
    ax,
    source: SourceModels,
    window: dict[str, float | int],
    show_ylabel: bool = True,
    show_xlabel: bool = True,
    show_residual_ylabel: bool = True,
    annotate_source: bool = False,
    residual_ax=None,
) -> None:
    region = int(window["region"])
    xlo = float(window["xlo_A"])
    xhi = float(window["xhi_A"])
    center = float(window["center_A"])
    x, y, yerr = source.testdata.get_region(region)
    mask = (x >= xlo) & (x <= xhi)
    xplot = x[mask] - center

    ax.fill_between(
        xplot,
        y[mask] - yerr[mask],
        y[mask] + yerr[mask],
        color="#b8b8b8",
        alpha=0.30,
        linewidth=0,
        zorder=1,
    )
    ax.plot(xplot, y[mask], color="black", lw=1.0, label="Observed", zorder=OBSERVED_ZORDER)

    best_b = float(source.params["B"])
    best_style = B_STYLES["best"]
    ax.plot(
        xplot,
        source.models[region][best_b][mask],
        color=best_style["color"],
        lw=best_style["lw"],
        ls=best_style["ls"],
        alpha=best_style["alpha"],
        label=f"Best fit (B={best_b:.2f} kG)",
        zorder=BEST_FIT_ZORDER,
    )
    for bfield in COMPARISON_B_GRID:
        style = B_STYLES[bfield]
        ax.plot(
            xplot,
            source.models[region][bfield][mask],
            color=style["color"],
            lw=style["lw"],
            ls=style["ls"],
            alpha=style["alpha"],
            label=f"B={bfield:g} kG",
            zorder=B_FIELD_ZORDER,
        )

    ax.set_xlim(xlo - center, xhi - center)
    ax.set_ylim(*y_limits_for_window(source, region, xlo, xhi))
    ax.grid(True, color="#dddddd", lw=0.5, alpha=0.7)
    ax.tick_params(axis="both", labelsize=14)
    if show_ylabel:
        ax.set_ylabel("Norm. Flux", fontsize=15)
    if show_xlabel:# and residual_ax is None:
        ax.set_xlabel(r"$ \Delta \lambda$ - {center:.1f}" +r"\r, \AA", fontsize=14)
    if residual_ax is not None:
        ax.set_xlabel(r"$ \Delta \lambda$ - {center:.1f}" +r"\r, \AA", fontsize=14)
        # ax.tick_params(labelbottom=False)
    if annotate_source:
        ax.text(
            0.02,
            0.98,
            source.display_name,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=14,
            fontweight="bold",
            # bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )
    # ax.text(
    #     0.98,
    #     0.08,
    #     f"Reg. {region}",
    #     transform=ax.transAxes,
    #     ha="right",
    #     va="bottom",
    #     fontsize=8,
    #     color="#333333",
    # )

    if residual_ax is None:
        return

    residual_ax.fill_between(
        xplot,
        -yerr[mask],
        yerr[mask],
        color="#b8b8b8",
        alpha=0.20,
        linewidth=0,
        zorder=1,
    )
    residual_ax.axhline(0.0, color="#555555", lw=0.7, alpha=0.75, zorder=2)
    residual_ax.plot(
        xplot,
        y[mask] - source.models[region][best_b][mask],
        color=best_style["color"],
        lw=max(1.0, best_style["lw"] - 0.2),
        ls=best_style["ls"],
        alpha=best_style["alpha"],
        zorder=BEST_FIT_ZORDER,
    )
    for bfield in COMPARISON_B_GRID:
        style = B_STYLES[bfield]
        residual_ax.plot(
            xplot,
            y[mask] - source.models[region][bfield][mask],
            color=style["color"],
            lw=max(1.1, style["lw"] - 0.25),
            ls=style["ls"],
            alpha=style["alpha"],
            zorder=B_FIELD_ZORDER,
        )

    residual_ax.set_xlim(xlo - center, xhi - center)
    residual_ax.set_ylim(*residual_limits_for_window(source, region, xlo, xhi))
    residual_ax.grid(True, color="#dddddd", lw=0.45, alpha=0.65)
    residual_ax.tick_params(axis="both", labelsize=12)
    if show_residual_ylabel:
        residual_ax.set_ylabel("Residual", fontsize=10)
    if show_xlabel:
        residual_ax.set_xlabel(f"{center:.1f} - " +r"$ \Delta \lambda$ " +r"$\rm (\AA)$", fontsize=16, labelpad=-2)
    else:
        residual_ax.tick_params(labelbottom=False)


def draw_all_region_panel(ax, source: SourceModels, region: int) -> None:
    x, y, yerr = source.testdata.get_region(region)
    ax.fill_between(x, y - yerr, y + yerr, color="#b8b8b8", alpha=0.35, linewidth=0)
    ax.plot(x, y, color="black", lw=0.8, label="Observed")
    best_b = float(source.params["B"])
    best_style = B_STYLES["best"]
    ax.plot(x, source.models[region][best_b], color=best_style["color"], lw=1.5, ls=best_style["ls"])
    for bfield in COMPARISON_B_GRID:
        style = B_STYLES[bfield]
        ax.plot(x, source.models[region][bfield], color=style["color"], lw=1.2, ls=style["ls"])

    region_length = float(np.nanmax(x)) - float(np.nanmin(x))
    ax.set_xlim(float(np.nanmin(x)) + region_length*0.2, float(np.nanmax(x)) - region_length*0.2)
    values = np.concatenate(

        [y[np.isfinite(y)]]
        + [
            source.models[region][b][np.isfinite(source.models[region][b])]
            for b in plot_b_values_for_source(source)
        ]
    )
    lo, hi = np.nanpercentile(values, [1, 99])
    pad = max(0.02, 0.15 * (hi - lo))
    ax.set_ylim(max(0.0, lo - pad)-0.02, min(1.2, hi + pad)+0.02)
    # ax.set_ylabel(f"R{region}", fontsize=8)
    # ax.grid(True, color="#dddddd", lw=0.5, alpha=0.7)
    ax.tick_params(axis="both", labelsize=12)


def legend_handles(best_label: str = "Best fit model") -> list[Line2D]:
    handles = [
        Line2D([0], [0], color="black", lw=1.3, label="Observed"),
        Line2D(
            [0],
            [0],
            color=B_STYLES["best"]["color"],
            lw=B_STYLES["best"]["lw"],
            ls=B_STYLES["best"]["ls"],
            alpha=B_STYLES["best"]["alpha"],
            label=best_label,
        ),
    ]
    handles.extend(
        Line2D(
            [0],
            [0],
            color=B_STYLES[b]["color"],
            lw=B_STYLES[b]["lw"],
            ls=B_STYLES[b]["ls"],
            alpha=B_STYLES[b]["alpha"],
            label=f"B={b:g} kG",
        )
        for b in COMPARISON_B_GRID
    )
    return handles


def save_per_source_zoom(source: SourceModels, windows, outdir: Path) -> Path:
    n = len(windows)
    fig = plt.figure(figsize=(4.4 * n, 4.05))
    outer = fig.add_gridspec(1, n, left=0.08, right=0.985, bottom=0.12, top=0.78, wspace=0.14)
    for idx, window in enumerate(windows):
        inner = outer[0, idx].subgridspec(
            2,
            1,
            height_ratios=[1.0, RESIDUAL_HEIGHT_RATIO],
            hspace=0.04,
        )
        flux_ax = fig.add_subplot(inner[0])
        residual_ax = fig.add_subplot(inner[1], sharex=flux_ax)
        draw_window_panel(
            flux_ax,
            source,
            window,
            show_ylabel=(idx == 0),
            show_xlabel=True,
            show_residual_ylabel=(idx == 0),
            annotate_source=(idx == 0),
            residual_ax=residual_ax,
        )
        # flux_ax.set_title(
            # f"Region {int(window['region'])}: {float(window['xlo_A']):.1f}-{float(window['xhi_A']):.1f} A",
            # fontsize=16,
        # )
    p = source.params
    handles = legend_handles(best_label=f"Best fit (B={p['B']:.2f} kG)")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=3, frameon=False, fontsize=14.5)
    png = outdir / f"weak_bfield_{safe_name(source.display_name)}_manual_windows2.png"
    pdf = png.with_suffix(".pdf")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png


def save_per_source_all_regions(source: SourceModels, outdir: Path) -> Path:
    n_regions = len(source.regions)
    fig, axes = plt.subplots(n_regions, 1, figsize=(8.0, 1.65 * n_regions), sharex=False)
    if n_regions == 1:
        axes = [axes]
    for ax, region in zip(axes, source.regions):
        draw_all_region_panel(ax, source, region)
    axes[-1].set_xlabel("Wavelength"+r"$\rm (\AA)$", fontsize=12)
    plt.ylabel("Normalized Flux", fontsize=12)

    fig.legend(
        handles=legend_handles(best_label=f"Best fit (B={source.params['B']:.2f} kG)"),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
        frameon=False,
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    png = outdir / f"weak_bfield_{safe_name(source.display_name)}_all_regions.png"
    pdf = png.with_suffix(".pdf")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png


def save_compact_mosaic(sources: list[SourceModels], windows_by_source, outdir: Path) -> Path:
    n_rows = len(sources)
    n_cols = max(len(windows_by_source[src.filename]) for src in sources)
    fig = plt.figure(figsize=(4.2 * n_cols, 2.65 * n_rows))
    outer = fig.add_gridspec(
        n_rows,
        n_cols,
        left=0.08,
        right=0.985,
        bottom=0.055,
        top=0.93,
        hspace=0.25,
        wspace=0.20,
    )
    for row_idx, source in enumerate(sources):
        windows = windows_by_source[source.filename]
        for col_idx in range(n_cols):
            if col_idx >= len(windows):
                ax = fig.add_subplot(outer[row_idx, col_idx])
                # ax.axis("off")
                continue
            inner = outer[row_idx, col_idx].subgridspec(
                2,
                1,
                height_ratios=[1.0, RESIDUAL_HEIGHT_RATIO],
                hspace=0.04,
            )
            flux_ax = fig.add_subplot(inner[0])
            residual_ax = fig.add_subplot(inner[1], sharex=flux_ax)
            draw_window_panel(
                flux_ax,
                source,
                windows[col_idx],
                show_ylabel=(col_idx == 0),
                # show_xlabel=(row_idx == n_rows - 1),
                show_residual_ylabel=(col_idx == 0),
                annotate_source=(col_idx == 0),
                residual_ax=residual_ax,
                show_xlabel=True,
            )
            # if row_idx == 0:
                # flux_ax.set_ti tle(f"Diagnostic window {col_idx + 1}", fontsize=10)
    fig.legend(
        handles=legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        frameon=False,
        fontsize=16,
    )
    png = outdir / "weak_bfield_manual_window_mosaic2.png"
    pdf = png.with_suffix(".pdf")
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png


def chi2_for_source_b(source: SourceModels, bfield: float) -> tuple[float, int]:
    chi2 = 0.0
    n_pix = 0
    for region in source.regions:
        x, y, yerr = source.testdata.get_region(region)
        ymodel = source.models[region][bfield]
        finite = np.isfinite(y) & np.isfinite(yerr) & np.isfinite(ymodel) & (yerr > 0)
        masks = source.testdata.masks.get(region, [])
        if masks:
            keep = np.zeros(len(x), dtype=bool)
            for xlo, xhi in masks:
                keep |= (x >= xlo) & (x <= xhi)
            finite &= keep
        if np.count_nonzero(finite) == 0:
            continue
        chi2 += float(np.nansum((y[finite] - ymodel[finite]) ** 2 / yerr[finite] ** 2))
        n_pix += int(np.count_nonzero(finite))
    return chi2, n_pix


def save_chi2_panel(sources: list[SourceModels], outdir: Path) -> Path:
    fig, axes = plt.subplots(4, 2, figsize=(8.8, 10.5), sharex=True)
    axes = axes.reshape(-1)
    rows = []
    for idx, source in enumerate(sources):
        values = []
        for bfield in COMPARISON_B_GRID:
            chi2, n_pix = chi2_for_source_b(source, bfield)
            values.append(chi2)
            rows.append(
                {
                    "source": source.display_name,
                    "filename": source.filename,
                    "B_kG": bfield,
                    "chi2": chi2,
                    "n_pixels": n_pix,
                    "reduced_chi2_fixed_grid": chi2 / max(n_pix, 1),
                }
            )
        values = np.array(values)
        delta = values - np.nanmin(values)
        ax = axes[idx]
        ax.plot(COMPARISON_B_GRID, delta, marker="o", color="#264653", lw=1.6)
        ax.axvline(source.params["B"], color="#E76F51", lw=1.1, ls="--")
        ax.set_title(source.display_name, fontsize=10)
        ax.set_ylabel("Delta chi2", fontsize=8)
        ax.grid(True, color="#dddddd", lw=0.5, alpha=0.7)
        ax.tick_params(labelsize=8)
    for ax in axes[len(sources) :]:
        ax.axis("off")
    for ax in axes[-2:]:
        ax.set_xlabel("Fixed B (kG)", fontsize=9)
    fig.tight_layout()
    png = outdir / "weak_bfield_delta_chi2_vs_B.png"
    pdf = png.with_suffix(".pdf")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(outdir / "weak_bfield_delta_chi2_vs_B.csv", index=False, float_format="%.8g")
    return png


def write_window_table(sources: list[SourceModels], windows_by_source, outdir: Path) -> Path:
    rows = []
    for source in sources:
        p = source.params
        for idx, window in enumerate(windows_by_source[source.filename], start=1):
            rows.append(
                {
                    "source": source.display_name,
                    "filename": source.filename,
                    "panel": idx,
                    "region": int(window["region"]),
                    "xlo_A": float(window["xlo_A"]),
                    "xhi_A": float(window["xhi_A"]),
                    "center_A": float(window["center_A"]),
                    "center_um": float(window["center_um"]),
                    "sensitivity_score": float(window["score"]),
                    "Teff": p["Teff"],
                    "logg": p["logg"],
                    "rK": p["rK"],
                    "fitted_B": p["B"],
                    "vsini": p["vsini"],
                }
            )
    path = outdir / "weak_bfield_manual_windows.csv"
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.8g")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--summary-csv", type=Path, default=SUMMARY_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--window-full-width",
        type=float,
        default=DEFAULT_WINDOW_FULL_WIDTH_A,
        help="Full width of each manually selected diagnostic window in Angstroms.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    load_mcmc_run, retrieve_spectrum_preproc, MoogStokesNN = load_project_helpers(args.project_root)

    weak = load_summary(args.summary_csv)
    moognn = MoogStokesNN(nn_models_dir="data/", regions=range(7))

    sources = [
        load_source_models(row, moognn, load_mcmc_run, retrieve_spectrum_preproc)
        for _, row in weak.iterrows()
    ]
    half_width = 0.5 * float(args.window_full_width)
    windows_by_source = {source.filename: manual_windows_for_source(source, half_width) for source in sources}

    saved = []
    for source in sources:
        # saved.append(save_per_source_zoom(source, windows_by_source[source.filename], args.output_dir))
        saved.append(save_per_source_all_regions(source, args.output_dir))
    # saved.append(save_compact_mosaic(sources, windows_by_source, args.output_dir))
    window_table = write_window_table(sources, windows_by_source, args.output_dir)

    print(f"Wrote {len(saved)} figure PNGs plus matching PDFs to {args.output_dir}")
    print(f"Wrote {window_table}")


if __name__ == "__main__":
    main()
