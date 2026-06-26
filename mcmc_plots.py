import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import corner


def make_output_tag(metadata=None, discard=None, thin=None, timestamp=None):
    """
    Make a unique tag for output filenames.

    Example:
    20260625_182455_bin3_discard1000_thin5
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if metadata is None:
        metadata = {}

    nbin = metadata.get("nyquist_bin", "NA")

    pieces = [timestamp, f"bin{nbin}"]

    if discard is not None:
        pieces.append(f"discard{discard}")

    if thin is not None:
        pieces.append(f"thin{thin}")

    return "_".join(pieces)


def save_trace_plot(
    sampler,
    param_names,
    outdir,
    run_name,
    metadata=None,
    discard=0,
    thin=None,
    timestamp=None,
):
    os.makedirs(outdir, exist_ok=True)

    chain = sampler.get_chain()
    n_params = chain.shape[-1]

    fig, axes = plt.subplots(
        n_params,
        1,
        figsize=(10, 2.2 * n_params),
        sharex=True,
    )

    if n_params == 1:
        axes = [axes]

    for i in range(n_params):
        ax = axes[i]
        ax.plot(chain[:, :, i], alpha=0.4)
        ax.set_ylabel(param_names[i])

        if discard > 0:
            ax.axvline(discard, color="k", linestyle="--", alpha=0.5)

    axes[-1].set_xlabel("step")
    fig.suptitle(run_name)
    fig.tight_layout()

    tag = make_output_tag(
        metadata=metadata,
        discard=discard,
        thin=thin,
        timestamp=timestamp,
    )

    fname = os.path.join(outdir, f"{run_name}_{tag}_trace.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return fname


def save_corner_plot(
    flat_samples,
    param_names,
    outdir,
    run_name,
    metadata=None,
    discard=None,
    thin=None,
    timestamp=None,
):
    os.makedirs(outdir, exist_ok=True)

    fig = corner.corner(
        flat_samples,
        labels=param_names,
        show_titles=True,
        quantiles=[0.16, 0.5, 0.84],
    )

    tag = make_output_tag(
        metadata=metadata,
        discard=discard,
        thin=thin,
        timestamp=timestamp,
    )

    fname = os.path.join(outdir, f"{run_name}_{tag}_corner.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return fname


def unpack_mcmc_params(medians, fixed_vsini=None):
    """
    Return Teff, logg, rK, B, vsini.

    Stage 1 medians order:
        Teff, logg, rK, B, vsini

    Stage 2 medians order:
        Teff, logg, rK, B
        with fixed_vsini supplied separately.
    """
    if fixed_vsini is None:
        Teff, logg, rK, B, vsini = medians
    else:
        Teff, logg, rK, B = medians
        vsini = fixed_vsini

    return Teff, logg, rK, B, vsini


def save_bestfit_spectrum_plot(
    testdata,
    moognn,
    medians,
    outdir,
    run_name,
    fixed_vsini=None,
    metadata=None,
    discard=None,
    thin=None,
    timestamp=None,
    ylim=None,
):
    os.makedirs(outdir, exist_ok=True)

    Teff, logg, rK, B, vsini = unpack_mcmc_params(
        medians,
        fixed_vsini=fixed_vsini,
    )

    n_regions = len(testdata.regions)

    fig, axes = plt.subplots(
        n_regions,
        1,
        figsize=(10, 2.2 * n_regions),
        sharex=False,
    )

    if n_regions == 1:
        axes = [axes]

    for ax, r in zip(axes, testdata.regions):
        x, y, yerr = testdata.get_region(r)

        model = moognn.make_moogstokes_model(
            Teff=Teff,
            logg=logg,
            rK=rK,
            B=B,
            vsini=vsini,
            region=r,
        )

        if testdata.resolution is not None or testdata.kernel is not None:
            model.resolution_change(
                resolution=testdata.resolution,
                Kernel=testdata.kernel,
            )

        ymodel = model.interpolate(x)

        ax.plot(x, y, label="data")
        ax.plot(x, ymodel, label="model")
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.25)

        ax.set_title(f"Region {r}")
        ax.set_ylabel("Flux")

        if ylim is not None:
            ax.set_ylim(*ylim)

    axes[-1].set_xlabel("Wavelength")
    axes[0].legend()

    title = (
        f"{run_name}\n"
        f"Teff={Teff:.0f}, logg={logg:.2f}, rK={rK:.2f}, "
        f"B={B:.2f}, vsini={vsini:.2f}"
    )
    fig.suptitle(title)
    fig.tight_layout()

    tag = make_output_tag(
        metadata=metadata,
        discard=discard,
        thin=thin,
        timestamp=timestamp,
    )

    fname = os.path.join(outdir, f"{run_name}_{tag}_bestfit.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return fname


def save_residual_spectrum_plot(
    testdata,
    moognn,
    medians,
    outdir,
    run_name,
    fixed_vsini=None,
    metadata=None,
    discard=None,
    thin=None,
    timestamp=None,
    ylim=None,
):
    """
    Save residual plot: data - model for each fitted region.
    """
    os.makedirs(outdir, exist_ok=True)

    Teff, logg, rK, B, vsini = unpack_mcmc_params(
        medians,
        fixed_vsini=fixed_vsini,
    )

    n_regions = len(testdata.regions)

    fig, axes = plt.subplots(
        n_regions,
        1,
        figsize=(10, 2.0 * n_regions),
        sharex=False,
    )

    if n_regions == 1:
        axes = [axes]

    for ax, r in zip(axes, testdata.regions):
        x, y, yerr = testdata.get_region(r)

        model = moognn.make_moogstokes_model(
            Teff=Teff,
            logg=logg,
            rK=rK,
            B=B,
            vsini=vsini,
            region=r,
        )

        if testdata.resolution is not None or testdata.kernel is not None:
            model.resolution_change(
                resolution=testdata.resolution,
                Kernel=testdata.kernel,
            )

        ymodel = model.interpolate(x)
        residual = y - ymodel

        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.plot(x, residual, label="data - model")
        ax.fill_between(x, -yerr, yerr, alpha=0.25, label="±1σ")

        ax.set_title(f"Region {r}")
        ax.set_ylabel("Residual")

        if ylim is not None:
            ax.set_ylim(*ylim)

    axes[-1].set_xlabel("Wavelength")
    axes[0].legend()

    title = (
        f"{run_name} residuals\n"
        f"Teff={Teff:.0f}, logg={logg:.2f}, rK={rK:.2f}, "
        f"B={B:.2f}, vsini={vsini:.2f}"
    )
    fig.suptitle(title)
    fig.tight_layout()

    tag = make_output_tag(
        metadata=metadata,
        discard=discard,
        thin=thin,
        timestamp=timestamp,
    )

    fname = os.path.join(outdir, f"{run_name}_{tag}_residuals.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return fname
