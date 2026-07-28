import os
from datetime import datetime

import corner
import matplotlib.pyplot as plt
import numpy as np

from mcmc import full_params_dict_from_free, normalize_fixed_params


def make_output_tag(metadata=None, discard=None, thin=None, timestamp=None):
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


def save_trace_plot_from_chain(chain, param_names, outdir, run_name, metadata=None, discard=0, thin=None, timestamp=None):
    os.makedirs(outdir, exist_ok=True)
    n_params = chain.shape[-1]
    fig, axes = plt.subplots(n_params, 1, figsize=(10, 2.2 * n_params), sharex=True)
    if n_params == 1:
        axes = [axes]
    for i in range(n_params):
        ax = axes[i]
        ax.plot(chain[:, :, i], alpha=0.4)
        ax.set_ylabel(param_names[i])
        if discard and discard > 0:
            ax.axvline(discard, color="k", linestyle="--", alpha=0.5)
    axes[-1].set_xlabel("step")
    fig.suptitle(run_name)
    fig.tight_layout()
    tag = make_output_tag(metadata=metadata, discard=discard, thin=thin, timestamp=timestamp)
    fname = os.path.join(outdir, f"{run_name}_{tag}_trace.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return fname


def save_trace_plot(sampler, param_names, outdir, run_name, metadata=None, discard=0, thin=None, timestamp=None):
    return save_trace_plot_from_chain(
        sampler.get_chain(),
        param_names=param_names,
        outdir=outdir,
        run_name=run_name,
        metadata=metadata,
        discard=discard,
        thin=thin,
        timestamp=timestamp,
    )


def save_corner_plot(
    flat_samples,
    param_names,
    outdir,
    run_name,
    metadata=None,
    discard=None,
    thin=None,
    timestamp=None,
    quantiles=None,
):
    os.makedirs(outdir, exist_ok=True)
    if quantiles is None:
        quantiles = [0.16, 0.5, 0.84]
    fig = corner.corner(flat_samples, labels=param_names, show_titles=True, quantiles=quantiles)
    tag = make_output_tag(metadata=metadata, discard=discard, thin=thin, timestamp=timestamp)
    fname = os.path.join(outdir, f"{run_name}_{tag}_corner.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return fname


def _full_params(medians, free_param_names, fixed_params):
    return full_params_dict_from_free(medians, free_param_names, normalize_fixed_params(fixed_params))


def save_bestfit_spectrum_plot(
    testdata,
    moognn,
    medians,
    free_param_names,
    fixed_params,
    outdir,
    run_name,
    metadata=None,
    discard=None,
    thin=None,
    timestamp=None,
    ylim=None,
):
    os.makedirs(outdir, exist_ok=True)
    p = _full_params(medians, free_param_names, fixed_params)
    n_regions = len(testdata.regions)
    fig, axes = plt.subplots(n_regions, 1, figsize=(10, 2.2 * n_regions), sharex=False)
    if n_regions == 1:
        axes = [axes]

    for ax, r in zip(axes, testdata.regions):
        x, y, yerr = testdata.get_region(r)
        model = moognn.make_moogstokes_model(
            Teff=p["Teff"], logg=p["logg"], rK=p["rK"], B=p["B"], vsini=p["vsini"], region=r
        )
        if testdata.resolution is not None and testdata.kernel is not None:
            model.resolution_change(resolution=testdata.resolution, Kernel=testdata.kernel)
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
    title = (f"{run_name}\nTeff={p['Teff']:.0f}, logg={p['logg']:.2f}, rK={p['rK']:.2f}, "
             f"B={p['B']:.2f}, vsini={p['vsini']:.2f}")
    fig.suptitle(title)
    fig.tight_layout()
    tag = make_output_tag(metadata=metadata, discard=discard, thin=thin, timestamp=timestamp)
    fname = os.path.join(outdir, f"{run_name}_{tag}_bestfit.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return fname


def save_residual_spectrum_plot(
    testdata,
    moognn,
    medians,
    free_param_names,
    fixed_params,
    outdir,
    run_name,
    metadata=None,
    discard=None,
    thin=None,
    timestamp=None,
    ylim=None,
):
    os.makedirs(outdir, exist_ok=True)
    p = _full_params(medians, free_param_names, fixed_params)
    n_regions = len(testdata.regions)
    fig, axes = plt.subplots(n_regions, 1, figsize=(10, 2.0 * n_regions), sharex=False)
    if n_regions == 1:
        axes = [axes]

    for ax, r in zip(axes, testdata.regions):
        x, y, yerr = testdata.get_region(r)
        model = moognn.make_moogstokes_model(
            Teff=p["Teff"], logg=p["logg"], rK=p["rK"], B=p["B"], vsini=p["vsini"], region=r
        )
        if testdata.resolution is not None and testdata.kernel is not None:
            model.resolution_change(resolution=testdata.resolution, Kernel=testdata.kernel)
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
    title = (f"{run_name} residuals\nTeff={p['Teff']:.0f}, logg={p['logg']:.2f}, rK={p['rK']:.2f}, "
             f"B={p['B']:.2f}, vsini={p['vsini']:.2f}")
    fig.suptitle(title)
    fig.tight_layout()
    tag = make_output_tag(metadata=metadata, discard=discard, thin=thin, timestamp=timestamp)
    fname = os.path.join(outdir, f"{run_name}_{tag}_residuals.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return fname
