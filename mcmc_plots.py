import os
import numpy as np
import matplotlib.pyplot as plt
import corner


def save_trace_plot(sampler, param_names, outdir, run_name, discard=0):
    os.makedirs(outdir, exist_ok=True)

    chain = sampler.get_chain()
    n_params = chain.shape[-1]

    fig, axes = plt.subplots(n_params, 1, figsize=(10, 2.2 * n_params), sharex=True)

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

    fname = os.path.join(outdir, f"{run_name}_trace.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return fname

def save_corner_plot(flat_samples, param_names, outdir, run_name):
    os.makedirs(outdir, exist_ok=True)

    fig = corner.corner(
        flat_samples,
        labels=param_names,
        show_titles=True,
        quantiles=[0.16, 0.5, 0.84],
    )

    fname = os.path.join(outdir, f"{run_name}_corner.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return fname


def save_bestfit_spectrum_plot(
    testdata,
    moognn,
    medians,
    outdir,
    run_name,
    fixed_vsini=None,
    ylim=None,
):
    os.makedirs(outdir, exist_ok=True)

    if fixed_vsini is None:
        Teff, logg, rK, B, vsini = medians
    else:
        Teff, logg, rK, B = medians
        vsini = fixed_vsini

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

    fname = os.path.join(outdir, f"{run_name}_bestfit.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return fname
