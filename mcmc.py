import ast
import csv
import os
from typing import Optional

import emcee
import numpy as np
from astropy.io import fits
from numpy.typing import NDArray

from nn_helpers import MoogStokesNN
from spectra import SpectralDataForMoogStokes

PARAM_ORDER = ["Teff", "logg", "rK", "B", "vsini"]
PARAM_BOUNDS = {
    "Teff": (3200.0, 7000.0),
    "logg": (2.5, 5.1),
    "rK": (0.0, 10.0),
    "B": (0.0, 3.0),
    "vsini": (2.0, 57.0),
}


def normalize_fixed_params(fixed_params: Optional[dict]) -> dict:
    """Return a clean fixed-parameter dictionary using canonical names."""
    if fixed_params is None:
        return {}
    clean = {}
    for key, val in fixed_params.items():
        if val is None:
            continue
        if key not in PARAM_ORDER:
            raise ValueError(f"Unknown parameter to fix: {key}. Allowed: {PARAM_ORDER}")
        clean[key] = float(val)
    return clean


def free_params_from_fixed(fixed_params: Optional[dict], free_param_names: Optional[list[str]] = None) -> list[str]:
    fixed_params = normalize_fixed_params(fixed_params)
    if free_param_names is None:
        free_param_names = [p for p in PARAM_ORDER if p not in fixed_params]
    for p in free_param_names:
        if p not in PARAM_ORDER:
            raise ValueError(f"Unknown free parameter: {p}. Allowed: {PARAM_ORDER}")
        if p in fixed_params:
            raise ValueError(f"Parameter {p} cannot be both free and fixed.")
    return list(free_param_names)


def full_params_dict_from_free(p_free: NDArray, free_param_names: list[str], fixed_params: Optional[dict]) -> dict:
    fixed_params = normalize_fixed_params(fixed_params)
    if len(p_free) != len(free_param_names):
        raise ValueError("Length of p_free does not match free_param_names.")

    full = dict(fixed_params)
    for name, val in zip(free_param_names, p_free):
        full[name] = float(val)

    missing = [p for p in PARAM_ORDER if p not in full]
    if missing:
        raise ValueError(f"Missing parameter values for: {missing}")
    return full


def full_params_array_from_free(p_free: NDArray, free_param_names: list[str], fixed_params: Optional[dict]) -> NDArray:
    full = full_params_dict_from_free(p_free, free_param_names, fixed_params)
    return np.array([full[p] for p in PARAM_ORDER], dtype=float)


def lnlike(p_full: NDArray, spectrum: SpectralDataForMoogStokes, model_generator: MoogStokesNN) -> float:
    """Likelihood expects full parameter vector in order Teff, logg, rK, B, vsini."""
    ydata_all = []
    ymodel_all = []
    yerr_all = []

    Teff, logg, rK, B, vsini = p_full

    for r in spectrum.regions:
        x, y, yerr = spectrum.get_region(r)
        model = model_generator.make_moogstokes_model(
            Teff=Teff, logg=logg, rK=rK, B=B, vsini=vsini, region=r
        )

        # Only convolve if both are present. A kernel without resolution can crash.
        if spectrum.resolution is not None and spectrum.kernel is not None:
            model.resolution_change(resolution=spectrum.resolution, Kernel=spectrum.kernel)

        ymodel = model.interpolate(x)

        masks = spectrum.masks.get(r, [])
        if masks:
            keep = np.zeros(len(x), dtype=bool)
            for xlo, xhi in masks:
                keep |= (x >= xlo) & (x <= xhi)
            x_chi, y_chi, yerr_chi, ymodel_chi = x[keep], y[keep], yerr[keep], ymodel[keep]
        else:
            x_chi, y_chi, yerr_chi, ymodel_chi = x, y, yerr, ymodel

        ydata_all.append(y_chi)
        ymodel_all.append(ymodel_chi)
        yerr_all.append(yerr_chi)

    ydata_all = np.concatenate(ydata_all)
    ymodel_all = np.concatenate(ymodel_all)
    yerr_all = np.concatenate(yerr_all)

    finite = np.isfinite(ydata_all) & np.isfinite(ymodel_all) & np.isfinite(yerr_all) & (yerr_all > 0)
    if np.count_nonzero(finite) == 0:
        return -np.inf

    return -0.5 * np.nansum((ydata_all[finite] - ymodel_all[finite]) ** 2 / yerr_all[finite] ** 2)


def lnprior_full(p_full: NDArray) -> float:
    for name, val in zip(PARAM_ORDER, p_full):
        lo, hi = PARAM_BOUNDS[name]
        if not (lo <= val <= hi):
            return -np.inf
    return 0.0


def lnprob_flexible(
    p_free: NDArray,
    spectrum: SpectralDataForMoogStokes,
    model_generator: MoogStokesNN,
    free_param_names: list[str],
    fixed_params: Optional[dict] = None,
) -> float:
    p_full = full_params_array_from_free(p_free, free_param_names, fixed_params)
    lp = lnprior_full(p_full)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(p_full, spectrum, model_generator)


def fit_params_mcmc(
    spectrum: SpectralDataForMoogStokes,
    model_generator: MoogStokesNN,
    nwalkers: int = 64,
    nsteps: int = 4000,
    fixed_params: Optional[dict] = None,
    free_param_names: Optional[list[str]] = None,
    progress: bool = True,
) -> emcee.EnsembleSampler:
    """Run MCMC for arbitrary free/fixed subsets of Teff, logg, rK, B, vsini."""
    fixed_params = normalize_fixed_params(fixed_params)
    free_param_names = free_params_from_fixed(fixed_params, free_param_names)
    ndim = len(free_param_names)
    if ndim == 0:
        raise ValueError("At least one parameter must be free for MCMC.")

    lows = np.array([PARAM_BOUNDS[p][0] for p in free_param_names], dtype=float)
    highs = np.array([PARAM_BOUNDS[p][1] for p in free_param_names], dtype=float)
    p0 = np.random.uniform(low=lows, high=highs, size=(nwalkers, ndim))

    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        lnprob_flexible,
        args=(spectrum, model_generator, free_param_names, fixed_params),
    )
    sampler.run_mcmc(p0, nsteps, progress=progress)

    # Attach useful metadata to the sampler object.
    sampler.free_param_names = free_param_names
    sampler.fixed_params = fixed_params
    return sampler


def retrieve_spectrum_preproc(
    basename,
    data_path="data/science",
    preparams_fname="spectrum_params.csv",
    regions_override=None,
    return_metadata=False,
    default_resolution=0.1,
):
    """Load preprocessing settings from spectrum_params.csv.

    regions_override allows any subset, e.g. [0,1,4,5,6] or [0,1,3,4,5].
    Shifts/renormalization are stored for all 7 regions, but only selected regions are fitted/plotted.
    """
    with open(os.path.join(data_path, preparams_fname), newline="") as f:
        reader = csv.DictReader(f)
        record = None
        for row in reader:
            if row["filename"] == basename:
                record = row
                break

    if record is None:
        raise ValueError(f"No preprocessing row found for {basename}")

    shifts = np.array([float(record[f"shift_{i}"]) for i in range(7)])
    renormalization = np.array([float(record[f"renorm_{i}"]) for i in range(7)])

    if regions_override is None:
        regions = list(ast.literal_eval(record["regions"]))
    else:
        regions = list(regions_override)

    kernel = record.get("kernel", "None")
    if kernel in ("None", "", None):
        kernel = None

    # If a kernel is selected, resolution cannot be None.
    resolution = None if kernel is None else default_resolution

    masks = {}
    for i in range(7):
        raw = record.get(f"mask_{i}", "None")
        if raw and raw != "None":
            masks[i] = [tuple(m) for m in ast.literal_eval(raw)]

    Nbin = int(record["nyquist_bin"])

    data = SpectralDataForMoogStokes(
        fname=os.path.join(data_path, basename),
        name=os.path.splitext(basename)[0],
        shifts=shifts,
        renormalization=renormalization,
        regions=regions,
        masks=masks,
        kernel=kernel,
        resolution=resolution,
    )
    data.Nyquist_bin_spectrum(N=Nbin)

    metadata = {
        "basename": basename,
        "regions": regions,
        "n_regions": len(regions),
        "shifts": shifts,
        "renormalization": renormalization,
        "masks": masks,
        "nyquist_bin": Nbin,
        "kernel": kernel,
        "resolution": resolution,
        "abs_path": data.abs_path,
    }

    if return_metadata:
        return data, metadata
    return data


def count_used_pixels(spectrum):
    n = 0
    for r in spectrum.regions:
        x, y, yerr = spectrum.get_region(r)
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr)
        masks = spectrum.masks.get(r, [])
        if masks:
            keep = np.zeros(len(x), dtype=bool)
            for xlo, xhi in masks:
                keep |= (x >= xlo) & (x <= xhi)
            finite &= keep
        n += np.count_nonzero(finite)
    return n


def compute_reduced_chi2(
    spectrum,
    model_generator,
    medians_free,
    free_param_names,
    fixed_params=None,
):
    p_full = full_params_array_from_free(medians_free, free_param_names, fixed_params)
    chi2 = -2.0 * lnlike(p_full, spectrum, model_generator)
    n_pix = count_used_pixels(spectrum)
    n_free = len(medians_free)
    dof = n_pix - n_free
    if dof <= 0:
        raise ValueError(f"Non-positive degrees of freedom: n_pix={n_pix}, n_free={n_free}")
    return chi2, chi2 / dof, dof, n_pix


def summary_values_from_free(medians, q16_q84, free_param_names, fixed_params=None):
    """Return dict with value/error/status for all five physical parameters."""
    fixed_params = normalize_fixed_params(fixed_params)
    values = full_params_dict_from_free(medians, free_param_names, fixed_params)
    out = {}
    for p in PARAM_ORDER:
        out[p] = values[p]
        if p in free_param_names:
            idx = free_param_names.index(p)
            out[f"{p}_err_minus"] = values[p] - q16_q84[0, idx]
            out[f"{p}_err_plus"] = q16_q84[1, idx] - values[p]
            out[f"{p}_status"] = "free"
        else:
            out[f"{p}_err_minus"] = np.nan
            out[f"{p}_err_plus"] = np.nan
            out[f"{p}_status"] = "fixed"
    return out


def save_mcmc_results_extended(
    data_path,
    run_name,
    basename,
    flat_samples,
    medians,
    q16_q84,
    metadata,
):
    outdir = os.path.join(data_path, "mcmc_outputs", run_name)
    os.makedirs(outdir, exist_ok=True)
    fits_path = os.path.join(outdir, f"{run_name}_mcmc.fits")

    primary = fits.PrimaryHDU()
    hdr = primary.header
    hdr["RUNNAME"] = run_name
    hdr["BASENAME"] = basename
    hdr["NPARAM"] = len(medians)
    hdr["NREG"] = metadata["n_regions"]
    hdr["YERSCL"] = metadata.get("error_scale_factor", np.nan)
    hdr["CHI2_0"] = metadata.get("chi2_initial", np.nan)
    hdr["RCHI2_0"] = metadata.get("reduced_chi2_initial", np.nan)
    hdr["CHI2_F"] = metadata.get("chi2_final", np.nan)
    hdr["RCHI2_F"] = metadata.get("reduced_chi2_final", np.nan)
    hdr["DOF"] = metadata.get("dof", -1)
    hdr["NPIX"] = metadata.get("n_used_pixels", -1)
    hdr["NWALKER"] = metadata.get("nwalkers", -1)
    hdr["NSTEPS"] = metadata.get("nsteps", -1)
    hdr["DISCARD"] = metadata.get("discard", -1)
    hdr["THIN"] = metadata.get("thin", -1)
    hdr["SCLDISC"] = metadata.get("scale_discard", -1)
    hdr["SCLTHIN"] = metadata.get("scale_thin", -1)
    hdr["KERNEL"] = str(metadata.get("kernel", "None"))
    hdr["NBIN"] = metadata.get("nyquist_bin", -1)

    fixed_params = normalize_fixed_params(metadata.get("fixed_params", {}))
    for key, value in fixed_params.items():
        fits_key = f"FIX{key.upper()}"[:8]
        hdr[fits_key] = value

    hdus = fits.HDUList([
        primary,
        fits.ImageHDU(data=flat_samples, name="CHAIN"),
        fits.ImageHDU(data=medians, name="MEDIANS"),
        fits.ImageHDU(data=q16_q84, name="PERCENTILES"),
        fits.ImageHDU(data=np.array(metadata["regions"]), name="REGIONS"),
        fits.ImageHDU(data=np.array(metadata["shifts"]), name="SHIFTS"),
        fits.ImageHDU(data=np.array(metadata["renormalization"]), name="RENORM"),
    ])

    free_param_names = metadata.get("free_param_names", [])
    fixed_param_names = list(fixed_params.keys())

    hdus.append(fits.BinTableHDU.from_columns([
        fits.Column(name="free_param_name", format="20A", array=np.array(free_param_names, dtype="S20")),
    ], name="FREE_PARAMS"))

    if fixed_param_names:
        hdus.append(fits.BinTableHDU.from_columns([
            fits.Column(name="fixed_param_name", format="20A", array=np.array(fixed_param_names, dtype="S20")),
            fits.Column(name="fixed_value", format="D", array=np.array([fixed_params[k] for k in fixed_param_names], dtype=float)),
        ], name="FIXED_PARAMS"))

    hdus.writeto(fits_path, overwrite=True)
    return fits_path
