import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import csv
import os
import ast

from astropy.io import fits
from numpy.typing import NDArray
from typing import Union
from nn_helpers import MoogStokesNN
from spectra import SpectralDataForMoogStokes, MoogStokesModel, setup_regions_plot

PARAM_NAMES_5D = ["Teff", "logg", "rK", "B", "vsini"]
PARAM_NAMES_4D = ["Teff", "logg", "rK", "B"]

def lnlike(p: NDArray, spectrum: SpectralDataForMoogStokes, model_generator: MoogStokesNN) -> float:
    ydata_all = []
    ymodel_all = []
    yerr_all = []

    for r in spectrum.regions:
        x, y, yerr = spectrum.get_region(r)
        model = model_generator.make_moogstokes_model(Teff=p[0], logg=p[1], rK=p[2],
                                                      B=p[3], vsini=p[4], region=r)
        if spectrum.resolution is not None or spectrum.kernel is not None:
            model.resolution_change(resolution=spectrum.resolution, Kernel=spectrum.kernel)
        ymodel = model.interpolate(x)

        # Restrict to mask if one is defined for this region
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
    return -0.5 * np.nansum((ydata_all-ymodel_all)**2/yerr_all**2)

def lnprob(p: NDArray, spectrum: SpectralDataForMoogStokes, model_generator: MoogStokesNN,
           vsini: float | None = None) -> float:
    if vsini is not None:
        p = np.append(p, vsini)
    lp = lnprior(p)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(p, spectrum, model_generator)

def lnprior(p: NDArray) -> float:
    Teff, logg, rK, B, vsini = p
    if not (3000 <= Teff <= 7000):
        return -np.inf
    if not (2.5 <= logg <= 5.1):
        return -np.inf
    if not (0.0 <= rK <= 10.0):
        return -np.inf
    if not (2.0 <= vsini <= 57):
        return -np.inf
    if not (0 <= B <= 3):
        return -np.inf
    return 0

def fit_params_mcmc(spectrum: SpectralDataForMoogStokes, model_generator: MoogStokesNN,
             nwalkers: int = 64, nsteps: int = 4000, vsini: float | None = None) -> emcee.EnsembleSampler:

    # set the vsini if there is a prior fit to it.
    if vsini is not None:
        ndim = 4
        p0 = np.random.uniform(low=[3200, 2.5, 0.0, 0.0],
                            high=[7000, 5.1, 3.0, 3.0],
                            size=(nwalkers, ndim))

        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(spectrum, model_generator, vsini))
        sampler.run_mcmc(p0, nsteps,progress=True)

    else:
        ndim = 5
        p0 = np.random.uniform(low=[3200, 2.5, 0.0, 0.0, 2.0],
                            high=[7000, 5.1, 3.0, 3.0, 57],
                            size=(nwalkers, ndim))

        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(spectrum, model_generator, None))
        sampler.run_mcmc(p0, nsteps,progress=True)
    return sampler

def save_mcmc_results(data_path, name, basename, flat_samples, medians, errs):
    hdus = fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(data=flat_samples, name="CHAIN"),
        fits.ImageHDU(data=medians,      name="MEDIANS"),
        fits.ImageHDU(data=errs,         name="ERRS"),
    ])
    for hdu in hdus[1:]:
        hdu.header["NAME"]     = name
        hdu.header["BASENAME"] = basename
    hdus.writeto(os.path.join(data_path, f"{name}_mcmc.fits"), overwrite=True)


def load_mcmc_results(name, data_path="data/science"):
    with fits.open(os.path.join(data_path, f"{name}_mcmc.fits")) as hdul:
        return {
            "basename":     hdul["CHAIN"].header["BASENAME"],
            "flat_samples": hdul["CHAIN"].data,
            "medians":      hdul["MEDIANS"].data,
            "errs":         hdul["ERRS"].data,
        }

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
    hdr["YERSCL"] = metadata["error_scale_factor"]
    hdr["CHI2_0"] = metadata["chi2_initial"]
    hdr["RCHI2_0"] = metadata["reduced_chi2_initial"]
    hdr["CHI2_F"] = metadata["chi2_final"]
    hdr["RCHI2_F"] = metadata["reduced_chi2_final"]
    hdr["DOF"] = metadata["dof"]
    hdr["NPIX"] = metadata["n_used_pixels"]

    hdr["NWALKER"] = metadata.get("nwalkers", -1)
    hdr["NSTEPS"] = metadata.get("nsteps", -1)
    hdr["DISCARD"] = metadata.get("discard", -1)
    hdr["THIN"] = metadata.get("thin", -1)
    hdr["SCLDISC"] = metadata.get("scale_discard", -1)
    hdr["SCLTHIN"] = metadata.get("scale_thin", -1)

    hdr["VSINSRC"] = str(metadata.get("fixed_vsini_source", "none"))
    hdr["VSINDISC"] = metadata.get("vsini_discard", -1)
    hdr["VSINTHIN"] = metadata.get("vsini_thin", -1)

    if metadata["fixed_vsini"] is not None:
        hdr["FIXVSINI"] = metadata["fixed_vsini"]

    hdr["KERNEL"] = str(metadata["kernel"])
    hdr["NBIN"] = metadata["nyquist_bin"]

    hdus = fits.HDUList([
        primary,
        fits.ImageHDU(data=flat_samples, name="CHAIN"),
        fits.ImageHDU(data=medians, name="MEDIANS"),
        fits.ImageHDU(data=q16_q84, name="PERCENTILES"),
        fits.ImageHDU(data=np.array(metadata["regions"]), name="REGIONS"),
        fits.ImageHDU(data=np.array(metadata["shifts"]), name="SHIFTS"),
        fits.ImageHDU(data=np.array(metadata["renormalization"]), name="RENORM"),
    ])

    param_col = fits.Column(
        name="param_name",
        format="20A",
        array=np.array(metadata["param_names"], dtype="S20"),
    )
    hdus.append(fits.BinTableHDU.from_columns([param_col], name="PARAM_NAMES"))

    hdus.writeto(fits_path, overwrite=True)

    return fits_path

def retrieve_spectrum_preproc(
    basename,
    data_path="data/science",
    preparams_fname="spectrum_params.csv",
    regions_override=None,
    return_metadata=False,
):
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

    kernel = record["kernel"]
    if kernel in ("None", "", None):
        kernel = None

    resolution = None
    kernel = None

    masks = {}
    for i in range(7):
        raw = record[f"mask_{i}"]
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


def full_params_from_sampled(medians, fixed_vsini=None):
    if fixed_vsini is None:
        return np.asarray(medians)

    return np.array([
        medians[0],      # Teff
        medians[1],      # logg
        medians[2],      # rK
        medians[3],      # B
        fixed_vsini,     # vsini
    ])


def compute_reduced_chi2(spectrum, model_generator, medians, fixed_vsini=None):
    full_params = full_params_from_sampled(medians, fixed_vsini=fixed_vsini)

    chi2 = -2.0 * lnlike(full_params, spectrum, model_generator)

    n_pix = count_used_pixels(spectrum)
    n_free = len(medians)
    dof = n_pix - n_free

    if dof <= 0:
        raise ValueError(f"Non-positive degrees of freedom: n_pix={n_pix}, n_free={n_free}")

    reduced_chi2 = chi2 / dof

    return chi2, reduced_chi2, dof, n_pix

# def retrieve_spectrum_preproc(basename, data_path="data/science", preparams_fname="spectrum_params.csv"):
#
#     with open(os.path.join(data_path, preparams_fname), newline="") as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             if row["filename"] == basename:
#                 shifts          = np.array([float(row[f"shift_{i}"]) for i in range(7)])
#                 renormalization = np.array([float(row[f"renorm_{i}"]) for i in range(7)])
#                 regions = range(7)
#                 #regions = np.array([0,1,2,4,5])
#                 kernel = row["kernel"]
#                 kernel = None
#                 resolution = None
#                 masks = {}
#                 for i in range(7):
#                     raw = row[f"mask_{i}"]
#                     if raw and raw != "None":
#                         masks[i] = [tuple(m) for m in ast.literal_eval(raw)]
#                 Nbin = int(row["nyquist_bin"])
#
#
#     data = SpectralDataForMoogStokes(
#         fname = os.path.join(data_path, basename),
#         shifts=shifts,
#         renormalization=renormalization,
#         regions=regions,
#         masks=masks,
#         kernel=kernel,
#         resolution=resolution
#     )
#     data.Nyquist_bin_spectrum(N=Nbin)
#     return data
