import os
import csv
from datetime import datetime

import numpy as np

from mcmc import (
    PARAM_NAMES_5D,
    PARAM_NAMES_4D,
    retrieve_spectrum_preproc,
    fit_params_mcmc,
    compute_reduced_chi2,
    save_mcmc_results_extended,
)

from mcmc_plots import (
    save_trace_plot,
    save_corner_plot,
    save_bestfit_spectrum_plot,
    save_residual_spectrum_plot,
)

def run_error_rescaled_mcmc(
    basename,
    run_name,
    moognn,
    data_path="data/science",
    regions=(0, 1, 2, 3, 4, 5, 6),
    fixed_vsini=None,
    nwalkers=16,
    nsteps=1000,
    scale_discard=500,
    scale_thin=5,
):
    """
    Run the two-pass error-rescaled MCMC.

    This function performs the expensive part:
        1. first MCMC
        2. estimate reduced chi2
        3. rescale yerr
        4. second MCMC

    It does NOT decide the final discard used for plots/summary.
    That is done later in postprocess_mcmc_run().
    """

    # Load fresh data for this specific stage
    testdata, meta = retrieve_spectrum_preproc(
        basename,
        data_path=data_path,
        regions_override=regions,
        return_metadata=True,
    )

    # First pass
    sampler1 = fit_params_mcmc(
        testdata,
        moognn,
        nwalkers=nwalkers,
        nsteps=nsteps,
        vsini=fixed_vsini,
    )

    flat1_for_scale = sampler1.get_chain(
        discard=scale_discard,
        thin=scale_thin,
        flat=True,
    )

    med1_for_scale = np.median(flat1_for_scale, axis=0)

    chi2_1, redchi2_1, dof_1, n_pix_1 = compute_reduced_chi2(
        testdata,
        moognn,
        med1_for_scale,
        fixed_vsini=fixed_vsini,
    )

    errscale = np.sqrt(redchi2_1)

    # Rescale errors and regenerate fitting regions
    testdata.rescale_yerr(errscale)

    # Second pass: this is the final sampler you will postprocess later
    sampler2 = fit_params_mcmc(
        testdata,
        moognn,
        nwalkers=nwalkers,
        nsteps=nsteps,
        vsini=fixed_vsini,
    )

    meta.update({
        "run_name": run_name,
        "fixed_vsini": fixed_vsini,
        "param_names": PARAM_NAMES_4D if fixed_vsini is not None else PARAM_NAMES_5D,

        "chi2_initial": chi2_1,
        "reduced_chi2_initial": redchi2_1,
        "error_scale_factor": errscale,

        "dof_initial": dof_1,
        "n_used_pixels_initial": n_pix_1,

        "nwalkers": nwalkers,
        "nsteps": nsteps,
        "scale_discard": scale_discard,
        "scale_thin": scale_thin,
    })

    return {
        "testdata": testdata,
        "sampler": sampler2,
        "metadata": meta,
    }

def postprocess_mcmc_run(
    run_result,
    basename,
    source_name,
    run_name,
    stage,
    moognn,
    data_path="data/science",
    discard=1000,
    thin=5,
    summary_csv=None,
    timestamp=None,
):
    """
    Postprocess an already-run sampler.

    This is the cheap part. You can call it many times with different
    discard/thin values without rerunning the MCMC.
    """

    if summary_csv is None:
        summary_csv = os.path.join(
            data_path,
            "mcmc_outputs",
            "mcmc_summary_extended.csv",
        )

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    testdata = run_result["testdata"]
    sampler = run_result["sampler"]
    metadata = run_result["metadata"].copy()

    fixed_vsini = metadata.get("fixed_vsini", None)

    flat_samples = sampler.get_chain(
        discard=discard,
        thin=thin,
        flat=True,
    )

    medians = np.median(flat_samples, axis=0)
    q16_q84 = np.percentile(flat_samples, [16, 84], axis=0)

    chi2_final, redchi2_final, dof_final, n_pix_final = compute_reduced_chi2(
        testdata,
        moognn,
        medians,
        fixed_vsini=fixed_vsini,
    )

    metadata.update({
        "discard": discard,
        "thin": thin,
        "chi2_final": chi2_final,
        "reduced_chi2_final": redchi2_final,
        "dof": dof_final,
        "n_used_pixels": n_pix_final,
    })

    outdir = os.path.join(data_path, "mcmc_outputs", run_name)
    os.makedirs(outdir, exist_ok=True)

    fits_file = save_mcmc_results_extended(
        data_path=data_path,
        run_name=f"{run_name}_{timestamp}_bin{metadata.get('nyquist_bin', 'NA')}_discard{discard}_thin{thin}",
        basename=basename,
        flat_samples=flat_samples,
        medians=medians,
        q16_q84=q16_q84,
        metadata=metadata,
    )

    trace_plot = save_trace_plot(
        sampler=sampler,
        param_names=metadata["param_names"],
        outdir=outdir,
        run_name=run_name,
        metadata=metadata,
        discard=discard,
        thin=thin,
        timestamp=timestamp,
    )

    corner_plot = save_corner_plot(
        flat_samples=flat_samples,
        param_names=metadata["param_names"],
        outdir=outdir,
        run_name=run_name,
        metadata=metadata,
        discard=discard,
        thin=thin,
        timestamp=timestamp,
    )

    bestfit_plot = save_bestfit_spectrum_plot(
        testdata=testdata,
        moognn=moognn,
        medians=medians,
        outdir=outdir,
        run_name=run_name,
        fixed_vsini=fixed_vsini,
        metadata=metadata,
        discard=discard,
        thin=thin,
        timestamp=timestamp,
    )

    residual_plot = save_residual_spectrum_plot(
        testdata=testdata,
        moognn=moognn,
        medians=medians,
        outdir=outdir,
        run_name=run_name,
        fixed_vsini=fixed_vsini,
        metadata=metadata,
        discard=discard,
        thin=thin,
        timestamp=timestamp,
    )

    append_mcmc_summary_csv(
        csv_path=summary_csv,
        source_name=source_name,
        basename=basename,
        run_name=run_name,
        stage=stage,
        medians=medians,
        errs=q16_q84,
        metadata=metadata,
        fits_file=fits_file,
        trace_plot=trace_plot,
        corner_plot=corner_plot,
        bestfit_plot=bestfit_plot,
        residual_plot=residual_plot,
    )

    return {
        "run_name": run_name,
        "stage": stage,
        "flat_samples": flat_samples,
        "medians": medians,
        "percentiles": q16_q84,
        "metadata": metadata,
        "fits_file": fits_file,
        "trace_plot": trace_plot,
        "corner_plot": corner_plot,
        "bestfit_plot": bestfit_plot,
        "residual_plot": residual_plot,
        "summary_csv": summary_csv,
    }


def run_two_stage_source_raw(
    basename,
    moognn,
    data_path="data/science",
    nwalkers=64,
    nsteps=4000,
    scale_discard=1000,
    scale_thin=5,
    vsini_discard=1000,
    vsini_thin=5,
    stage2_fixed_vsini=None,
):
    """
    Run Stage 1 and Stage 2, but do not save plots/summary yet.

    Stage 1:
        regions = [0, 1, 2, 3, 4, 5, 6]
        free parameters = Teff, logg, rK, B, vsini

    Stage 2:
        regions = [0, 1, 2, 3, 4, 5]
        free parameters = Teff, logg, rK, B
        fixed vsini = either Stage 1 median vsini or manual value

    Parameters
    ----------
    stage2_fixed_vsini : float or None
        If None, Stage 2 uses the Stage 1 median vsini.
        If float, Stage 2 uses this manually supplied vsini value.
    """

    source_name = os.path.splitext(basename)[0]

    # ============================================================
    # Stage 1: 7 regions, 5 free parameters
    # ============================================================
    stage1_name = f"{source_name}_stage1_7reg_5par"

    stage1_raw = run_error_rescaled_mcmc(
        basename=basename,
        run_name=stage1_name,
        moognn=moognn,
        data_path=data_path,
        regions=[0, 1, 2, 3, 4, 5, 6],
        fixed_vsini=None,
        nwalkers=nwalkers,
        nsteps=nsteps,
        scale_discard=scale_discard,
        scale_thin=scale_thin,
    )

    # ============================================================
    # Choose fixed vsini for Stage 2
    # ============================================================
    if stage2_fixed_vsini is None:
        flat1_for_vsini = stage1_raw["sampler"].get_chain(
            discard=vsini_discard,
            thin=vsini_thin,
            flat=True,
        )

        med1_for_vsini = np.median(flat1_for_vsini, axis=0)
        fixed_vsini = med1_for_vsini[4]
        fixed_vsini_source = "stage1_median"

    else:
        fixed_vsini = float(stage2_fixed_vsini)
        fixed_vsini_source = "manual"


    # ============================================================
    # Stage 2: 6 regions, 4 free parameters, fixed vsini
    # ============================================================
    stage2_name = f"{source_name}_stage2_6reg_4par_fixedvsini"

    stage2_raw = run_error_rescaled_mcmc(
        basename=basename,
        run_name=stage2_name,
        moognn=moognn,
        data_path=data_path,
        regions=[0, 1, 2, 3, 4, 5],
        fixed_vsini=fixed_vsini,
        nwalkers=nwalkers,
        nsteps=nsteps,
        scale_discard=scale_discard,
        scale_thin=scale_thin,
    )

    stage1_raw["name"] = stage1_name
    stage1_raw["stage"] = "stage1_7regions_5params"

    stage2_raw["name"] = stage2_name
    stage2_raw["stage"] = "stage2_6regions_4params_fixed_vsini"

    # Store provenance in Stage 2 metadata too
    stage2_raw["metadata"]["fixed_vsini"] = fixed_vsini
    stage2_raw["metadata"]["fixed_vsini_source"] = fixed_vsini_source
    stage2_raw["metadata"]["vsini_discard"] = vsini_discard
    stage2_raw["metadata"]["vsini_thin"] = vsini_thin

    return {
        "source_name": source_name,
        "basename": basename,
        "stage1": stage1_raw,
        "stage2": stage2_raw,
        "fixed_vsini": fixed_vsini,
        "fixed_vsini_source": fixed_vsini_source,
        "vsini_discard": vsini_discard,
        "vsini_thin": vsini_thin,
    }

def postprocess_two_stage_results(
    raw_results,
    moognn,
    data_path="data/science",
    discard=1000,
    thin=5,
    summary_csv=None,
    timestamp=None,
):
    """
    Postprocess Stage 1 and Stage 2 with a chosen discard/thin.
    You can call this many times without rerunning MCMC.
    """


    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    basename = raw_results["basename"]
    source_name = raw_results["source_name"]

    stage1_out = postprocess_mcmc_run(
        run_result=raw_results["stage1"],
        basename=basename,
        source_name=source_name,
        run_name=raw_results["stage1"]["name"],
        stage=raw_results["stage1"]["stage"],
        moognn=moognn,
        data_path=data_path,
        discard=discard,
        thin=thin,
        summary_csv=summary_csv,
        timestamp=timestamp,
    )

    stage2_out = postprocess_mcmc_run(
        run_result=raw_results["stage2"],
        basename=basename,
        source_name=source_name,
        run_name=raw_results["stage2"]["name"],
        stage=raw_results["stage2"]["stage"],
        moognn=moognn,
        data_path=data_path,
        discard=discard,
        thin=thin,
        summary_csv=summary_csv,
        timestamp=timestamp,
    )

    return {
        "stage1": stage1_out,
        "stage2": stage2_out,
        "summary_csv": stage1_out["summary_csv"],
        "timestamp": timestamp,
        "fixed_vsini": raw_results["fixed_vsini"],
    }

def append_mcmc_summary_csv(
    csv_path,
    source_name,
    basename,
    run_name,
    stage,
    medians,
    errs,
    metadata,
    fits_file=None,
    trace_plot=None,
    corner_plot=None,
    bestfit_plot=None,
    residual_plot=None,
):
    """
    Append one MCMC result row to a human-readable CSV file.

    Parameters
    ----------
    medians : array
        Median posterior values.
        Stage 1 order: Teff, logg, rK, B, vsini
        Stage 2 order: Teff, logg, rK, B

    errs : array
        Percentiles from np.percentile(flat_samples, [16, 84], axis=0)

    metadata : dict
        Contains regions, shifts, renormalization, chi2, error scaling, etc.
    """

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    fixed_vsini = metadata.get("fixed_vsini", None)

    # ----------------------------
    # Handle Stage 1: 5 parameters
    # ----------------------------
    if fixed_vsini is None:
        Teff, logg, rK, B, vsini = medians

        Teff_err_minus = Teff - errs[0, 0]
        Teff_err_plus  = errs[1, 0] - Teff

        logg_err_minus = logg - errs[0, 1]
        logg_err_plus  = errs[1, 1] - logg

        rK_err_minus = rK - errs[0, 2]
        rK_err_plus  = errs[1, 2] - rK

        B_err_minus = B - errs[0, 3]
        B_err_plus  = errs[1, 3] - B

        vsini_err_minus = vsini - errs[0, 4]
        vsini_err_plus  = errs[1, 4] - vsini

        vsini_status = "free"

    # ----------------------------
    # Handle Stage 2: 4 parameters
    # ----------------------------
    else:
        Teff, logg, rK, B = medians
        vsini = fixed_vsini

        Teff_err_minus = Teff - errs[0, 0]
        Teff_err_plus  = errs[1, 0] - Teff

        logg_err_minus = logg - errs[0, 1]
        logg_err_plus  = errs[1, 1] - logg

        rK_err_minus = rK - errs[0, 2]
        rK_err_plus  = errs[1, 2] - rK

        B_err_minus = B - errs[0, 3]
        B_err_plus  = errs[1, 3] - B

        # Since vsini was fixed, the Stage 2 chain has no vsini uncertainty.
        vsini_err_minus = np.nan
        vsini_err_plus  = np.nan

        vsini_status = "fixed_from_stage1"

    row = {
        "source_name": source_name,
        "basename": basename,
        "run_name": run_name,
        "stage": stage,

        "regions": str(metadata["regions"]),
        "n_regions": metadata["n_regions"],
        "n_free_params": len(medians),
        "fixed_vsini": fixed_vsini,
        "fixed_vsini_source": metadata.get("fixed_vsini_source"),
        "vsini_discard": metadata.get("vsini_discard"),
        "vsini_thin": metadata.get("vsini_thin"),
        "vsini_status": vsini_status,

        "Teff": Teff,
        "Teff_err_minus": Teff_err_minus,
        "Teff_err_plus": Teff_err_plus,

        "logg": logg,
        "logg_err_minus": logg_err_minus,
        "logg_err_plus": logg_err_plus,

        "rK": rK,
        "rK_err_minus": rK_err_minus,
        "rK_err_plus": rK_err_plus,

        "B": B,
        "B_err_minus": B_err_minus,
        "B_err_plus": B_err_plus,

        "vsini": vsini,
        "vsini_err_minus": vsini_err_minus,
        "vsini_err_plus": vsini_err_plus,

        "chi2_initial": metadata.get("chi2_initial"),
        "reduced_chi2_initial": metadata.get("reduced_chi2_initial"),
        "error_scale_factor": metadata.get("error_scale_factor"),
        "chi2_final": metadata.get("chi2_final"),
        "reduced_chi2_final": metadata.get("reduced_chi2_final"),
        "dof": metadata.get("dof"),
        "n_used_pixels": metadata.get("n_used_pixels"),

        "shifts": str(metadata.get("shifts")),
        "renormalization": str(metadata.get("renormalization")),
        "masks": str(metadata.get("masks")),

        "nwalkers": metadata.get("nwalkers"),
        "nsteps": metadata.get("nsteps"),
        "discard": metadata.get("discard"),
        "thin": metadata.get("thin"),
        "scale_discard": metadata.get("scale_discard"),
        "scale_thin": metadata.get("scale_thin"),

        "fits_file": fits_file,
        "trace_plot": trace_plot,
        "corner_plot": corner_plot,
        "bestfit_plot": bestfit_plot,
        "residual_plot": residual_plot,

    }

    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)
