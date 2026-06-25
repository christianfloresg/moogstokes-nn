# from mcmc import *
import os
import csv
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
    discard=500,
    thin=5,
):
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

    flat1 = sampler1.get_chain(discard=discard, thin=thin, flat=True)
    med1 = np.median(flat1, axis=0)

    chi2_1, redchi2_1, dof_1, n_pix_1 = compute_reduced_chi2(
        testdata,
        moognn,
        med1,
        fixed_vsini=fixed_vsini,
    )

    errscale = np.sqrt(redchi2_1)

    # Rescale errors and regenerate fitting regions
    testdata.rescale_yerr(errscale)

    # Second pass: this is the posterior you should save/use
    sampler2 = fit_params_mcmc(
        testdata,
        moognn,
        nwalkers=nwalkers,
        nsteps=nsteps,
        vsini=fixed_vsini,
    )

    flat2 = sampler2.get_chain(discard=discard, thin=thin, flat=True)
    med2 = np.median(flat2, axis=0)
    q16_q84 = np.percentile(flat2, [16, 84], axis=0)

    chi2_2, redchi2_2, dof_2, n_pix_2 = compute_reduced_chi2(
        testdata,
        moognn,
        med2,
        fixed_vsini=fixed_vsini,
    )

    meta.update({
        "run_name": run_name,
        "fixed_vsini": fixed_vsini,
        "param_names": PARAM_NAMES_4D if fixed_vsini is not None else PARAM_NAMES_5D,
        "chi2_initial": chi2_1,
        "reduced_chi2_initial": redchi2_1,
        "error_scale_factor": errscale,
        "chi2_final": chi2_2,
        "reduced_chi2_final": redchi2_2,
        "dof": dof_2,
        "n_used_pixels": n_pix_2,
    })

    return testdata, sampler2, flat2, med2, q16_q84, meta



def run_two_stage_source(
    basename,
    moognn,
    data_path="data/science",
    nwalkers=64,
    nsteps=4000,
    discard=1000,
    thin=5,
    summary_csv=None,
):
    source_name = os.path.splitext(basename)[0]

    if summary_csv is None:
        summary_csv = os.path.join(data_path, "mcmc_outputs", "mcmc_summary_extended.csv")

    # ============================================================
    # Stage 1: 7 regions, 5 free parameters
    # ============================================================
    stage1_name = f"{source_name}_stage1_7reg_5par"

    data1, sampler1, flat1, med1, q1, meta1 = run_error_rescaled_mcmc(
        basename=basename,
        run_name=stage1_name,
        moognn=moognn,
        data_path=data_path,
        regions=[0, 1, 2, 3, 4, 5, 6],
        fixed_vsini=None,
        nwalkers=nwalkers,
        nsteps=nsteps,
        discard=discard,
        thin=thin,
    )

    fits1 = save_mcmc_results_extended(
        data_path=data_path,
        run_name=stage1_name,
        basename=basename,
        flat_samples=flat1,
        medians=med1,
        q16_q84=q1,
        metadata=meta1,
    )

    outdir1 = os.path.join(data_path, "mcmc_outputs", stage1_name)

    trace1 = save_trace_plot(
        sampler=sampler1,
        param_names=meta1["param_names"],
        outdir=outdir1,
        run_name=stage1_name,
        discard=discard,
    )

    corner1 = save_corner_plot(
        flat_samples=flat1,
        param_names=meta1["param_names"],
        outdir=outdir1,
        run_name=stage1_name,
    )

    bestfit1 = save_bestfit_spectrum_plot(
        testdata=data1,
        moognn=moognn,
        medians=med1,
        outdir=outdir1,
        run_name=stage1_name,
        fixed_vsini=None,
    )

    append_mcmc_summary_csv(
        csv_path=summary_csv,
        source_name=source_name,
        basename=basename,
        run_name=stage1_name,
        stage="stage1_7regions_5params",
        medians=med1,
        errs=q1,
        metadata=meta1,
        fits_file=fits1,
        trace_plot=trace1,
        corner_plot=corner1,
        bestfit_plot=bestfit1,
    )

    fixed_vsini = med1[4]

    # ============================================================
    # Stage 2: 6 regions, 4 free parameters, fixed vsini
    # ============================================================
    stage2_name = f"{source_name}_stage2_6reg_4par_fixedvsini"

    data2, sampler2, flat2, med2, q2, meta2 = run_error_rescaled_mcmc(
        basename=basename,
        run_name=stage2_name,
        moognn=moognn,
        data_path=data_path,
        regions=[0, 1, 2, 3, 4, 5],
        fixed_vsini=fixed_vsini,
        nwalkers=nwalkers,
        nsteps=nsteps,
        discard=discard,
        thin=thin,
    )

    fits2 = save_mcmc_results_extended(
        data_path=data_path,
        run_name=stage2_name,
        basename=basename,
        flat_samples=flat2,
        medians=med2,
        q16_q84=q2,
        metadata=meta2,
    )

    outdir2 = os.path.join(data_path, "mcmc_outputs", stage2_name)

    trace2 = save_trace_plot(
        sampler=sampler2,
        param_names=meta2["param_names"],
        outdir=outdir2,
        run_name=stage2_name,
        discard=discard,
    )

    corner2 = save_corner_plot(
        flat_samples=flat2,
        param_names=meta2["param_names"],
        outdir=outdir2,
        run_name=stage2_name,
    )

    bestfit2 = save_bestfit_spectrum_plot(
        testdata=data2,
        moognn=moognn,
        medians=med2,
        outdir=outdir2,
        run_name=stage2_name,
        fixed_vsini=fixed_vsini,
    )

    append_mcmc_summary_csv(
        csv_path=summary_csv,
        source_name=source_name,
        basename=basename,
        run_name=stage2_name,
        stage="stage2_6regions_4params_fixed_vsini",
        medians=med2,
        errs=q2,
        metadata=meta2,
        fits_file=fits2,
        trace_plot=trace2,
        corner_plot=corner2,
        bestfit_plot=bestfit2,
    )

    return {
        "stage1": {
            "name": stage1_name,
            "fits": fits1,
            "trace_plot": trace1,
            "corner_plot": corner1,
            "bestfit_plot": bestfit1,
            "medians": med1,
            "percentiles": q1,
            "metadata": meta1,
        },
        "stage2": {
            "name": stage2_name,
            "fits": fits2,
            "trace_plot": trace2,
            "corner_plot": corner2,
            "bestfit_plot": bestfit2,
            "medians": med2,
            "percentiles": q2,
            "metadata": meta2,
        },
        "summary_csv": summary_csv,
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

        "fits_file": fits_file,
        "trace_plot": trace_plot,
        "corner_plot": corner_plot,
        "bestfit_plot": bestfit_plot,
    }

    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)
