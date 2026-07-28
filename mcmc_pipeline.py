import csv
import os
from datetime import datetime

import numpy as np

from mcmc import (
    PARAM_ORDER,
    compute_reduced_chi2,
    fit_params_mcmc,
    flat_lnprob_from_array,
    flat_samples_from_chain,
    free_params_from_fixed,
    load_mcmc_run,
    normalize_fixed_params,
    percentile_summary,
    retrieve_spectrum_preproc,
    save_mcmc_run,
    save_mcmc_results_extended,
    summary_values_from_free,
)
from mcmc_plots import (
    save_bestfit_spectrum_plot,
    save_corner_plot,
    save_residual_spectrum_plot,
    save_trace_plot_from_chain,
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def resolve_project_path(path):
    if path is None or os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def make_regions_tag(regions):
    return "reg" + "-".join(str(r) for r in regions)


def make_fixed_tag(fixed_params):
    fixed_params = normalize_fixed_params(fixed_params)
    if not fixed_params:
        return "allfree"
    pieces = []
    for k in PARAM_ORDER:
        if k in fixed_params:
            pieces.append(f"fix{k}{fixed_params[k]:g}".replace(".", "p"))
    return "_".join(pieces)


def make_run_name(source_name, stage_label, regions, fixed_params=None):
    return f"{source_name}_{stage_label}_{make_regions_tag(regions)}_{make_fixed_tag(fixed_params)}"


def run_stage_raw(
    basename,
    moognn,
    data_path="data/science",
    regions=(0, 1, 2, 3, 4, 5, 6),
    fixed_params=None,
    free_param_names=None,
    stage_label="stage",
    run_name=None,
    nwalkers=64,
    nsteps=4000,
    scale_discard=1000,
    scale_thin=5,
    progress=True,
    mcmc_runs_dir="mcmc_runs",
    timestamp=None,
    save_mcmc_run_file=True,
):
    """Run one independent MCMC stage with arbitrary regions and fixed parameters.

    This performs the expensive part only:
      1. first MCMC
      2. compute reduced chi2 using scale_discard/scale_thin
      3. rescale yerr
      4. final MCMC

    It does not save plots/summary; call postprocess_stage_result() afterward.
    """
    source_name = os.path.splitext(basename)[0]
    regions = list(regions)
    fixed_params = normalize_fixed_params(fixed_params)
    free_param_names = free_params_from_fixed(fixed_params, free_param_names)

    if run_name is None:
        run_name = make_run_name(source_name, stage_label, regions, fixed_params)
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    testdata, meta = retrieve_spectrum_preproc(
        basename,
        data_path=data_path,
        regions_override=regions,
        return_metadata=True,
    )

    sampler1 = fit_params_mcmc(
        testdata,
        moognn,
        nwalkers=nwalkers,
        nsteps=nsteps,
        fixed_params=fixed_params,
        free_param_names=free_param_names,
        progress=progress,
    )

    flat1_for_scale = sampler1.get_chain(discard=scale_discard, thin=scale_thin, flat=True)
    med1_for_scale = np.median(flat1_for_scale, axis=0)

    chi2_1, redchi2_1, dof_1, n_pix_1 = compute_reduced_chi2(
        testdata,
        moognn,
        med1_for_scale,
        free_param_names=free_param_names,
        fixed_params=fixed_params,
    )

    errscale = np.sqrt(redchi2_1)
    testdata.rescale_yerr(errscale)

    sampler2 = fit_params_mcmc(
        testdata,
        moognn,
        nwalkers=nwalkers,
        nsteps=nsteps,
        fixed_params=fixed_params,
        free_param_names=free_param_names,
        progress=progress,
    )

    meta.update({
        "run_name": run_name,
        "stage_label": stage_label,
        "fixed_params": fixed_params,
        "free_param_names": free_param_names,
        "data_path": data_path,
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

    mcmc_run_file = None
    if save_mcmc_run_file:
        mcmc_run_file = save_mcmc_run(
            outdir=resolve_project_path(mcmc_runs_dir),
            run_name=run_name,
            sampler=sampler2,
            metadata=meta,
            timestamp=timestamp,
        )
        meta["mcmc_run_file"] = mcmc_run_file

    return {
        "basename": basename,
        "source_name": source_name,
        "run_name": run_name,
        "stage_label": stage_label,
        "regions": regions,
        "fixed_params": fixed_params,
        "free_param_names": free_param_names,
        "testdata": testdata,
        "sampler": sampler2,
        "metadata": meta,
        "mcmc_run_file": mcmc_run_file,
    }


def append_mcmc_summary_csv(
    csv_path,
    source_name,
    basename,
    run_name,
    stage,
    medians,
    q16_q84,
    free_param_names,
    fixed_params,
    metadata,
    fits_file=None,
    trace_plot=None,
    corner_plot=None,
    bestfit_plot=None,
    residual_plot=None,
    mcmc_run_file=None,
):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    vals = summary_values_from_free(medians, q16_q84, free_param_names, fixed_params)

    row = {
        "source_name": source_name,
        "basename": basename,
        "run_name": run_name,
        "stage": stage,
        "regions": str(metadata.get("regions")),
        "n_regions": metadata.get("n_regions"),
        "free_param_names": str(list(free_param_names)),
        "fixed_params": str(normalize_fixed_params(fixed_params)),
        "n_free_params": len(free_param_names),
        "nwalkers": metadata.get("nwalkers"),
        "nsteps": metadata.get("nsteps"),
        "discard": metadata.get("discard"),
        "thin": metadata.get("thin"),
        "scale_discard": metadata.get("scale_discard"),
        "scale_thin": metadata.get("scale_thin"),
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
        "best_percentile": metadata.get("best_percentile"),
        "uncertainty_percentiles": str(metadata.get("uncertainty_percentiles")),
        "mcmc_run_file": mcmc_run_file,
        "fits_file": fits_file,
        "trace_plot": trace_plot,
        "corner_plot": corner_plot,
        "bestfit_plot": bestfit_plot,
        "residual_plot": residual_plot,
    }

    for p in PARAM_ORDER:
        row[p] = vals[p]
        row[f"{p}_err_minus"] = vals[f"{p}_err_minus"]
        row[f"{p}_err_plus"] = vals[f"{p}_err_plus"]
        row[f"{p}_status"] = vals[f"{p}_status"]

    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def postprocess_stage_result(
    raw_result,
    moognn,
    data_path="data/science",
    discard=1000,
    thin=5,
    summary_csv=None,
    timestamp=None,
    save_fits=False,
    save_plots=True,
    mcmc_runs_dir="mcmc_runs",
    figures_dir="figures",
    mcmc_run_file=None,
    best_percentile=50,
    uncertainty_percentiles=(16, 84),
    corner_percentiles=None,
):
    """Postprocess one already-run stage with a chosen discard/thin.

    This is cheap. You can call it repeatedly with different discard/thin values
    without rerunning the MCMC.
    """
    if summary_csv is None:
        summary_csv = os.path.join(resolve_project_path(mcmc_runs_dir), "mcmc_summary_flexible.csv")
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    basename = raw_result["basename"]
    source_name = raw_result["source_name"]
    run_name = raw_result["run_name"]
    stage_label = raw_result["stage_label"]
    testdata = raw_result["testdata"]
    fixed_params = normalize_fixed_params(raw_result["fixed_params"])
    free_param_names = list(raw_result["free_param_names"])
    metadata = raw_result["metadata"].copy()
    mcmc_run_file = mcmc_run_file or raw_result.get("mcmc_run_file") or metadata.get("mcmc_run_file")

    if mcmc_run_file is None:
        mcmc_run_file = save_mcmc_run(
            outdir=resolve_project_path(mcmc_runs_dir),
            run_name=run_name,
            sampler=raw_result["sampler"],
            metadata=metadata,
            timestamp=timestamp,
        )

    run_payload = load_mcmc_run(mcmc_run_file)
    chain = run_payload["chain"]
    lnprob = run_payload["lnprob"]
    metadata.update(run_payload["metadata"])

    flat_samples = flat_samples_from_chain(chain, discard=discard, thin=thin)
    flat_lnprob = flat_lnprob_from_array(lnprob, discard=discard, thin=thin)
    medians, q16_q84 = percentile_summary(
        flat_samples,
        best_percentile=best_percentile,
        uncertainty_percentiles=uncertainty_percentiles,
    )

    chi2_final, redchi2_final, dof_final, n_pix_final = compute_reduced_chi2(
        testdata,
        moognn,
        medians,
        free_param_names=free_param_names,
        fixed_params=fixed_params,
    )

    metadata.update({
        "discard": discard,
        "thin": thin,
        "chi2_final": chi2_final,
        "reduced_chi2_final": redchi2_final,
        "dof": dof_final,
        "n_used_pixels": n_pix_final,
        "fixed_params": fixed_params,
        "free_param_names": free_param_names,
        "best_percentile": best_percentile,
        "uncertainty_percentiles": list(uncertainty_percentiles),
        "mcmc_run_file": mcmc_run_file,
    })

    outdir = resolve_project_path(figures_dir)
    os.makedirs(outdir, exist_ok=True)
    tag = f"{timestamp}_bin{metadata.get('nyquist_bin', 'NA')}_discard{discard}_thin{thin}"

    fits_file = None
    if save_fits:
        fits_file = save_mcmc_results_extended(
            data_path=data_path,
            run_name=f"{run_name}_{tag}",
            basename=basename,
            flat_samples=flat_samples,
            medians=medians,
            q16_q84=q16_q84,
            metadata=metadata,
        )

    trace_plot = corner_plot = bestfit_plot = residual_plot = None
    if save_plots:
        trace_plot = save_trace_plot_from_chain(
            chain=chain,
            param_names=free_param_names,
            outdir=outdir,
            run_name=run_name,
            metadata=metadata,
            discard=discard,
            thin=thin,
            timestamp=timestamp,
        )
        if corner_percentiles is None:
            corner_percentiles = [uncertainty_percentiles[0], best_percentile, uncertainty_percentiles[1]]
        corner_quantiles = [p / 100 for p in corner_percentiles]
        corner_plot = save_corner_plot(
            flat_samples=flat_samples,
            param_names=free_param_names,
            outdir=outdir,
            run_name=run_name,
            metadata=metadata,
            discard=discard,
            thin=thin,
            timestamp=timestamp,
            quantiles=corner_quantiles,
        )
        bestfit_plot = save_bestfit_spectrum_plot(
            testdata=testdata,
            moognn=moognn,
            medians=medians,
            free_param_names=free_param_names,
            fixed_params=fixed_params,
            outdir=outdir,
            run_name=run_name,
            metadata=metadata,
            discard=discard,
            thin=thin,
            timestamp=timestamp,
        )
        residual_plot = save_residual_spectrum_plot(
            testdata=testdata,
            moognn=moognn,
            medians=medians,
            free_param_names=free_param_names,
            fixed_params=fixed_params,
            outdir=outdir,
            run_name=run_name,
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
        stage=stage_label,
        medians=medians,
        q16_q84=q16_q84,
        free_param_names=free_param_names,
        fixed_params=fixed_params,
        metadata=metadata,
        fits_file=fits_file,
        trace_plot=trace_plot,
        corner_plot=corner_plot,
        bestfit_plot=bestfit_plot,
        residual_plot=residual_plot,
        mcmc_run_file=mcmc_run_file,
    )

    vals = summary_values_from_free(medians, q16_q84, free_param_names, fixed_params)

    return {
        "run_name": run_name,
        "stage": stage_label,
        "flat_samples": flat_samples,
        "flat_lnprob": flat_lnprob,
        "chain": chain,
        "lnprob": lnprob,
        "medians": medians,
        "percentiles": q16_q84,
        "values": vals,
        "metadata": metadata,
        "fits_file": fits_file,
        "trace_plot": trace_plot,
        "corner_plot": corner_plot,
        "bestfit_plot": bestfit_plot,
        "residual_plot": residual_plot,
        "mcmc_run_file": mcmc_run_file,
        "summary_csv": summary_csv,
        "timestamp": timestamp,
    }


def postprocess_mcmc_run_file(
    mcmc_run_file,
    moognn,
    data_path=None,
    discard=1000,
    thin=5,
    summary_csv=None,
    timestamp=None,
    save_fits=False,
    save_plots=True,
    mcmc_runs_dir="mcmc_runs",
    figures_dir="figures",
    best_percentile=50,
    uncertainty_percentiles=(16, 84),
    corner_percentiles=None,
):
    """Regenerate summaries and plots from a saved MCMC run without rerunning MCMC."""
    run_payload = load_mcmc_run(mcmc_run_file)
    metadata = run_payload["metadata"]
    basename = metadata["basename"]
    if data_path is None:
        data_path = metadata.get("data_path", "data/science")

    testdata, preproc_meta = retrieve_spectrum_preproc(
        basename,
        data_path=data_path,
        regions_override=metadata.get("regions"),
        return_metadata=True,
    )
    errscale = metadata.get("error_scale_factor")
    if errscale is not None and np.isfinite(errscale):
        testdata.rescale_yerr(errscale)

    merged_metadata = preproc_meta.copy()
    merged_metadata.update(metadata)
    source_name = os.path.splitext(basename)[0]
    raw_result = {
        "basename": basename,
        "source_name": source_name,
        "run_name": metadata.get("run_name", source_name),
        "stage_label": metadata.get("stage_label", "stage"),
        "regions": list(metadata.get("regions", [])),
        "fixed_params": run_payload["fixed_params"],
        "free_param_names": run_payload["free_param_names"],
        "testdata": testdata,
        "metadata": merged_metadata,
        "mcmc_run_file": mcmc_run_file,
    }

    return postprocess_stage_result(
        raw_result,
        moognn=moognn,
        data_path=data_path,
        discard=discard,
        thin=thin,
        summary_csv=summary_csv,
        timestamp=timestamp,
        save_fits=save_fits,
        save_plots=save_plots,
        mcmc_runs_dir=mcmc_runs_dir,
        figures_dir=figures_dir,
        mcmc_run_file=mcmc_run_file,
        best_percentile=best_percentile,
        uncertainty_percentiles=uncertainty_percentiles,
        corner_percentiles=corner_percentiles,
    )
