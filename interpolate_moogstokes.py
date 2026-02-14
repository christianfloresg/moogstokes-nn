import os
import time
import numpy as np
import pandas as pd
import warnings

from itertools import product
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from typing import Dict, Iterable, Tuple, Optional
from spectra import MoogStokesModel

def check_moogstokes_grid_rectangularity(
    models_dir: str,
    interpolation_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    verbose: bool = True,
    missing_outfile: Optional[str] = None,
) -> bool:
    """
    Check whether the MoogStokes model grid is rectangular (complete)
    over Teff, logg, B, and vsini within optional interpolation bounds.

    Parameters
    ----------
    models_dir : str
        Directory containing original Moog-Stokes model folders.

    interpolation_bounds : dict or None
        Optional bounds of the form:
        {
            "Teff": (min, max),
            "logg": (min, max),
            "B": (min, max),
            "vsini": (min, max),
        }

    verbose : bool
        If True, prints diagnostic information and missing grid points.

    missing_outfile : str or None
        If provided, writes missing parameter combinations to a
        space-delimited text file.

    Returns
    -------
    rectangular : bool
        True if the grid is complete and rectangular, False otherwise.
    """

    observed = set()

    for folder in os.listdir(models_dir):
        folder_path = os.path.join(models_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        try:
            parts = folder.split('_')
            Teff = int(parts[2][1:])
            logg = int(parts[3][1:]) / 1000.0
            B = float(parts[5][2:])
            vsini = float(parts[6][4:])
        except Exception:
            continue

        params = dict(Teff=Teff, logg=logg, B=B, vsini=vsini)

        if interpolation_bounds is not None:
            if any(
                params[p] < pmin or params[p] > pmax
                for p, (pmin, pmax) in interpolation_bounds.items()
            ):
                continue

        observed.add((Teff, logg, B, vsini))

    if not observed:
        raise RuntimeError(
            "No models found within the specified interpolation bounds."
        )

    Teff_vals  = sorted({p[0] for p in observed})
    logg_vals  = sorted({p[1] for p in observed})
    B_vals     = sorted({p[2] for p in observed})
    vsini_vals = sorted({p[3] for p in observed})

    expected = set(product(Teff_vals, logg_vals, B_vals, vsini_vals))
    missing = expected - observed

    if missing and missing_outfile is not None:
        with open(missing_outfile, "w") as f:
            f.write("Teff logg B vsini\n")
            for T, g, Bv, v in sorted(missing):
                f.write(f"{T} {g} {Bv} {v}\n")

    if missing and verbose:
        print(
            f"Grid is NOT rectangular: "
            f"{len(missing)} missing parameter combinations."
        )
        max_show = 20
        for i, (T, g, Bv, v) in enumerate(sorted(missing)):
            if i >= max_show:
                print(f"  ... ({len(missing) - max_show} more)")
                break
            print(f"  Teff={T}, logg={g}, B={Bv}, vsini={v}")

    if not missing and verbose:
        print(
            "Grid is rectangular and complete.\n"
            f"  Teff:  {Teff_vals}\n"
            f"  logg:  {logg_vals}\n"
            f"  B:     {B_vals}\n"
            f"  vsini: {vsini_vals}"
        )

    return len(missing) == 0

def interpolate_moogstokes_models(
    models_dir: str,
    output_dir: str,
    region: int,
    target_grids: dict,
    interpolation_bounds: dict | None = None,
    verbose: bool = False,
):
    """
    Wrapper that converts Cartesian target_grids
    into a list of parameter points.
    """

    # remove veiling parameter
    target_grids = {k: v for k, v in target_grids.items() if k != "rK"}

    Teff_vals = target_grids["Teff"]
    logg_vals = target_grids["logg"]
    B_vals = target_grids["B"]
    vsini_vals = target_grids["vsini"]

    # Cartesian product
    target_params = np.array([
        [Teff, logg, B, vsini]
        for Teff in Teff_vals
        for logg in logg_vals
        for B in B_vals
        for vsini in vsini_vals
    ])

    interpolate_moogstokes_to_points(
        models_dir=models_dir,
        output_dir=output_dir,
        region=region,
        target_params=target_params,
        interpolation_bounds=interpolation_bounds,
        verbose=verbose
    )


def interpolate_moogstokes_to_points(
    models_dir: str,
    output_dir: str,
    region: int,
    target_params: np.ndarray,  # shape (N, 4)
    interpolation_bounds: dict | None = None,
    verbose: bool = False,
):
    """
    Interpolate MoogStokes models onto an arbitrary array of
    parameter points.
    """

    if target_params.ndim != 2 or target_params.shape[1] != 4:
        raise ValueError("target_params must have shape (N, 4)")

    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    print("\n" + "-" * 72)
    print(f"Interpolating MoogStokes models (region={region})")
    vprint("-" * 72)

    param_cols = ["Teff", "logg", "B", "vsini"]
    tstart = time.time()

    # ----------------------------------------------------------
    # 1. Load model grid
    # ----------------------------------------------------------
    print("[1] Loading original models")
    vprint("    Applying interpolation bounds and reading spectra...")

    records = []

    for folder in os.listdir(models_dir):
        folder_path = os.path.join(models_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        try:
            parts = folder.split("_")
            Teff = int(parts[2][1:])
            logg = int(parts[3][1:]) / 1000.0
            B = float(parts[5][2:])
            vsini = float(parts[6][4:])
        except Exception:
            continue

        params = dict(Teff=Teff, logg=logg, B=B, vsini=vsini)

        if interpolation_bounds is not None:
            if any(
                params[p] < pmin or params[p] > pmax
                for p, (pmin, pmax) in interpolation_bounds.items()
            ):
                continue

        fname = MoogStokesModel.make_model_fname(
            Teff=Teff,
            logg=logg,
            rK=0.0,
            B=B,
            vsini=vsini,
            region=region,
            models_dir=models_dir,
            use_hash=False,
        )

        if not os.path.exists(fname):
            continue

        try:
            data = np.loadtxt(fname)
        except Exception:
            continue

        records.append(
            {
                **params,
                "wave": data[:, 0],
                "flux": data[:, 1],
            }
        )

    if len(records) == 0:
        raise RuntimeError("No valid input models found.")

    df = pd.DataFrame(records)

    vprint(f"    Loaded {len(df)} models.")

    # ----------------------------------------------------------
    # 2. Enforce common wavelength grid
    # ----------------------------------------------------------
    print("[2] Enforcing common wavelength grid")

    min_len = df["flux"].apply(len).min()
    df["wave"] = df["wave"].apply(lambda x: x[:min_len])
    df["flux"] = df["flux"].apply(lambda x: x[:min_len])
    wave_ref = df["wave"].iloc[0]

    vprint(f"    Truncated spectra to {min_len} wavelength points.")

    # ----------------------------------------------------------
    # 3. Bounds check (no extrapolation)
    # ----------------------------------------------------------
    print("[3] Validating interpolation bounds")
    vprint("    Ensuring no target parameters fall outside model grid...")

    param_min = df[param_cols].min().values
    param_max = df[param_cols].max().values

    if np.any(target_params < param_min) or np.any(target_params > param_max):
        raise ValueError(
            "Some target points fall outside model grid bounds."
        )

    # ----------------------------------------------------------
    # 4. Check rectangularity
    # ----------------------------------------------------------
    print("[4] Checking grid structure")

    Teff_vals = np.sort(df["Teff"].unique())
    logg_vals = np.sort(df["logg"].unique())
    B_vals = np.sort(df["B"].unique())
    vsini_vals = np.sort(df["vsini"].unique())

    expected = (
        len(Teff_vals)
        * len(logg_vals)
        * len(B_vals)
        * len(vsini_vals)
    )

    n_unique = (
        df[["Teff", "logg", "B", "vsini"]]
        .drop_duplicates()
        .shape[0]
    )

    rectangular = (n_unique == expected)

    if rectangular:
        print("    Rectangular grid detected")
        vprint(f"    Unique models: {n_unique} / Expected: {expected}")
    else:
        print("    Incomplete grid detected")
        vprint(f"    Unique models: {n_unique} / Expected: {expected}")

    # ----------------------------------------------------------
    # 5. Build interpolator
    # ----------------------------------------------------------
    print("[5] Building interpolator")

    if rectangular:

        idx_T = {v: i for i, v in enumerate(Teff_vals)}
        idx_g = {v: i for i, v in enumerate(logg_vals)}
        idx_B = {v: i for i, v in enumerate(B_vals)}
        idx_v = {v: i for i, v in enumerate(vsini_vals)}

        nλ = min_len
        flux_hypercube = np.empty(
            (len(Teff_vals), len(logg_vals),
             len(B_vals), len(vsini_vals), nλ)
        )

        for _, row in df.iterrows():
            flux_hypercube[
                idx_T[row.Teff],
                idx_g[row.logg],
                idx_B[row.B],
                idx_v[row.vsini],
                :
            ] = row.flux

        interpolator = RegularGridInterpolator(
            (Teff_vals, logg_vals, B_vals, vsini_vals),
            flux_hypercube,
            bounds_error=True,
        )

        flux_out = interpolator(target_params)

    else:

        warnings.warn(
            "Model grid incomplete. Using LinearNDInterpolator.",
            RuntimeWarning,
        )

        param_grid = df[param_cols].values
        flux_grid = np.vstack(df["flux"].values)

        interpolators = [
            LinearNDInterpolator(param_grid, flux_grid[:, i])
            for i in range(flux_grid.shape[1])
        ]

        flux_out = np.vstack([
            [interp(p) for interp in interpolators]
            for p in target_params
        ])

    # ----------------------------------------------------------
    # 6. Write models
    # ----------------------------------------------------------
    print("[6] Saving interpolated models")
    vprint("    Writing spectra to disk...")

    for p, flux in zip(target_params, flux_out):
        Teff, logg, B, vsini = p

        out_fname = MoogStokesModel.make_model_fname(
            Teff=Teff,
            logg=logg,
            rK=0.0,
            B=B,
            vsini=vsini,
            region=region,
            models_dir=output_dir,
            use_hash=True,
        )

        os.makedirs(os.path.dirname(out_fname), exist_ok=True)
        np.savetxt(
            out_fname,
            np.column_stack((wave_ref, flux)),
            fmt="%.10f",
        )

    total_time = time.time() - tstart
    vprint("-" * 72)
    print(f"Finished in {total_time:.1f} seconds")
    print("-" * 72 + "\n")

    return wave_ref, flux_out
