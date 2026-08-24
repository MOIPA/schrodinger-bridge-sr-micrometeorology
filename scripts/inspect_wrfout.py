"""
Analyze WRF output files to check available variables for NS equation constraints.

Run on server:
  python scripts/inspect_wrfout.py --data_dir /path/to/wrfout --domain d04

This script:
  1. Lists all variables in a sample WRF output file
  2. Checks for key NS-related variables (P, PB, TKE, KM, KH, etc.)
  3. Reports vertical levels, time steps, and spatial dimensions
  4. Computes what fraction of NS equation terms can now be constrained

Python 2/3 compatible, ASCII only.
"""
import argparse
import os
import sys

import numpy as np

try:
    from netCDF4 import Dataset
except ImportError:
    print("ERROR: netCDF4 not installed. Run: pip install netCDF4")
    sys.exit(1)


# Variables needed for each NS equation term
NS_REQUIRED = {
    # 3 Momentum equations: dU/dt + U*grad(U) = -(1/rho)*grad(P) + fV + nu*laplacian(U)
    "time_derivative": {
        "desc": "dU/dt, dV/dt, dW/dt",
        "needs": ["U", "V", "W"],
        "needs_time": True,
    },
    "advection": {
        "desc": "U*grad(U) + V*grad(V) + W*grad(W) (3D advection)",
        "needs": ["U", "V", "W"],
        "needs_3d": True,
    },
    "pressure_gradient": {
        "desc": "-(1/rho)*grad(P) (3D pressure gradient)",
        "needs": ["P", "PB", "T"],  # P_total = P + PB, rho = P/(R*T)
        "needs_3d": True,
    },
    "coriolis": {
        "desc": "fV, -fU (Coriolis force)",
        "needs": ["U", "V", "XLAT"],
    },
    "turbulent_viscosity": {
        "desc": "nu*laplacian(U) (turbulent diffusion)",
        "needs": ["TKE", "KM", "KH"],  # Eddy viscosity from PBL scheme
        "needs_3d": True,
    },
    "continuity": {
        "desc": "div(u) = 0 (3D divergence)",
        "needs": ["U", "V", "W"],
        "needs_3d": True,
    },
}

# WRF variable name aliases (different WRF versions / PBL schemes may use different names)
VAR_ALIASES = {
    "U": ["U", "UU"],
    "V": ["V", "VV"],
    "W": ["W", "WW"],
    "P": ["P"],
    "PB": ["PB"],
    "T": ["T", "THETA"],
    "PH": ["PH", "PHB"],
    "PHB": ["PHB"],
    "TKE": ["TKE", "TKE_MYJ", "TKE_PBL"],
    "KM": ["KM", "EXCH_M", "EXCH_H"],
    "KH": ["KH", "EXCH_H"],
    "XLAT": ["XLAT", "XLAT_M"],
    "XLONG": ["XLONG", "XLONG_M"],
    "HGT": ["HGT"],
    "PSFC": ["PSFC"],
    "PBLH": ["PBLH"],
    "Q2": ["Q2", "Q2MV", "Q2M"],
    "T2": ["T2", "T2MV", "T2M"],
    "U10": ["U10", "U10M"],
    "V10": ["V10", "V10M"],
    "SWDOWN": ["SWDOWN"],
    "GLW": ["GLW"],
    "HFX": ["HFX"],
    "LH": ["LH"],
    "UST": ["UST"],
    "QFX": ["QFX"],
    "RAINC": ["RAINC"],
    "RAINNC": ["RAINNC"],
}


def find_variable(nc, target):
    """Find a variable in netCDF file by checking aliases."""
    for alias in VAR_ALIASES.get(target, [target]):
        if alias in nc.variables:
            return alias
    return None


def inspect_wrfout(file_path):
    """Inspect a single WRF output file."""
    line = "=" * 70
    print("")
    print(line)
    print("File: {}".format(file_path))
    print(line)

    nc = Dataset(file_path, "r")
    n_variables = len(nc.variables)

    # 1. Global attributes
    print("")
    print("--- Global Attributes ---")
    for attr in ["TITLE", "START_DATE", "SIMULATION_START_DATE", "GRIDTYPE",
                  "DX", "DY", "DT", "WEST-EAST_GRID_DIMENSION",
                  "SOUTH-NORTH_GRID_DIMENSION", "BOTTOM-TOP_GRID_DIMENSION"]:
        if attr in nc.ncattrs():
            print("  {}: {}".format(attr, getattr(nc, attr)))

    # 2. Dimensions
    print("")
    print("--- Dimensions ---")
    for dim_name, dim in nc.dimensions.items():
        print("  {}: {}".format(dim_name, len(dim)))

    # 3. All variables with dimensions and shape
    print("")
    print("--- All Variables ({} total) ---".format(n_variables))
    for var_name, var in nc.variables.items():
        dims = var.dimensions
        shape = var.shape
        units = getattr(var, "units", "")
        desc = getattr(var, "description", "")
        print("  {:<20s} dims={:<40s} shape={:<30s} units={}".format(
            var_name, str(dims), str(shape), units))

    # 4. Check what's available for NS equation
    print("")
    print("--- NS Equation Variable Check ---")
    found = {}
    missing = []
    for var_name in VAR_ALIASES:
        alias = find_variable(nc, var_name)
        if alias:
            var = nc.variables[alias]
            found[var_name] = {
                "alias": alias,
                "dims": var.dimensions,
                "shape": var.shape,
                "ndim": len(var.dimensions),
            }
        else:
            missing.append(var_name)

    print("")
    print("  FOUND:")
    for k, v in found.items():
        print("    {:<12s} -> {:<10s}  shape={}".format(k, v["alias"], v["shape"]))

    print("")
    print("  MISSING: {}".format(missing))

    # 5. Check time dimension
    has_time = "Time" in nc.dimensions
    n_times = len(nc.dimensions["Time"]) if has_time else 1
    print("")
    print("--- Time Dimension ---")
    print("  Time steps in file: {}".format(n_times))
    if has_time and n_times > 1:
        times = nc.variables["Times"]
        if len(times.shape) == 2:
            first = "".join([b.decode() if isinstance(b, bytes) else b
                             for b in times[0]])
            last = "".join([b.decode() if isinstance(b, bytes) else b
                            for b in times[-1]])
            print("  First: {}, Last: {}".format(first, last))
            dt_minutes = None
            try:
                from datetime import datetime
                t0 = datetime.strptime(first, "%Y-%m-%d_%H:%M:%S")
                t1 = datetime.strptime(last, "%Y-%m-%d_%H:%M:%S")
                dt_minutes = (t1 - t0).total_seconds() / 60.0 / (n_times - 1)
                print("  Output interval: ~{:.1f} min".format(dt_minutes))
            except Exception:
                pass

    # 6. Check vertical levels
    has_bottom_top = "bottom_top" in nc.dimensions
    n_levels = len(nc.dimensions["bottom_top"]) if has_bottom_top else 0
    print("")
    print("--- Vertical Levels ---")
    print("  bottom_top levels: {}".format(n_levels))

    # 7. Check 3D variables (those with bottom_top dimension)
    print("")
    print("--- 3D Variables (have bottom_top dimension) ---")
    vars_3d = []
    for var_name, var in nc.variables.items():
        if "bottom_top" in var.dimensions:
            vars_3d.append(var_name)
    print("  Count: {}".format(len(vars_3d)))
    for v in sorted(vars_3d):
        print("    {}".format(v))

    # 8. Summary: what NS terms can be computed
    print("")
    print(line)
    print("NS EQUATION FEASIBILITY ASSESSMENT")
    print(line)

    u_ok = "U" in found and found["U"]["ndim"] >= 3
    v_ok = "V" in found and found["V"]["ndim"] >= 3
    w_ok = "W" in found and found["W"]["ndim"] >= 3
    p_ok = "P" in found and found["P"]["ndim"] >= 3
    pb_ok = "PB" in found and found["PB"]["ndim"] >= 3
    t_ok = "T" in found and found["T"]["ndim"] >= 3
    tke_ok = "TKE" in found
    km_ok = "KM" in found
    kh_ok = "KH" in found
    geo_ok = "XLAT" in found and "XLONG" in found
    ph_ok = "PH" in found  # perturbation geopotential
    phb_ok = "PHB" in found  # base geopotential

    wind_3d = u_ok and v_ok and w_ok
    pressure_3d = p_ok and pb_ok
    density = p_ok and pb_ok and t_ok  # rho = (P+PB)/(R_d*T)
    eddy_visc = tke_ok or km_ok

    print("")
    print("  {:<30s} {:<30s} {:<10s}".format("Term", "Required", "Available"))
    print("  " + "-" * 70)

    terms_status = {}
    for term_name, req in NS_REQUIRED.items():
        if term_name == "time_derivative":
            av = wind_3d and has_time and n_times > 1
        elif term_name == "advection":
            av = wind_3d
        elif term_name == "pressure_gradient":
            av = pressure_3d
        elif term_name == "coriolis":
            av = wind_3d and geo_ok
        elif term_name == "turbulent_viscosity":
            av = eddy_visc
        elif term_name == "continuity":
            av = wind_3d
        else:
            av = False

        status = "YES" if av else "MISSING"
        terms_status[term_name] = av
        print("  {:<30s} {:<30s} {:<10s}".format(
            term_name, req["desc"], status))

    n_available = sum(terms_status.values())
    n_total = len(terms_status)
    print("")
    print("  >> {}/{} NS equation terms can be constrained".format(
        n_available, n_total))

    # Extra: geopotential height for computing physical heights of levels
    if ph_ok or phb_ok:
        print("  >> PH/PHB present: geopotential height available for level mapping")

    nc.close()

    return {
        "file": file_path,
        "n_variables": n_variables,
        "found": dict(found),
        "missing": missing,
        "has_time": has_time,
        "n_times": n_times,
        "n_levels": n_levels,
        "terms_status": terms_status,
        "n_terms_available": n_available,
    }


def scan_directory(data_dir, domain="d04", max_files=5):
    """Scan WRF output directory for files."""
    # WRF output naming: wrfout_d04_YYYY-MM-DD_HH:MM:SS
    wrf_files = []
    prefix = "wrfout_" + domain
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if prefix in f:
                wrf_files.append(os.path.join(root, f))
    wrf_files.sort()

    print("Found {} WRF output files for {}".format(len(wrf_files), domain))

    if len(wrf_files) == 0:
        print("")
        print("Searching for any wrfout files...")
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if "wrfout" in f:
                    wrf_files.append(os.path.join(root, f))
        wrf_files.sort()
        print("Found {} wrfout files total".format(len(wrf_files)))

    # Inspect a few sample files
    results = []
    sample_indices = [0]
    if len(wrf_files) > 1:
        sample_indices.append(len(wrf_files) // 2)
        sample_indices.append(min(len(wrf_files) - 1, max_files - 1))

    for idx in sample_indices:
        if idx < len(wrf_files):
            results.append(inspect_wrfout(wrf_files[idx]))

    return wrf_files, results


def main():
    parser = argparse.ArgumentParser(
        description="Inspect WRF output for NS equation feasibility"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to WRF output directory (e.g., /fsb/home/yutingwang/share/Data_WRFout/case01_Shenzhen/meso_202007_myj)"
    )
    parser.add_argument(
        "--domain", type=str, default="d04",
        help="WRF domain to check (default: d04, the 1km domain)"
    )
    parser.add_argument(
        "--max_files", type=int, default=3,
        help="Max sample files to inspect"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print("ERROR: Directory not found: {}".format(args.data_dir))
        sys.exit(1)

    wrf_files, results = scan_directory(args.data_dir, args.domain, args.max_files)

    if len(wrf_files) == 0:
        print("")
        print("No WRF output files found. Check the path and domain.")
        print("Example usage:")
        print("  python scripts/inspect_wrfout.py \\")
        print("    --data_dir /fsb/home/yutingwang/share/Data_WRFout/case01_Shenzhen/meso_202007_myj \\")
        print("    --domain d04")
        sys.exit(1)

    # Print final summary
    print("")
    line = "=" * 70
    print(line)
    print("FINAL SUMMARY")
    print(line)
    print("Total wrfout files: {}".format(len(wrf_files)))

    if results:
        r = results[0]
        print("")
        print("File inspected: {}".format(os.path.basename(r["file"])))
        print("Time steps: {}".format(r["n_times"]))
        print("Vertical levels: {}".format(r["n_levels"]))

        print("")
        print("Key findings for NS equation:")
        for term in ["time_derivative", "advection", "pressure_gradient",
                      "coriolis", "turbulent_viscosity", "continuity"]:
            status = "YES" if r["terms_status"].get(term) else "MISSING"
            print("  {:<25s}: {}".format(term, status))

        if r["n_terms_available"] == 6:
            print("")
            print("  *** ALL 6 NS equation terms can be constrained! ***")
        elif r["n_terms_available"] >= 4:
            print("")
            print("  *** {}/6 terms available - major improvement over current data ***".format(
                r["n_terms_available"]))
        else:
            print("")
            print("  *** Only {}/6 terms available - limited improvement ***".format(
                r["n_terms_available"]))

    print("")
    print("To check a different domain, use --domain d03")
    print("To check LES data, use --domain les_100m")


if __name__ == "__main__":
    main()
