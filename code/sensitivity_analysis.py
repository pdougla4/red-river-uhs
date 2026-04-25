#!/usr/bin/env python
"""
sensitivity_analysis.py
=======================
Comprehensive Morris + Sobol sensitivity analysis for Red River UHS.

Runs FOUR analyses:
  - overall   : joint_margin (min of all three mechanisms)
                Parameter count: 35  (caprock + reservoir + PGN + shared/transient)
  - pgn       : PGN_margin_caprock only
                Parameter count: 15  (PGN capillary + shared/transient)
  - aig       : AIG_margin only
                Parameter count: 16  (caprock + shared/transient)
  - rmg       : RMG_margin only
                Parameter count: 16  (reservoir + shared/transient)

Each analysis per (q, ΔT) point:
  1. Morris screening  → identify top-N drivers (reduces Sobol dimensionality)
  2. Sobol S1 + ST     → quantify first-order and total-order variance

  ST aggregation note: parameters are zero-imputed at points where Morris did
  not select them for Sobol. This prevents selection bias from inflating ST for
  parameters that only appear in favourable corners of the operating space.

Charts produced:
  - ST bar chart (total-order indices, mean +/- std across space)
  - S1 vs ST scatter (interaction detector)
  - ST heatmap across (q, DT) space, one figure per analysis

Convention: DT is signed (negative = cooling). Consistent with envelope workflow.

Usage
-----
  python sensitivity_analysis.py [fast|standard|thorough] [--workers N]

  fast      :  512 Sobol samples,  8 Morris trajectories, 200 pts/region
  standard  : 1024 Sobol samples, 10 Morris trajectories, 400 pts/region
  thorough  : 2048 Sobol samples, 15 Morris trajectories, 800 pts/region

  --workers N  : override number of parallel workers (default: os.cpu_count() - 1)
"""

import sys
import os
import json
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    from SALib.sample import sobol as sobol_sample
    from SALib.analyze import sobol as sobol_analyze
    HAS_SALIB = True
except ImportError:
    print("ERROR: SALib required — pip install SALib")
    HAS_SALIB = False

try:
    from scipy.stats import qmc
    from scipy.interpolate import griddata
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from uq_runner import morris_fixed as morris
from kernels import (
    AIG_margin,
    RMG_margin,
    PGN_margin_caprock,
    Bg_rm3_per_Sm3,
)
from warren_root_pressure import compute_transient_multipliers

N_WORKERS = os.cpu_count()

# =============================================================================
# MODE SELECTION & CLI ARGS
# =============================================================================

# Parse args: positional mode + optional --workers N
_args = sys.argv[1:]
_workers_override = None
if "--workers" in _args:
    idx = _args.index("--workers")
    try:
        _workers_override = int(_args[idx + 1])
        _args = [a for i, a in enumerate(_args) if i not in (idx, idx + 1)]
    except (IndexError, ValueError):
        print("ERROR: --workers requires an integer argument")
        sys.exit(1)

MODE = _args[0].lower() if _args else "standard"

# Worker count: CLI override > default of cpu_count-1 (leave one core free)
N_WORKERS = _workers_override if _workers_override else max(1, os.cpu_count() - 1)

CONFIGS = {
    "fast":     {"sobol_N": 512,  "morris_r": 8,  "pts_per_region": 200, "top_n": 8},
    "standard": {"sobol_N": 1024, "morris_r": 10, "pts_per_region": 400, "top_n": 10},
    "thorough": {"sobol_N": 2048, "morris_r": 15, "pts_per_region": 800, "top_n": 12},
    "overnight": {"sobol_N": 4096, "morris_r": 20, "pts_per_region": 1600, "top_n": 12},    
}
if MODE not in CONFIGS:
    print(f"Unknown mode '{MODE}'. Choose: fast | standard | thorough | overnight")
    sys.exit(1)

CFG = CONFIGS[MODE]
OUTPUT_PREFIX = f"sa_{MODE}_"

print("=" * 70)
print(f"SENSITIVITY ANALYSIS — {MODE.upper()} MODE")
print(f"  Sobol N={CFG['sobol_N']}  Morris r={CFG['morris_r']}  {CFG['pts_per_region']} pts/region")
print(f"  Workers: {N_WORKERS}")
print("=" * 70)

# =============================================================================
# PARAMETER SETS — full set for overall, reduced per binder
# NOTE on PGN: uses log10_pore_throat_radius_caprock [-8.3, -7.0] (anhydrite seal)
#              NOT log10_pore_throat_radius_reservoir (carbonate)
# =============================================================================

priors = json.loads(Path("priors.json").read_text())

def _rng(section, key):
    return list(priors[section][key])

# Shared well/flow params — appear in all analyses
SHARED_PARAMS = {
    "PI_m3day_per_MPa":  _rng("well", "PI_m3day_per_MPa"),
    "skin":              _rng("well", "skin"),
    "P_res_MPa":         _rng("well", "P_res_MPa"),
    "T_res_C":           _rng("well", "T_res_C"),
    "omega_storativity": _rng("warren_root_transient", "omega_storativity"),
    "tau_exchange_days": _rng("warren_root_transient", "tau_exchange_days"),
    "f_plume":           _rng("well", "plume_injectivity_factor"),
}

PARAMS = {
    # ── Overall: all 35 active parameters (caprock + reservoir + PGN + shared/transient) ──
    "overall": {
        # Caprock
        "cap_E_GPa":          _rng("caprock_anhydrite", "E_GPa"),
        "cap_nu":             _rng("caprock_anhydrite", "nu"),
        "cap_alpha_T_perK":   _rng("caprock_anhydrite", "alpha_T_perK"),
        "cap_UCS_MPa":        _rng("caprock_anhydrite", "UCS_MPa"),
        "cap_alpha_m2s":      _rng("caprock_anhydrite", "alpha_m2s"),
        "cap_chi":            _rng("caprock_anhydrite", "chi_confinement"),
        "cap_p_trans":        _rng("caprock_anhydrite", "pressure_transmission_frac"),
        "cap_dE_dT":          _rng("temperature_coupling", "dE_dT_GPa_per_C_caprock"),
        "cap_embrittle":      _rng("temperature_coupling", "embrittlement_factor_per_10C_caprock"),
        "cap_standoff_h_m":   _rng("well", "cap_thickness_m"),
        "cap_tau_cool":       _rng("caprock_anhydrite", "tau_cool_days"),
        "cap_tau_warm":       _rng("caprock_anhydrite", "tau_warm_days"),
        "cap_H2_stiff":       _rng("caprock_anhydrite", "H2_stiffening_frac"),
        "cap_H2_weaken":      _rng("caprock_anhydrite", "H2_strength_reduction_frac"),
        # Reservoir
        "res_E_GPa":          _rng("reservoir_carbonate", "E_GPa"),
        "res_nu":             _rng("reservoir_carbonate", "nu"),
        "res_alpha_T_perK":   _rng("reservoir_carbonate", "alpha_T_perK"),
        "res_T_tensile_MPa":  _rng("reservoir_carbonate", "T_tensile_MPa"),
        "res_T_tensile_dmg":  _rng("reservoir_carbonate", "T_tensile_damaged_MPa"),
        "res_chi":            _rng("reservoir_carbonate", "chi_confinement"),
        "res_dE_dT":          _rng("temperature_coupling", "dE_dT_GPa_per_C_reservoir"),
        "res_embrittle":      _rng("temperature_coupling", "embrittlement_factor_per_10C_reservoir"),
        "res_damage_prob":    [0.0, 1.0],
        "res_tau_cool":       _rng("reservoir_carbonate", "tau_cool_days"),
        "res_tau_warm":       _rng("reservoir_carbonate", "tau_warm_days"),
        "res_H2_stiff":       _rng("reservoir_carbonate", "H2_stiffening_frac"),
        "res_H2_weaken":      _rng("reservoir_carbonate", "H2_strength_reduction_frac"),
        # PGN
        "pgn_IFT":            _rng("PGN_capillary", "IFT_mN_per_m"),
        "pgn_IFT_dT":         _rng("PGN_capillary", "IFT_scale_dT_perC"),
        "pgn_IFT_dSal":       _rng("PGN_capillary", "IFT_scale_dSal_perM"),
        "pgn_theta":          _rng("PGN_capillary", "contact_angle_deg"),
        "pgn_dtheta_H2":      _rng("PGN_capillary", "dtheta_H2_deg"),
        "pgn_salinity":       _rng("PGN_capillary", "salinity_M"),
        "pgn_log10r":         _rng("PGN_capillary", "log10_pore_throat_radius_caprock"),
        "pgn_vug":            _rng("PGN_capillary", "vug_weakening_factor"),
        "pgn_frac":           _rng("PGN_capillary", "fracture_weakening_factor"),
        **SHARED_PARAMS,
    },

    # ── PGN: capillary seal params + shared ──────────────────────────────────
    "pgn": {
        "pgn_IFT":            _rng("PGN_capillary", "IFT_mN_per_m"),
        "pgn_IFT_dT":         _rng("PGN_capillary", "IFT_scale_dT_perC"),
        "pgn_IFT_dSal":       _rng("PGN_capillary", "IFT_scale_dSal_perM"),
        "pgn_theta":          _rng("PGN_capillary", "contact_angle_deg"),
        "pgn_dtheta_H2":      _rng("PGN_capillary", "dtheta_H2_deg"),
        "pgn_salinity":       _rng("PGN_capillary", "salinity_M"),
        "pgn_log10r":         _rng("PGN_capillary", "log10_pore_throat_radius_caprock"),
        "pgn_vug":            _rng("PGN_capillary", "vug_weakening_factor"),
        "pgn_frac":           _rng("PGN_capillary", "fracture_weakening_factor"),
        **SHARED_PARAMS,
    },

    # ── AIG: caprock params + shared ─────────────────────────────────────
    "aig": {
        "cap_E_GPa":          _rng("caprock_anhydrite", "E_GPa"),
        "cap_nu":             _rng("caprock_anhydrite", "nu"),
        "cap_alpha_T_perK":   _rng("caprock_anhydrite", "alpha_T_perK"),
        "cap_UCS_MPa":        _rng("caprock_anhydrite", "UCS_MPa"),
        "cap_alpha_m2s":      _rng("caprock_anhydrite", "alpha_m2s"),
        "cap_chi":            _rng("caprock_anhydrite", "chi_confinement"),
        "cap_p_trans":        _rng("caprock_anhydrite", "pressure_transmission_frac"),
        "cap_dE_dT":          _rng("temperature_coupling", "dE_dT_GPa_per_C_caprock"),
        "cap_embrittle":      _rng("temperature_coupling", "embrittlement_factor_per_10C_caprock"),
        "cap_standoff_h_m":   _rng("well", "cap_thickness_m"),
        "cap_tau_cool":       _rng("caprock_anhydrite", "tau_cool_days"),
        "cap_tau_warm":       _rng("caprock_anhydrite", "tau_warm_days"),
        "cap_H2_stiff":       _rng("caprock_anhydrite", "H2_stiffening_frac"),
        "cap_H2_weaken":      _rng("caprock_anhydrite", "H2_strength_reduction_frac"),
        **SHARED_PARAMS,
    },

    # ── RMG: reservoir params + shared ───────────────────────────────────
    "rmg": {
        "res_E_GPa":          _rng("reservoir_carbonate", "E_GPa"),
        "res_nu":             _rng("reservoir_carbonate", "nu"),
        "res_alpha_T_perK":   _rng("reservoir_carbonate", "alpha_T_perK"),
        "res_T_tensile_MPa":  _rng("reservoir_carbonate", "T_tensile_MPa"),
        "res_T_tensile_dmg":  _rng("reservoir_carbonate", "T_tensile_damaged_MPa"),
        "res_chi":            _rng("reservoir_carbonate", "chi_confinement"),
        "res_dE_dT":          _rng("temperature_coupling", "dE_dT_GPa_per_C_reservoir"),
        "res_embrittle":      _rng("temperature_coupling", "embrittlement_factor_per_10C_reservoir"),
        "res_damage_prob":    [0.0, 1.0],
        "res_tau_cool":       _rng("reservoir_carbonate", "tau_cool_days"),
        "res_tau_warm":       _rng("reservoir_carbonate", "tau_warm_days"),
        "res_H2_stiff":       _rng("reservoir_carbonate", "H2_stiffening_frac"),
        "res_H2_weaken":      _rng("reservoir_carbonate", "H2_strength_reduction_frac"),
        **SHARED_PARAMS,
    },
}

# Pre-compute name/bound lists for each analysis
PARAM_NAMES  = {k: list(v.keys())  for k, v in PARAMS.items()}
PARAM_BOUNDS = {k: list(v.values()) for k, v in PARAMS.items()}

# =============================================================================
# (q, ΔT) SAMPLING GRID
# =============================================================================

Q_MIN  = 14_158
Q_MAX  = 141_584
DT_MIN = -40.0
DT_MAX =  0.0

# Region labels are kept for batch parallelism but derived post-hoc from
# point location rather than pre-assigned. This eliminates the cluster gaps
# that caused interpolation banding in the heatmaps.
def _region_label(q, dT):
    q_tier  = "low" if q < 50000 else ("mid" if q < 100000 else "high")
    dT_tier = "mild" if dT > -10 else ("mod" if dT > -25 else "severe")
    return f"{q_tier}_q_{dT_tier}"

def generate_test_points():
    """Single uniform LHS over the full (q, DT) space -- no regional gaps."""
    # Same total count as the old 9-region scheme
    n_total = CFG["pts_per_region"] * 9
    if HAS_SCIPY:
        sampler = qmc.LatinHypercube(d=2, seed=42)
        s = sampler.random(n=n_total)
    else:
        rng = np.random.default_rng(42)
        s = rng.random((n_total, 2))

    qs  = Q_MIN  + s[:, 0] * (Q_MAX  - Q_MIN)
    dTs = DT_MIN + s[:, 1] * (DT_MAX - DT_MIN)

    return [
        {"q": float(q), "dT": float(dT), "region": _region_label(q, dT)}
        for q, dT in zip(qs, dTs)
    ]


# =============================================================================
# PARAMETER DICT BUILDERS
# =============================================================================

def _f_transient(omega, tau_days, t_eval=30.0, t_anchor=182.5):
    try:
        mults = compute_transient_multipliers(
            omega, tau_days, t_eval_days=[t_eval, t_anchor], t_anchor_days=t_anchor
        )
        return float(mults[t_eval])
    except Exception:
        return 1.0

def build_cap(p, f_trans):
    Bg = Bg_rm3_per_Sm3(p.get("P_res_MPa", 34.0), p.get("T_res_C", 110.0))
    return {
        "E_GPa":                    p.get("cap_E_GPa", 65.0),
        "nu":                       p.get("cap_nu", 0.25),
        "alpha_T_perK":             p.get("cap_alpha_T_perK", 3e-5),
        "UCS_MPa":                  p.get("cap_UCS_MPa", 45.0),
        "alpha_m2s":                p.get("cap_alpha_m2s", 3e-6),
        "chi_confinement":          p.get("cap_chi", 0.25),
        "pressure_transmission_frac": p.get("cap_p_trans", 0.15),
        "h_m":                      p.get("cap_standoff_h_m", 10.0),
        "exposure_days":            30.0,
        "FoS":                      1.81,
        "dE_dT_GPa_per_C":          p.get("cap_dE_dT", -0.025),
        "embrittlement_factor_per_10C": p.get("cap_embrittle", 0.065),
        "PI_m3day_per_MPa":         p.get("PI_m3day_per_MPa", 380.0),
        "skin":                     p.get("skin", 1.0),
        "Bg_rm3_per_Sm3":           Bg,
        "tau_cool_days":            p.get("cap_tau_cool", 0.0),
        "tau_warm_days":            p.get("cap_tau_warm", 0.0),
        "H2_stiffening_frac":       p.get("cap_H2_stiff", 0.0),
        "H2_strength_reduction_frac": p.get("cap_H2_weaken", 0.0),
    }, f_trans

def build_res(p, f_trans):
    Bg = Bg_rm3_per_Sm3(p.get("P_res_MPa", 34.0), p.get("T_res_C", 110.0))
    return {
        "E_GPa":                    p.get("res_E_GPa", 55.0),
        "nu":                       p.get("res_nu", 0.25),
        "alpha_T_perK":             p.get("res_alpha_T_perK", 9e-6),
        "chi_confinement":          p.get("res_chi", 0.25),
        "T_tensile_MPa":            p.get("res_T_tensile_MPa", 5.5),
        "T_tensile_damaged_MPa":    p.get("res_T_tensile_dmg", 3.0),
        "is_damaged_zone":          p.get("res_damage_prob", 0.5) < 0.2,
        "FoS":                      p.get("res_FoS", 1.5),
        "dE_dT_GPa_per_C":          p.get("res_dE_dT", -0.02),
        "embrittlement_factor_per_10C": p.get("res_embrittle", 0.05),
        "PI_m3day_per_MPa":         p.get("PI_m3day_per_MPa", 380.0),
        "skin":                     p.get("skin", 1.0),
        "Bg_rm3_per_Sm3":           Bg,
        "tau_cool_days":            p.get("res_tau_cool", 0.0),
        "tau_warm_days":            p.get("res_tau_warm", 0.0),
        "H2_stiffening_frac":       p.get("res_H2_stiff", 0.0),
        "H2_strength_reduction_frac": p.get("res_H2_weaken", 0.0),
    }, f_trans

def build_pgn(p):
    return {
        "IFT_mN_per_m":             p.get("pgn_IFT", 40.0),
        "IFT_scale_dT_perC":        p.get("pgn_IFT_dT", -0.2),
        "IFT_scale_dSal_perM":      p.get("pgn_IFT_dSal", 2.5),
        "contact_angle_deg":        p.get("pgn_theta", 40.0),
        "dtheta_H2_deg":            p.get("pgn_dtheta_H2", 15.0),
        "salinity_M":               p.get("pgn_salinity", 1.5),
        "log10_pore_throat_radius": p.get("pgn_log10r", -7.65),
        "vug_weakening_factor":     p.get("pgn_vug", 0.75),
        "fracture_weakening_factor": p.get("pgn_frac", 0.6),
    }

def eval_margin(analysis, q, dT, param_dict):
    """Evaluate the correct margin function for a given analysis."""
    omega = param_dict.get("omega_storativity", 0.2)
    tau   = param_dict.get("tau_exchange_days", 300.0)
    f     = _f_transient(omega, tau)

    if analysis == "pgn":
        pgn = build_pgn(param_dict)
        cap_well = {
            "PI_m3day_per_MPa": param_dict.get("PI_m3day_per_MPa", 380.0),
            "skin":             param_dict.get("skin", 1.0),
            "Bg_rm3_per_Sm3":   Bg_rm3_per_Sm3(
                param_dict.get("P_res_MPa", 34.0),
                param_dict.get("T_res_C", 110.0)),
        }
        return PGN_margin_caprock(q, dT, pgn, cap_well, f_transient=f)

    elif analysis == "aig":
        cap, f = build_cap(param_dict, f)
        return AIG_margin(q, dT, cap, f_transient=f)

    elif analysis == "rmg":
        res, f = build_res(param_dict, f)
        return RMG_margin(q, dT, res, f_transient=f)

    elif analysis == "overall":
        # Joint margin: min of all three
        pgn = build_pgn(param_dict)
        cap, _ = build_cap(param_dict, f)
        res, _ = build_res(param_dict, f)
        mp = PGN_margin_caprock(q, dT, pgn, cap, f_transient=f)
        mc = AIG_margin(q, dT, cap, f_transient=f)
        mr = RMG_margin(q, dT, res, f_transient=f)
        return min(mp, mc, mr)

    raise ValueError(f"Unknown analysis: {analysis}")

# =============================================================================
# MORRIS SCREENING (per analysis, per point)
# =============================================================================

def run_morris(analysis, q, dT):
    names  = PARAM_NAMES[analysis]
    bounds = PARAM_BOUNDS[analysis]
    D = len(names)

    def func(x_norm):
        pd = {names[i]: bounds[i][0] + x_norm[i] * (bounds[i][1] - bounds[i][0])
              for i in range(D)}
        try:
            return eval_margin(analysis, q, dT, pd)
        except Exception:
            return 0.0

    M = morris(func, [(0.0, 1.0)] * D, r=CFG["morris_r"], p=4)
    sorted_idx = np.argsort(M["mu_star"])[::-1]
    top_n = min(CFG["top_n"], D)
    return {
        "top_params":  [names[i] for i in sorted_idx[:top_n]],
        "top_mu_star": [float(M["mu_star"][i]) for i in sorted_idx[:top_n]],
        "all_mu_star": M["mu_star"].tolist(),
        "all_sigma":   M["sigma"].tolist(),
        "names":       names,
    }

# =============================================================================
# SOBOL ANALYSIS (per analysis, per point)
# =============================================================================

def run_sobol(analysis, q, dT, top_params):
    if not HAS_SALIB:
        return None

    names  = PARAM_NAMES[analysis]
    bounds = PARAM_BOUNDS[analysis]

    # Use top params from Morris; fix others at midpoint
    sub_names  = top_params
    sub_bounds = [bounds[names.index(p)] for p in sub_names]

    problem = {
        "num_vars": len(sub_names),
        "names":    sub_names,
        "bounds":   sub_bounds,
    }

    try:
        X = sobol_sample.sample(problem, CFG["sobol_N"], calc_second_order=False)
    except Exception:
        from SALib.sample import saltelli
        X = saltelli.sample(problem, CFG["sobol_N"], calc_second_order=False)

    Y = np.empty(X.shape[0])
    for i in range(X.shape[0]):
        # Build full param dict: sampled params + midpoints for rest
        pd = {n: 0.5 * (bounds[names.index(n)][0] + bounds[names.index(n)][1])
              for n in names}
        for j, pname in enumerate(sub_names):
            pd[pname] = X[i, j]
        try:
            Y[i] = eval_margin(analysis, q, dT, pd)
        except Exception:
            Y[i] = 0.0

    try:
        Si = sobol_analyze.analyze(problem, Y, calc_second_order=False,
                                   print_to_console=False)
        return {
            "S1":        Si["S1"].tolist(),
            "ST":        Si["ST"].tolist(),
            "S1_conf":   Si["S1_conf"].tolist(),
            "ST_conf":   Si["ST_conf"].tolist(),
            "names":     sub_names,
            "margin_mean": float(np.mean(Y)),
            "margin_std":  float(np.std(Y)),
            "p_fail":      float(np.mean(Y < 0)),
        }
    except Exception as e:
        print(f"WARNING: Sobol analysis failed at ({q:.0f}, {dT:.1f}): {e}")
        return None

# =============================================================================
# PARALLEL WORKER
# =============================================================================

def _worker(args):
    """Process one batch: all points in a (region, analysis) block."""
    analysis, region, pts_batch = args
    batch_results = []
    for pt in pts_batch:
        try:
            morris_r = run_morris(analysis, pt["q"], pt["dT"])
            sobol_r  = run_sobol(analysis, pt["q"], pt["dT"], morris_r["top_params"])
            batch_results.append({
                "q": pt["q"], "dT": pt["dT"], "region": region,
                "morris": morris_r, "sobol": sobol_r,
            })
        except Exception as e:
            # Log and skip — don't let one bad point kill the batch
            print(f"  WARNING: ({analysis}, q={pt['q']:.0f}, dT={pt['dT']:.1f}) failed: {e}")
            batch_results.append({
                "q": pt["q"], "dT": pt["dT"], "region": region,
                "morris": None, "sobol": None,
            })
    return analysis, region, batch_results

# =============================================================================
# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def run_all():
    t0 = time.time()
    analyses = list(PARAMS.keys())

    # Batch by region -- coarser granularity reduces future overhead vs one future/point
    from collections import defaultdict
    pts_by_region = defaultdict(list)
    for pt in generate_test_points():
        pts_by_region[pt["region"]].append(pt)

    tasks = [
        (analysis, region, pts)
        for analysis in analyses
        for region, pts in pts_by_region.items()
    ]
    n_batches = len(tasks)
    n_pts_total = sum(len(pts) for _, _, pts in tasks)

    print(f"\n{len(analyses)} analyses x {len(pts_by_region)} regions = {n_batches} batches")
    print(f"Total (q,dT) evaluations: {n_pts_total:,}")
    print(f"Workers: {N_WORKERS}\n")

    results = {a: [] for a in analyses}
    done = 0

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_worker, t): t for t in tasks}
        for fut in as_completed(futures):
            try:
                analysis, region, batch_res = fut.result()
                results[analysis].extend(batch_res)
            except Exception as e:
                task = futures[fut]
                print(f"  ERROR: batch ({task[0]}, {task[1]}) failed entirely: {e}")
            done += 1
            if done % max(1, n_batches // 10) == 0:
                elapsed = time.time() - t0
                pct = done / n_batches * 100
                eta = elapsed / pct * (100 - pct) if pct > 0 else 0
                print(f"  {pct:5.1f}%  elapsed {elapsed:.0f}s  ETA {eta:.0f}s  "
                      f"({done}/{n_batches} batches)")

    elapsed = time.time() - t0
    print(f"\nAll batches complete in {elapsed/60:.1f} min")

    fname_pkl = f"{OUTPUT_PREFIX}results.pkl"
    with open(fname_pkl, "wb") as f:
        pickle.dump(results, f)
    print(f"  -> {fname_pkl}")

    return results

# =============================================================================
# PLOTTING
# =============================================================================

ANALYSIS_LABELS = {
    "overall": "Overall (Joint Margin)",
    "pgn":     "PGN (Capillary Seal)",
    "aig": "AIG (Anhydrite Integrity)",
    "rmg": "RMG (Reservoir Mechanical)",
}
ANALYSIS_COLORS = {
    "overall": "#FFFFFF",
    "pgn":     "#2196F3",
    "aig":     "#FF9800",
    "rmg":     "#F44336",
}

# Per-parameter mechanism color for the overall bar chart.
# PGN = blue, AIG = orange, RMG = red, shared/transient = light gray.
# Matches binding map color convention exactly.
_PGN_COLOR    = "#2196F3"   # blue
_AIG_COLOR    = "#FF9800"   # orange
_RMG_COLOR    = "#F44336"   # red
_SHARED_COLOR = "#AAAAAA"   # gray — shared / transient params

PARAM_MECHANISM_COLOR = {
    # PGN — capillary seal
    "pgn_IFT":        _PGN_COLOR,
    "pgn_IFT_dT":     _PGN_COLOR,
    "pgn_IFT_dSal":   _PGN_COLOR,
    "pgn_theta":      _PGN_COLOR,
    "pgn_dtheta_H2":  _PGN_COLOR,
    "pgn_salinity":   _PGN_COLOR,
    "pgn_log10r":     _PGN_COLOR,
    "pgn_vug":        _PGN_COLOR,
    "pgn_frac":       _PGN_COLOR,
    # AIG — caprock anhydrite integrity
    "cap_E_GPa":      _AIG_COLOR,
    "cap_nu":         _AIG_COLOR,
    "cap_alpha_T_perK": _AIG_COLOR,
    "cap_UCS_MPa":    _AIG_COLOR,
    "cap_alpha_m2s":  _AIG_COLOR,
    "cap_chi":        _AIG_COLOR,
    "cap_p_trans":    _AIG_COLOR,
    "cap_dE_dT":      _AIG_COLOR,
    "cap_embrittle":  _AIG_COLOR,
    "cap_standoff_h_m": _AIG_COLOR,
    "cap_tau_cool":   _AIG_COLOR,
    "cap_tau_warm":   _AIG_COLOR,
    "cap_H2_stiff":   _AIG_COLOR,
    "cap_H2_weaken":  _AIG_COLOR,
    # RMG — reservoir mechanical
    "res_E_GPa":      _RMG_COLOR,
    "res_nu":         _RMG_COLOR,
    "res_alpha_T_perK": _RMG_COLOR,
    "res_T_tensile_MPa": _RMG_COLOR,
    "res_T_tensile_dmg": _RMG_COLOR,
    "res_chi":        _RMG_COLOR,
    "res_dE_dT":      _RMG_COLOR,
    "res_embrittle":  _RMG_COLOR,
    "res_damage_prob": _RMG_COLOR,
    "res_tau_cool":   _RMG_COLOR,
    "res_tau_warm":   _RMG_COLOR,
    "res_H2_stiff":   _RMG_COLOR,
    "res_H2_weaken":  _RMG_COLOR,
    # Shared / transient — appear in all analyses
    "PI_m3day_per_MPa":  _SHARED_COLOR,
    "skin":              _SHARED_COLOR,
    "P_res_MPa":         _SHARED_COLOR,
    "T_res_C":           _SHARED_COLOR,
    "omega_storativity": _SHARED_COLOR,
    "tau_exchange_days": _SHARED_COLOR,
    "f_plume":           _SHARED_COLOR,
    # Aliases used in short-name display (fallback)
    "PI_per":            _SHARED_COLOR,
    "P_res":             _SHARED_COLOR,
}

def _aggregate_ST(pts, analysis, top_n=10):
    """
    Aggregate ST indices across all points, with zero-imputation for unselected params.

    Parameters that Morris did not select at a given point are zero-imputed (ST=0)
    rather than omitted. This prevents upward bias for params that happen to look
    important only in corners of (q, dT) space where Morris selects them.

    Returns (names, means, stds) for top_n parameters by mean ST.
    """
    all_param_names = PARAM_NAMES[analysis]  # full list for this analysis
    # Accumulate ST across all points; default 0 for unselected params
    param_ST = {n: [] for n in all_param_names}

    for pt in pts:
        if pt["sobol"] is None:
            # Entire point failed -- impute zeros for all params
            for n in all_param_names:
                param_ST[n].append(0.0)
            continue
        selected = set(pt["sobol"]["names"])
        sobol_map = dict(zip(pt["sobol"]["names"],
                             [max(s, 0.0) for s in pt["sobol"]["ST"]]))
        for n in all_param_names:
            # Zero-impute params not selected by Morris at this point
            param_ST[n].append(sobol_map.get(n, 0.0))

    sorted_p = sorted(param_ST.items(),
                      key=lambda x: np.mean(x[1]), reverse=True)
    names = [p for p, _ in sorted_p[:top_n]]
    means = [np.mean(v) for _, v in sorted_p[:top_n]]
    return names, means, []

def plot_ST_bar(results, top_n=10):
    """ST bar chart — saves combined (1x4) and individual per-analysis figures."""

    def _draw_single(ax, analysis, top_n):
        ax.set_facecolor("#0d1b2a")
        names, means, _ = _aggregate_ST(results[analysis], analysis, top_n)
        if not names:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    color="white", transform=ax.transAxes)
            return
        short = [n.replace("_MPa","").replace("_GPa","").replace("_perK","")
                   .replace("_per_C","").replace("_m3day","").replace("storativity","ω")
                   .replace("exchange","τ") for n in names]
        y_pos = list(range(len(names)))

        if analysis == "overall":
            # Color each bar by its primary mechanism (matches binding map palette)
            bar_colors = [PARAM_MECHANISM_COLOR.get(n, _SHARED_COLOR) for n in names]
            bar_colors_rev = bar_colors[::-1]
        else:
            bar_colors_rev = [ANALYSIS_COLORS[analysis]] * len(names)

        ax.barh(y_pos, means[::-1],
                color=bar_colors_rev, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(short[::-1], color="white", fontsize=8)
        n_pts_valid = len([p for p in results[analysis] if p.get("sobol") is not None])
        ax.set_xlabel(f"ST (zero-imputed mean, n={n_pts_valid} pts)", color="white", fontsize=8)
        ax.set_title(ANALYSIS_LABELS[analysis], color="white", fontsize=10, fontweight="bold")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334")


    # ── Combined 1×4 figure ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    fig.patch.set_facecolor("#0d1b2a")
    fig.suptitle("Sobol Total-Order Indices (ST) — Mean Across Operating Space",
                 color="white", fontsize=13, fontweight="bold")
    for ax, analysis in zip(axes, ["overall", "pgn", "aig", "rmg"]):
        _draw_single(ax, analysis, top_n)
    plt.tight_layout()
    fname = f"{OUTPUT_PREFIX}ST_bars.png"
    plt.savefig(fname, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  → {fname}")

    # ── Individual figures ────────────────────────────────────────────────────
    from matplotlib.patches import Patch
    for analysis in ["overall", "pgn", "aig", "rmg"]:
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor("#0d1b2a")
        fig.suptitle(f"Sobol ST — {ANALYSIS_LABELS[analysis]}",
                     color="white", fontsize=12, fontweight="bold")
        _draw_single(ax, analysis, top_n)

        # Add mechanism legend to overall figure only
        if analysis == "overall":
            legend_handles = [
                Patch(facecolor=_PGN_COLOR,    label="PGN  (Capillary Seal)",     alpha=0.85),
                Patch(facecolor=_AIG_COLOR,    label="AIG  (Anhydrite Integrity)", alpha=0.85),
                Patch(facecolor=_RMG_COLOR,    label="RMG  (Reservoir Mech.)",     alpha=0.85),
                Patch(facecolor=_SHARED_COLOR, label="Shared / Transient",         alpha=0.85),
            ]
            ax.legend(handles=legend_handles, loc="lower right",
                      fontsize=7, framealpha=0.25,
                      labelcolor="white", facecolor="#0d1b2a",
                      edgecolor="#445566")

        plt.tight_layout()
        fname_ind = f"{OUTPUT_PREFIX}ST_bars_{analysis}.png"
        plt.savefig(fname_ind, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"  → {fname_ind}")

def plot_S1_ST_scatter(results):
    """S1 vs ST scatter — interaction detector. Points above diagonal have interactions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.patch.set_facecolor("#0d1b2a")
    fig.suptitle("S1 vs ST Scatter — Interaction Detector\n"
                 "Points above diagonal (ST >> S1) indicate parameter interactions",
                 color="white", fontsize=13, fontweight="bold")

    for ax, analysis in zip(axes.flat, ["overall", "pgn", "aig", "rmg"]):
        ax.set_facecolor("#0d1b2a")

        # Collect per-parameter S1 and ST with zero-imputation for unselected params
        all_names = PARAM_NAMES[analysis]
        param_data = {n: {"S1": [], "ST": []} for n in all_names}
        for pt in results[analysis]:
            if pt["sobol"] is None:
                for n in all_names:
                    param_data[n]["S1"].append(0.0)
                    param_data[n]["ST"].append(0.0)
                continue
            s1_map = dict(zip(pt["sobol"]["names"],
                              [max(s, 0.0) for s in pt["sobol"]["S1"]]))
            st_map = dict(zip(pt["sobol"]["names"],
                              [max(s, 0.0) for s in pt["sobol"]["ST"]]))
            for n in all_names:
                param_data[n]["S1"].append(s1_map.get(n, 0.0))
                param_data[n]["ST"].append(st_map.get(n, 0.0))

        if not param_data:
            continue

        # Plot each parameter as a point (mean S1 vs mean ST)
        colors = plt.cm.tab20(np.linspace(0, 1, len(param_data)))
        for (name, data), c in zip(param_data.items(), colors):
            s1_m = np.mean(data["S1"])
            st_m = np.mean(data["ST"])
            ax.scatter(s1_m, st_m, color=c, s=60, zorder=3)
            short = name.split("_")[0] + "_" + name.split("_")[-1] if "_" in name else name
            if st_m > 0.05:  # only label significant params
                ax.annotate(short, (s1_m, st_m),
                            textcoords="offset points", xytext=(4, 3),
                            color="white", fontsize=7, alpha=0.9)

        # Diagonal line: ST = S1 (no interactions)
        lim = max(ax.get_xlim()[1], ax.get_ylim()[1], 0.1)
        ax.plot([0, lim], [0, lim], color="#aaaaaa", lw=1, ls="--", alpha=0.6,
                label="ST = S1 (no interactions)")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("S1 (First-Order)", color="white", fontsize=10)
        ax.set_ylabel("ST (Total-Order)", color="white", fontsize=10)
        ax.set_title(ANALYSIS_LABELS[analysis], color="white", fontsize=11,
                     fontweight="bold")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334")
        ax.grid(True, alpha=0.15, color="white")

    plt.tight_layout()
    fname = f"{OUTPUT_PREFIX}S1_ST_scatter.png"
    plt.savefig(fname, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  → {fname}")

def plot_ST_heatmaps(results, top_n=6):
    """ST heatmap across (q, ΔT) for top parameters per analysis."""
    for analysis in ["overall", "pgn", "aig", "rmg"]:
        pts = results[analysis]
        sobol_pts = [p for p in pts if p["sobol"] is not None]
        if not sobol_pts:
            continue

        # Find global top parameters by mean ST
        names_top, _, _ = _aggregate_ST(pts, analysis, top_n)
        if not names_top:
            continue

        qs  = np.array([p["q"]  for p in sobol_pts]) / 1000  # k Sm³/day
        dTs = np.array([p["dT"] for p in sobol_pts])

        q_grid  = np.linspace(qs.min(),  qs.max(),  80)
        dT_grid = np.linspace(dTs.min(), dTs.max(), 80)
        QG, DG  = np.meshgrid(q_grid, dT_grid)

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.patch.set_facecolor("#0d1b2a")
        color = ANALYSIS_COLORS[analysis]
        fig.suptitle(f"ST Heatmap — {ANALYSIS_LABELS[analysis]}",
                     color="white", fontsize=13, fontweight="bold")

        for idx, (ax, pname) in enumerate(zip(axes.flat, names_top)):
            ax.set_facecolor("#0d1b2a")

            # Extract ST for this parameter at each point
            ST_vals = []
            for pt in sobol_pts:
                if pname in pt["sobol"]["names"]:
                    i = pt["sobol"]["names"].index(pname)
                    ST_vals.append(max(pt["sobol"]["ST"][i], 0.0))
                else:
                    ST_vals.append(0.0)

            ST_arr = np.array(ST_vals)
            if ST_arr.max() < 1e-6:
                ax.text(0.5, 0.5, "ST ≈ 0", ha="center", va="center",
                        color="white", transform=ax.transAxes)
                continue

            if not HAS_SCIPY:
                ax.text(0.5, 0.5, "scipy required for heatmap",
                        ha="center", va="center", color="white",
                        transform=ax.transAxes)
                continue
            grid_z = griddata(
                np.column_stack([qs, dTs]), ST_arr,
                (QG, DG), method="linear", fill_value=np.nan
            )

            cf = ax.contourf(QG, DG, grid_z, levels=15, cmap="YlOrRd")
            ax.scatter(qs, dTs, c="black", s=1, alpha=0.2)
            cb = plt.colorbar(cf, ax=ax, pad=0.02)
            cb.set_label("ST", color="white", fontsize=8)
            cb.ax.yaxis.set_tick_params(color="white")
            plt.setp(cb.ax.yaxis.get_ticklabels(), color="white", fontsize=7)

            short = pname.replace("_MPa","").replace("_GPa","").replace("_perK","")
            ax.set_title(short, color="white", fontsize=10, fontweight="bold")
            ax.set_xlabel("q (k Sm³/day)", color="white", fontsize=8)
            ax.set_ylabel("ΔT (°C)", color="white", fontsize=8)
            ax.tick_params(colors="white", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#334")

        plt.tight_layout()
        fname = f"{OUTPUT_PREFIX}ST_heatmap_{analysis}.png"
        plt.savefig(fname, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"  → {fname}")

def plot_all(results):
    print("\nGenerating charts...")
    plot_ST_bar(results)
    plot_S1_ST_scatter(results)
    plot_ST_heatmaps(results)

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    if not HAS_SALIB:
        print("Install SALib: pip install SALib")
        sys.exit(1)

    results = run_all()
    plot_all(results)

    print("\n" + "=" * 70)
    print(f"SENSITIVITY ANALYSIS COMPLETE — {MODE.upper()}")
    print("Files:")
    print(f"  {OUTPUT_PREFIX}results.pkl              ← raw results")
    print(f"  {OUTPUT_PREFIX}ST_bars.png              ← ST bar chart (all 4 combined)")
    print(f"  {OUTPUT_PREFIX}ST_bars_overall.png      ← ST bars — overall")
    print(f"  {OUTPUT_PREFIX}ST_bars_pgn.png          ← ST bars — PGN")
    print(f"  {OUTPUT_PREFIX}ST_bars_aig.png          ← ST bars — AIG")
    print(f"  {OUTPUT_PREFIX}ST_bars_rmg.png          ← ST bars — RMG")
    print(f"  {OUTPUT_PREFIX}S1_ST_scatter.png        ← interaction detector")
    print(f"  {OUTPUT_PREFIX}ST_heatmap_overall.png   ← heatmap — overall")
    print(f"  {OUTPUT_PREFIX}ST_heatmap_pgn.png       ← heatmap — PGN")
    print(f"  {OUTPUT_PREFIX}ST_heatmap_aig.png       ← heatmap — AIG")
    print(f"  {OUTPUT_PREFIX}ST_heatmap_rmg.png       ← heatmap — RMG")
    print("=" * 70)
