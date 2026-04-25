#!/usr/bin/env python
"""
generate_envelope_transient.py
===============================
Generates the P50 / P90 safe operating envelope for Red River UHS in
(q, ΔT) space using:
  - LHS ensemble (N parameter sets) evaluated at each (q, ΔT) grid point
  - All three mechanisms: PGN capillary, AIG (Anhydrite Integrity), RMGervoir
  - Warren-Root transient pressure multiplier
  - Adaptive grid refinement near the P90 boundary
  - No-False-Green policy: P90 is the operational boundary

Outputs
-------
  envelope_results.npz        Raw arrays (margins, mechanisms, P50/P90 surfaces)
  envelope_results.csv        Tabular q_max per ΔT bin at P50 and P90
  envelope_binding_map.png    Mechanism binding map (who limits each point)
  envelope_surfaces.png       P50 / P90 margin surfaces + boundary curves
  envelope_dashboard.png      Combined 4-panel operator dashboard

Usage
-----
  python generate_envelope_transient.py [fast|standard|thorough]

  fast      : N=100, coarse 11×15 grid  (~2 min)   — sanity check
  standard  : N=250, adaptive grid       (~10 min)  — RECOMMENDED
  thorough  : N=500, fine adaptive grid  (~30 min)  — presentation quality
"""

import sys
import json
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    from scipy.stats import qmc
    from scipy.special import erfc
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not found — using numpy random LHS fallback")

from kernels import (
    AIG_margin,
    RMG_margin,
    PGN_margin_caprock,
    CMG_stability,
    Bg_rm3_per_Sm3,
)
from warren_root_pressure import compute_transient_multipliers

# =============================================================================
# MODE SELECTION
# =============================================================================

MODE       = sys.argv[1].lower() if len(sys.argv) > 1 else "standard"
PRIOR_FILE = sys.argv[2]       if len(sys.argv) > 2 else "priors.json"

# Scenario tag for output filenames: "priors_single_well.json" -> "single_well"
_stem = Path(PRIOR_FILE).stem
SCENARIO = _stem[_stem.index("_")+1:] if "_" in _stem else "sector"

CONFIGS = {
    "fast":       {"N_lhs": 100,  "n_q": 20,  "n_dT": 20,  "tag": "fast"},
    "standard":   {"N_lhs": 250,  "n_q": 40,  "n_dT": 40,  "tag": "std"},
    "thorough":   {"N_lhs": 500,  "n_q": 60,  "n_dT": 60,  "tag": "thorough"},
    "thorough2":   {"N_lhs": 500,  "n_q": 100,  "n_dT": 100,  "tag": "thorough2"},
    "thorough3":   {"N_lhs": 1000,  "n_q": 100,  "n_dT": 100,  "tag": "thorough3"},    
    "production": {"N_lhs": 2000, "n_q": 100, "n_dT": 100, "tag": "production"},
    "extreme": {"N_lhs": 500, "n_q": 300, "n_dT": 300, "tag": "extreme"},
    "extreme2": {"N_lhs": 1000, "n_q": 300, "n_dT": 300, "tag": "extreme2"},
    "bonkers": {"N_lhs": 1000, "n_q": 500, "n_dT": 500, "tag": "bonkers"},
    "insane": {"N_lhs": 2000, "n_q": 500, "n_dT": 500, "tag": "insane"},
    "poster": {"N_lhs": 2000, "n_q": 1000, "n_dT": 1000, "tag": "poster"},   
}
if MODE not in CONFIGS:
    print(f"Unknown mode '{MODE}'. Choose: fast | standard | thorough | production | extreme | bonkers | insane")
    sys.exit(1)

CFG = CONFIGS[MODE]
N_WORKERS = os.cpu_count()   # use all physical cores — Ryzen 7700X = 8c/16t
if __name__ == "__main__":
    print("=" * 70)
    print(f"ENVELOPE GENERATION — {MODE.upper()} MODE")
    print(f"  N_lhs={CFG['N_lhs']}  grid={CFG['n_q']}×{CFG['n_dT']} = {CFG['n_q']*CFG['n_dT']} points")
    print(f"  Total kernel calls: {CFG['N_lhs'] * CFG['n_q'] * CFG['n_dT']:,}")
    print(f"  Workers: {N_WORKERS} (os.cpu_count)")
    print("=" * 70)

# =============================================================================
# OPERATING SPACE (Red River Red River)
# =============================================================================

Q_MIN   = 14_158    # Sm³/day  (≈ 0.5 MMscf/d)
Q_MAX   = 141_584  # Sm³/day  (≈ 5 MMscf/d)
DT_MIN  = -40.0     # °C  (severe cooling)
DT_MAX  =  0.0      # °C  (no cooling)

# Percentile targets
P_BOUNDARY = 90     # % — operational Green boundary (No-False-Green)
P_CONTEXT  = 50     # % — most-likely surface (shown as reference)

# =============================================================================
# PRIORS & PARAMETER SETUP
# =============================================================================

priors = json.loads(Path(PRIOR_FILE).read_text())

def _rng(section, key):
    return tuple(priors[section][key])

PARAM_SPACE = {
    # Caprock anhydrite
    "cap_E_GPa":              _rng("caprock_anhydrite", "E_GPa"),
    "cap_nu":                 _rng("caprock_anhydrite", "nu"),
    "cap_alpha_T_perK":       _rng("caprock_anhydrite", "alpha_T_perK"),
    "cap_UCS_MPa":            _rng("caprock_anhydrite", "UCS_MPa"),
    "cap_T_tensile_MPa":      _rng("caprock_anhydrite", "T_tensile_MPa"),
    "cap_alpha_m2s":          _rng("caprock_anhydrite", "alpha_m2s"),
    "cap_chi":                _rng("caprock_anhydrite", "chi_confinement"),
    "cap_p_trans":            _rng("caprock_anhydrite", "pressure_transmission_frac"),
    "cap_dE_dT":              _rng("temperature_coupling", "dE_dT_GPa_per_C_caprock"),
    "cap_embrittle":          _rng("temperature_coupling", "embrittlement_factor_per_10C_caprock"),
    "cap_H2_stiff":           _rng("caprock_anhydrite", "H2_stiffening_frac"),
    "cap_H2_weaken":          _rng("caprock_anhydrite", "H2_strength_reduction_frac"),
    # Reservoir carbonate
    "res_E_GPa":              _rng("reservoir_carbonate", "E_GPa"),
    "res_nu":                 _rng("reservoir_carbonate", "nu"),
    "res_alpha_T_perK":       _rng("reservoir_carbonate", "alpha_T_perK"),
    "res_T_tensile_MPa":      _rng("reservoir_carbonate", "T_tensile_MPa"),
    "res_T_tensile_dmg_MPa":  _rng("reservoir_carbonate", "T_tensile_damaged_MPa"),
    "res_chi":                _rng("reservoir_carbonate", "chi_confinement"),
    "res_dE_dT":              _rng("temperature_coupling", "dE_dT_GPa_per_C_reservoir"),
    "res_embrittle":          _rng("temperature_coupling", "embrittlement_factor_per_10C_reservoir"),
    "res_tau_cool":           _rng("reservoir_carbonate", "tau_cool_days"),
    "res_tau_warm":           _rng("reservoir_carbonate", "tau_warm_days"),
    "cap_tau_cool":           _rng("caprock_anhydrite",   "tau_cool_days"),
    "cap_tau_warm":           _rng("caprock_anhydrite",   "tau_warm_days"),
    "res_H2_stiff":           _rng("reservoir_carbonate", "H2_stiffening_frac"),
    "res_H2_weaken":          _rng("reservoir_carbonate", "H2_strength_reduction_frac"),
    # Well / flow
    "PI_m3day_per_MPa":       _rng("well", "PI_m3day_per_MPa"),
    "skin":                   _rng("well", "skin"),
    "f_plume":                _rng("well", "plume_injectivity_factor"),
    "P_res_MPa":              _rng("well", "P_res_MPa"),
    "T_res_C":                _rng("well", "T_res_C"),
    # PGN capillary — log10_r uses the CAPROCK key (anhydrite, 5–100 nm),
    # NOT log10_pore_throat_radius_reservoir (0.1–3 µm carbonate).
    "pgn_IFT":                _rng("PGN_capillary", "IFT_mN_per_m"),
    "pgn_IFT_dT":             _rng("PGN_capillary", "IFT_scale_dT_perC"),
    "pgn_IFT_dSal":           _rng("PGN_capillary", "IFT_scale_dSal_perM"),
    "pgn_theta":              _rng("PGN_capillary", "contact_angle_deg"),
    "pgn_dtheta_H2":          _rng("PGN_capillary", "dtheta_H2_deg"),
    "pgn_salinity":           _rng("PGN_capillary", "salinity_M"),
    "pgn_log10r":             _rng("PGN_capillary", "log10_pore_throat_radius_caprock"),
    "pgn_vug":                _rng("PGN_capillary", "vug_weakening_factor"),
    "pgn_frac":               _rng("PGN_capillary", "fracture_weakening_factor"),
    "pgn_bc_lambda":          _rng("PGN_capillary", "bc_lambda"),       # Brooks-Corey pore-size distribution index
    "pgn_Sw_irr":             _rng("PGN_capillary", "Sw_irr"),          # irreducible brine saturation
    "pgn_Sw_face":            _rng("PGN_capillary", "Sw_face"),         # brine saturation at caprock face
    # CMG stability — continuous H2-baseline-normalized function
    "cmg_k_md":       _rng("CMG_stability", "k_md"),
    "cmg_phi":        _rng("CMG_stability", "phi_frac"),
    "cmg_mu_H2":      _rng("CMG_stability", "mu_H2_cP"),
    "cmg_mu_brine":   _rng("CMG_stability", "mu_brine_cP"),
    "cmg_kr_H2":      _rng("CMG_stability", "kr_H2_end"),
    "cmg_kr_brine":   _rng("CMG_stability", "kr_brine_end"),
    "cmg_r_w":        _rng("CMG_stability", "r_w_m"),
    "cmg_h_perf":     _rng("CMG_stability", "h_perf_m"),
    "cmg_M_ref":      _rng("CMG_stability", "M_ref"),
    "cmg_w_M":        _rng("CMG_stability", "w_M"),
    "cmg_Ca_ref":     _rng("CMG_stability", "Ca_ref"),
    "cmg_w_Ca":       _rng("CMG_stability", "w_Ca"),
    "cmg_w_blend":    _rng("CMG_stability", "w_blend"),
    "cmg_f_min":      _rng("CMG_stability", "f_min"),
    # Warren-Root transient
    "omega":                  _rng("warren_root_transient", "omega_storativity"),
    "tau_exchange_days":      _rng("warren_root_transient", "tau_exchange_days"),
    # Stochastic damage flag (encoded as probability, threshold at draw)
    # Range [0.0, 1.0]: uniform draw; is_damaged = (draw < 0.25) → 25% damage probability
    "damage_prob":            (0.0, 1.0),
    # FoS — reservoir
    "FoS_res":                (1.5, 1.5),    
    # Standoff for caprock heat conduction
    "cap_thickness_m":         _rng("well", "cap_thickness_m"),   # caprock thickness for AIG erfc (1-5 m)
    "reservoir_standoff_m":    _rng("well", "reservoir_standoff_m"), # perf-to-caprock distance for PGN pressure attenuation (10-20 m)
}

PARAM_NAMES = list(PARAM_SPACE.keys())
N_PARAMS     = len(PARAM_NAMES)
BOUNDS_LO    = np.array([v[0] for v in PARAM_SPACE.values()])
BOUNDS_HI    = np.array([v[1] for v in PARAM_SPACE.values()])

# =============================================================================
# LHS SAMPLING
# =============================================================================

def sample_lhs(N, seed=42):
    """Return N×D LHS samples mapped to physical parameter space."""
    D = N_PARAMS
    if HAS_SCIPY:
        sampler = qmc.LatinHypercube(d=D, seed=seed)
        unit = sampler.random(n=N)
    else:
        rng = np.random.default_rng(seed)
        unit = np.zeros((N, D))
        for j in range(D):
            perm = rng.permutation(N)
            unit[:, j] = (perm + rng.random(N)) / N
    return BOUNDS_LO + unit * (BOUNDS_HI - BOUNDS_LO)

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

def build_params(row):
    """Unpack one LHS row into (cap, res, pgn, f_trans) dicts."""
    p = dict(zip(PARAM_NAMES, row))

    P_res = p["P_res_MPa"]
    T_res = p["T_res_C"]
    Bg    = Bg_rm3_per_Sm3(P_res, T_res)

    f_trans = _f_transient(p["omega"], p["tau_exchange_days"])

    cap = {
        "E_GPa":                    p["cap_E_GPa"],
        "nu":                       p["cap_nu"],
        "alpha_T_perK":             p["cap_alpha_T_perK"],
        "UCS_MPa":                  p["cap_UCS_MPa"],
        "T_tensile_MPa":            p["cap_T_tensile_MPa"],
        "alpha_m2s":                p["cap_alpha_m2s"],
        "chi_confinement":          p["cap_chi"],
        "pressure_transmission_frac": p["cap_p_trans"],
        "cap_thickness_m":          p["cap_thickness_m"],   # AIG erfc thermal standoff
        "reservoir_standoff_m":     p["reservoir_standoff_m"], # PGN pressure attenuation standoff
        "tau_cool_days":            p["cap_tau_cool"],          # AIG cycle thermal memory
        "tau_warm_days":            p["cap_tau_warm"],
        "exposure_days":            30.0,
        "FoS":                      1.75,          # Caprock FoS fixed conservative
        "dE_dT_GPa_per_C":          p["cap_dE_dT"],
        "embrittlement_factor_per_10C": p["cap_embrittle"],
        # Well params needed for pressure in caprock check
        "PI_m3day_per_MPa":         p["PI_m3day_per_MPa"],
        "skin":                     p["skin"],
        "f_plume":                  p["f_plume"],
        "Bg_rm3_per_Sm3":           Bg,
        "H2_stiffening_frac":          p["cap_H2_stiff"],
        "H2_strength_reduction_frac":  p["cap_H2_weaken"],
    }

    is_damaged = p["damage_prob"] < 0.20   # 20% probability of damaged zone
    res = {
        "E_GPa":                    p["res_E_GPa"],
        "nu":                       p["res_nu"],
        "alpha_T_perK":             p["res_alpha_T_perK"],
        "chi_confinement":          p["res_chi"],
        "T_tensile_MPa":            p["res_T_tensile_MPa"],
        "T_tensile_damaged_MPa":    p["res_T_tensile_dmg_MPa"],
        "is_damaged_zone":          is_damaged,
        "FoS":                      p["FoS_res"],
        "dE_dT_GPa_per_C":          p["res_dE_dT"],
        "embrittlement_factor_per_10C": p["res_embrittle"],
        "PI_m3day_per_MPa":         p["PI_m3day_per_MPa"],
        "skin":                     p["skin"],
        "f_plume":                  p["f_plume"],
        "Bg_rm3_per_Sm3":           Bg,
        "tau_cool_days":            p["res_tau_cool"],
        "tau_warm_days":            p["res_tau_warm"],
        "H2_stiffening_frac":          p["res_H2_stiff"],
        "H2_strength_reduction_frac":  p["res_H2_weaken"],
    }

    pgn = {
        "IFT_mN_per_m":             p["pgn_IFT"],
        "IFT_scale_dT_perC":        p["pgn_IFT_dT"],
        "IFT_scale_dSal_perM":      p["pgn_IFT_dSal"],
        "contact_angle_deg":        p["pgn_theta"],
        "dtheta_H2_deg":            p["pgn_dtheta_H2"],
        "salinity_M":               p["pgn_salinity"],
        "log10_pore_throat_radius": p["pgn_log10r"],
        "vug_weakening_factor":     p["pgn_vug"],
        "f_plume":                  p["f_plume"],
        "fracture_weakening_factor": p["pgn_frac"],
        "bc_lambda":                 p["pgn_bc_lambda"],
        "Sw_irr":                    p["pgn_Sw_irr"],
        "Sw_face":                   p["pgn_Sw_face"],
    }

    cmg = {
        "k_md":        p["cmg_k_md"],
        "phi_frac":    p["cmg_phi"],
        "mu_H2_cP":    p["cmg_mu_H2"],
        "mu_brine_cP": p["cmg_mu_brine"],
        "kr_H2_end":   p["cmg_kr_H2"],
        "kr_brine_end": p["cmg_kr_brine"],
        "r_w_m":       p["cmg_r_w"],
        "h_perf_m":    p["cmg_h_perf"],
        "M_ref":       p["cmg_M_ref"],
        "w_M":         p["cmg_w_M"],
        "Ca_ref":      p["cmg_Ca_ref"],
        "w_Ca":        p["cmg_w_Ca"],
        "w_blend":     p["cmg_w_blend"],
        "f_min":       p["cmg_f_min"],
    }

    return cap, res, pgn, cmg, f_trans

# =============================================================================
# SINGLE-POINT ENSEMBLE EVALUATION
# =============================================================================

MECH_LABELS = ["PGN", "AIG", "RMG"]
MECH_COLORS = ["#2196F3", "#FF9800", "#F44336"]   # blue, orange, red

def eval_ensemble_at_point(q, dT, lhs_rows):
    """
    Evaluate all N LHS realisations at one (q, ΔT) point.
    Core physics only — CMG derate applied separately as post-process.

    Returns
    -------
    joint_margins   : (N,) minimum of PGN, AIG, RMG
    pgn_margins     : (N,)
    aig_margins : (N,)
    rmg_margins : (N,)
    binding_mechs   : (N,) int  (0=PGN, 1=AIG, 2=RMG)
    f_stab_arr      : (N,) CMG derate factors (computed but NOT applied to margins)
    """
    N = len(lhs_rows)
    m_joint    = np.empty(N)
    m_pgn      = np.empty(N)
    m_cap      = np.empty(N)
    m_res      = np.empty(N)
    binding    = np.empty(N, dtype=int)
    f_stab_arr = np.empty(N)

    for i, row in enumerate(lhs_rows):
        cap, res, pgn, cmg, f_trans = build_params(row)

        mp = PGN_margin_caprock(q, dT, pgn, cap, f_transient=f_trans)
        mc = AIG_margin(q, dT, cap, f_transient=f_trans)
        mr = RMG_margin(q, dT, res, f_transient=f_trans)

        margins_i  = [mp, mc, mr]
        m_pgn[i]   = mp
        m_cap[i]   = mc
        m_res[i]   = mr
        m_joint[i] = min(margins_i)
        binding[i] = int(np.argmin(margins_i))

        # CMG computed here for the f_stab surface, but NOT fed back into margins
        f_stab_arr[i], _, _, _ = CMG_stability(q, cmg, pgn=pgn)

    return m_joint, m_pgn, m_cap, m_res, binding, f_stab_arr

# =============================================================================
# GRID GENERATION
# =============================================================================

def make_grid():
    q_arr  = np.linspace(Q_MIN, Q_MAX, CFG["n_q"])
    dT_arr = np.linspace(DT_MAX, DT_MIN, CFG["n_dT"])  # warmest → coldest
    return q_arr, dT_arr

# =============================================================================
# PARALLEL WORKER  (module-level — required for ProcessPoolExecutor pickling)
# =============================================================================

# Global shared in each worker process — set once by _worker_init, never pickled per-task.
_WORKER_LHS_ROWS = None

def _worker_init(lhs_rows):
    """Called once per worker process to cache lhs_rows in process memory."""
    global _WORKER_LHS_ROWS
    _WORKER_LHS_ROWS = lhs_rows

def _worker(args):
    """Evaluate one (q, dT) grid point across all LHS realisations.
    lhs_rows is NOT passed per-task — lives in _WORKER_LHS_ROWS set by initializer.
    """
    i, j, q, dT = args
    mj, mp, mc, mr, bnd, fs = eval_ensemble_at_point(q, dT, _WORKER_LHS_ROWS)
    return i, j, mj, mp, mc, mr, bnd, fs


# =============================================================================
# MAIN ENVELOPE LOOP
# =============================================================================

def run_envelope():
    t0 = time.time()

    # ── 1. Draw LHS ensemble ──────────────────────────────────────────────────
    print(f"\n[1/3] Drawing {CFG['N_lhs']} LHS realisations ({N_PARAMS} params)...")
    lhs_rows = sample_lhs(CFG["N_lhs"])

    # ── 2. Full uniform grid evaluation ──────────────────────────────────────
    q_all, dT_all = make_grid()
    nq, ndT = len(q_all), len(dT_all)
    N = CFG["N_lhs"]

    print(f"[2/3] Evaluating {nq}×{ndT} grid × {N} realisations = {nq*ndT*N:,} kernel calls...")

    # ── Allocate scratch arrays: RAM for small runs, memmap for large ────────
    # Threshold: if 6 arrays would exceed 4 GB total, use memmap (disk-backed).
    # fast/standard/thorough stay in RAM; bonkers/insane use memmap.
    import tempfile, shutil
    _scratch_gb = nq * ndT * N * 8 * 6 / 1e9
    USE_MMAP = _scratch_gb > 4.0
    _mmap_dir = None

    if USE_MMAP:
        _mmap_dir = tempfile.mkdtemp(prefix="uhs_envelope_")
        def _alloc(fname, dtype=np.float64):
            return np.memmap(os.path.join(_mmap_dir, fname), dtype=dtype,
                             mode='w+', shape=(nq, ndT, N))
        print(f"  Scratch {_scratch_gb:.1f} GB → memmap: {_mmap_dir}")
    else:
        def _alloc(fname, dtype=np.float64):
            return np.empty((nq, ndT, N), dtype=dtype)
        print(f"  Scratch {_scratch_gb:.2f} GB → RAM")

    try:
        arr_joint   = _alloc("joint.dat")
        arr_pgn     = _alloc("pgn.dat")
        arr_cap     = _alloc("cap.dat")
        arr_res     = _alloc("res.dat")
        arr_binding = _alloc("binding.dat", dtype=np.int32)
        arr_fstab   = _alloc("fstab.dat")

        total_pts = nq * ndT
        # lhs_rows NOT in task tuple — sent once per worker via initializer
        tasks = [(i, j, q_all[i], dT_all[j]) for i in range(nq) for j in range(ndT)]

        # Throttled submission — cap in-flight futures to avoid WinError 1450
        # (Windows exhausts kernel handles if all 250k futures submitted at once).
        # MAX_INFLIGHT >> N_WORKERS keeps all workers busy; << 250k keeps handles low.
        MAX_INFLIGHT = min(N_WORKERS * 64, 2048)

        done = 0
        with ProcessPoolExecutor(max_workers=N_WORKERS,
                                 initializer=_worker_init,
                                 initargs=(lhs_rows,)) as pool:
            from collections import deque
            pending = deque()
            task_iter = iter(tasks)

            # Seed the queue
            for t in task_iter:
                pending.append(pool.submit(_worker, t))
                if len(pending) >= MAX_INFLIGHT:
                    break

            while pending:
                fut = pending.popleft()
                i, j, mj, mp, mc, mr, bnd, fs = fut.result()
                arr_joint[i, j, :]   = mj
                arr_pgn[i, j, :]     = mp
                arr_cap[i, j, :]     = mc
                arr_res[i, j, :]     = mr
                arr_binding[i, j, :] = bnd
                arr_fstab[i, j, :]   = fs
                done += 1
                # Submit one more to replace the one just consumed
                t = next(task_iter, None)
                if t is not None:
                    pending.append(pool.submit(_worker, t))
                if done % max(1, total_pts // 20) == 0:
                    elapsed = time.time() - t0
                    pct = done / total_pts * 100
                    eta = elapsed / pct * (100 - pct) if pct > 0 else 0
                    print(f"  {pct:5.1f}%  elapsed {elapsed:.0f}s  ETA {eta:.0f}s")

        # ── 3. Compute percentile surfaces ────────────────────────────────────
        print("[3/3] Computing percentile surfaces and saving outputs...")
        p50_all = np.percentile(arr_joint, P_CONTEXT,       axis=2)
        p90_all = np.percentile(arr_joint, 100-P_BOUNDARY,  axis=2)
        p80_all = np.percentile(arr_joint, 20,              axis=2)
        p95_all = np.percentile(arr_joint,  5,              axis=2)
        pgn_all = np.percentile(arr_pgn,   P_CONTEXT,       axis=2)
        cap_all = np.percentile(arr_cap,   P_CONTEXT,       axis=2)
        res_all = np.percentile(arr_res,   P_CONTEXT,       axis=2)
        pgn_p90 = np.percentile(arr_pgn,   100-P_BOUNDARY,  axis=2)
        cap_p90 = np.percentile(arr_cap,   100-P_BOUNDARY,  axis=2)
        res_p90 = np.percentile(arr_res,   100-P_BOUNDARY,  axis=2)
        fstab_median = np.median(arr_fstab, axis=2)

        binding_all = np.zeros((nq, ndT), dtype=int)
        for i in range(nq):
            for j in range(ndT):
                counts = np.bincount(arr_binding[i, j, :], minlength=3)
                binding_all[i, j] = int(np.argmax(counts))

        # ── Boundary curves ───────────────────────────────────────────────────
        def extract_boundary_curve(margin_surface, q_grid, dT_grid):
            q_max = np.full(len(dT_grid), np.nan)
            for j in range(len(dT_grid)):
                col = margin_surface[:, j]
                safe_mask = col >= 0
                if np.any(safe_mask):
                    q_max[j] = q_grid[safe_mask].max()
            return q_max

        q_max_p50_base = extract_boundary_curve(p50_all, q_all, dT_all)
        q_max_p90_base = extract_boundary_curve(p90_all, q_all, dT_all)
        q_max_p80_base = extract_boundary_curve(p80_all, q_all, dT_all)
        q_max_p95_base = extract_boundary_curve(p95_all, q_all, dT_all)

        fstab_p10 = np.percentile(arr_fstab, 10, axis=2)
        fstab_med = np.median(arr_fstab,         axis=2)

        def fstab_at_boundary(q_max_arr, fstab_surface, q_grid, dT_grid):
            f_at_bnd = np.ones(len(dT_grid))
            for j in range(len(dT_grid)):
                if not np.isnan(q_max_arr[j]):
                    idx = np.argmin(np.abs(q_grid - q_max_arr[j]))
                    f_at_bnd[j] = fstab_surface[idx, j]
            return f_at_bnd

        f_at_p90_bnd = fstab_at_boundary(q_max_p90_base, fstab_p10, q_all, dT_all)
        q_max_p90_cmg = np.where(
            ~np.isnan(q_max_p90_base),
            q_max_p90_base * f_at_p90_bnd,
            np.nan
        )

        q_max_p50 = q_max_p50_base
        q_max_p90 = q_max_p90_base
        q_max_p80 = q_max_p80_base
        q_max_p95 = q_max_p95_base
        fstab_median = fstab_p10  # P10 drives operational boundary

        # ── 5. Save results ────────────────────────────────────────────────────
        print("[4/4] Saving outputs...")
        np.savez(
            f"envelope_results_{SCENARIO}.npz",
            q_grid=q_all, dT_grid=dT_all,
            p50_surface=p50_all, p90_surface=p90_all,
            p80_surface=p80_all, p95_surface=p95_all,
            binding_mode=binding_all,
            pgn_p50=pgn_all, aig_p50=cap_all, rmg_p50=res_all,
            pgn_p90=pgn_p90, aig_p90=cap_p90, rmg_p90=res_p90,
            q_max_p50=q_max_p50, q_max_p90=q_max_p90,
            q_max_p80=q_max_p80, q_max_p95=q_max_p95,
            q_max_p90_cmg=q_max_p90_cmg,
            cmg_fstab_median=fstab_median,
            cmg_fstab_p10=fstab_p10,
        )
        print("  → envelope_results.npz")

        # CSV table
        import csv
        with open(f"envelope_results_{SCENARIO}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["dT_C", "q_max_P50_Sm3day", "q_max_P90_Sm3day",
                             "q_max_P50_MMscfd", "q_max_P90_MMscfd"])
            for j, dT in enumerate(dT_all):
                q50 = q_max_p50[j] if not np.isnan(q_max_p50[j]) else -1
                q90 = q_max_p90[j] if not np.isnan(q_max_p90[j]) else -1
                def to_mscfd(x): return x * 0.0353147 / 1000 if x > 0 else -1
                writer.writerow([f"{dT:.1f}", f"{q50:.0f}", f"{q90:.0f}",
                                  f"{to_mscfd(q50):.3f}", f"{to_mscfd(q90):.3f}"])
        print("  → envelope_results.csv")

        elapsed = time.time() - t0
        print(f"\nTotal runtime: {elapsed/60:.1f} min")

        # Copy 2D arrays to plain numpy BEFORE memmap cleanup
        _p50_all     = np.array(p50_all);    _p90_all = np.array(p90_all)
        _p80_all     = np.array(p80_all);    _p95_all = np.array(p95_all)
        _binding_all = np.array(binding_all)
        _pgn_all     = np.array(pgn_all);    _cap_all = np.array(cap_all);    _res_all = np.array(res_all)
        _pgn_p90     = np.array(pgn_p90);    _cap_p90 = np.array(cap_p90);    _res_p90 = np.array(res_p90)
        _fstab_med   = np.array(fstab_median)

    finally:
        del arr_joint, arr_pgn, arr_cap, arr_res, arr_binding, arr_fstab
        if _mmap_dir is not None:
            shutil.rmtree(_mmap_dir, ignore_errors=True)
            print(f"  Memmap scratch deleted: {_mmap_dir}")

    return (q_all, dT_all,
            _p50_all, _p90_all, _p80_all, _p95_all,
            _binding_all,
            _pgn_all, _cap_all, _res_all,
            _pgn_p90, _cap_p90, _res_p90,
            q_max_p50, q_max_p90, q_max_p80, q_max_p95,
            q_max_p90_cmg, _fstab_med)

# =============================================================================
# PLOTTING
# =============================================================================

def plot_all(q_all, dT_all, p50_all, p90_all, p80_all, p95_all, binding_all,
             pgn_all, cap_all, res_all, pgn_p90, cap_p90, res_p90,
             q_max_p50, q_max_p90, q_max_p80, q_max_p95,
             q_max_p90_cmg, fstab_median):

    BG      = "#1a1a2e"
    BG2     = "#0d1b2a"
    Q_km    = q_all / 1000.0
    QQ, DD  = np.meshgrid(Q_km, dT_all, indexing="ij")

    def _save(fname):
        p = Path(fname)
        tagged = p.stem + f"_{SCENARIO}" + p.suffix
        plt.tight_layout()
        plt.savefig(tagged, dpi=200, bbox_inches="tight",
                    facecolor=plt.gcf().get_facecolor())
        plt.close()
        print(f"  → {tagged}")

    def _style_ax(ax, bg=BG2):
        ax.set_facecolor(bg)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#555")

    # ── Figure 1: Binding Mechanism Map + Ca-M-G SOE ─────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(BG)
    _style_ax(ax, BG)

    cmap_mech = mcolors.ListedColormap(MECH_COLORS)
    ax.pcolormesh(QQ, DD, binding_all, cmap=cmap_mech,
                  vmin=-0.5, vmax=2.5, shading="auto")
    # P90 physics boundary
    ax.contour(QQ, DD, p90_all, levels=[0], colors=["white"],
               linewidths=2.5, linestyles="--")
    # Ca-M-G operational boundary + green shading
    valid = ~np.isnan(q_max_p90_cmg)
    if valid.any():
        ax.plot(q_max_p90_cmg[valid] / 1000, dT_all[valid],
                color="yellow", lw=2.5, ls="--", zorder=5)
        ax.fill_betweenx(dT_all[valid],
                         Q_MIN / 1000,  # start at grid minimum, not zero
                         q_max_p90_cmg[valid] / 1000,
                         alpha=0.12, color="yellow", zorder=4)
    ax.set_xlim(Q_MIN / 1000, Q_MAX / 1000)
    ax.set_ylim(DT_MIN, DT_MAX)
    ax.set_xlabel("Injection Rate q (k Sm³/day)", color="white", fontsize=12)
    ax.set_ylabel("Temperature Differential ΔT (°C)", color="white", fontsize=12)
    ax.set_title("Binding Mechanism Map — Red River UHS\n"
                 f"N={CFG['N_lhs']} LHS  |  P90 boundary (—)  |  Yellow = Ca-M-G operational limit",
                 color="white", fontsize=13)
    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(MECH_COLORS, MECH_LABELS)]
    patches += [mpatches.Patch(color="white", label="P90 physics bnd"),
                mpatches.Patch(color="yellow", label="P90 + Ca-M-G derate (ops)")]
    ax.legend(handles=patches, loc="upper right", facecolor="#2a2a4a",
              labelcolor="white", fontsize=10)
    _save("envelope_binding_map.png")

    # ── Figure 2: q_max vs ΔT — operational rate limit ───────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(BG)
    _style_ax(ax, BG)

    ax.plot(q_max_p90 / 1000, dT_all, color="lime", lw=2.5, ls="-",
            label=f"P{P_BOUNDARY} physics boundary")
    ax.plot(q_max_p90_cmg / 1000, dT_all, color="yellow", lw=2, ls="--",
            label=f"P{P_BOUNDARY} + Ca-M-G derate (operational)")
    ax.fill_betweenx(dT_all, 0, q_max_p90_cmg / 1000,
                     alpha=0.15, color="yellow", label="Safe operating region")
    # Confidence band
    valid_band = ~(np.isnan(q_max_p80) | np.isnan(q_max_p95))
    ax.fill_betweenx(dT_all[valid_band],
                     q_max_p80[valid_band] / 1000,
                     q_max_p95[valid_band] / 1000,
                     alpha=0.15, color="lime", label="P80–P95 uncertainty band")
    ax.set_xlabel("q_max (k Sm³/day)", color="white", fontsize=12)
    ax.set_ylabel("ΔT (°C)", color="white", fontsize=12)
    ax.set_xlim(Q_MIN / 1000, Q_MAX / 1000 * 1.05); ax.set_ylim(DT_MIN, DT_MAX)
    ax.set_title("Max Allowable Injection Rate — Red River UHS\n"
                 "No-False-Green: P90 conservative boundary",
                 color="white", fontsize=13)
    ax.grid(True, alpha=0.15, color="white")
    ax.legend(fontsize=10, facecolor="#2a2a4a", labelcolor="white")
    _save("envelope_qmax_curve.png")

    # ── Figure 3: P90 Safety Margin surface ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(BG)
    _style_ax(ax, BG)

    vmax_d = max(abs(p90_all).max(), 0.5)
    cf = ax.contourf(QQ, DD, p90_all,
                     levels=np.linspace(-vmax_d, vmax_d, 40),
                     cmap="RdYlGn", extend="both")
    ax.contour(QQ, DD, p90_all, levels=[0], colors=["cyan"],
               linewidths=2.5, linestyles="--")
    pos_levels = [0.25, 0.5, 1.0, 1.5]
    cs = ax.contour(QQ, DD, p90_all, levels=pos_levels,
                    colors=["white"], linewidths=0.8, alpha=0.5)
    ax.clabel(cs, fmt="%.2f MPa", fontsize=8, colors="white")
    cb = plt.colorbar(cf, ax=ax, pad=0.02)
    cb.set_label("P90 Joint Margin (MPa)", color="white", fontsize=11)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
    ax.set_xlabel("Injection Rate q (k Sm³/day)", color="white", fontsize=12)
    ax.set_ylabel("Temperature Differential ΔT (°C)", color="white", fontsize=12)
    ax.set_xlim(Q_MIN / 1000, Q_MAX / 1000)
    ax.set_ylim(DT_MIN, DT_MAX)
    ax.set_title("P90 Safety Margin — Red River UHS\n"
                 "Green = headroom  |  Red = failed  |  Dashed = boundary",
                 color="white", fontsize=13)
    _save("envelope_p90_margin.png")

    # ── Figure 4: Mechanism competition map ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(BG)
    _style_ax(ax, BG)

    gap = pgn_p90 - res_p90
    vext = max(abs(gap).max(), 0.5)
    cf = ax.contourf(QQ, DD, gap,
                     levels=np.linspace(-vext, vext, 40),
                     cmap="coolwarm", extend="both")
    ax.contour(QQ, DD, gap, levels=[0], colors=["black"], linewidths=4)
    ax.contour(QQ, DD, gap, levels=[0], colors=["yellow"], linewidths=2)
    ax.contour(QQ, DD, p90_all, levels=[0], colors=["white"],
               linewidths=2, linestyles="--")
    cb = plt.colorbar(cf, ax=ax, pad=0.02)
    cb.set_label("PGN margin − RMG margin (MPa)", color="white", fontsize=11)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
    ax.set_xlabel("Injection Rate q (k Sm³/day)", color="white", fontsize=12)
    ax.set_ylabel("Temperature Differential ΔT (°C)", color="white", fontsize=12)
    ax.set_xlim(Q_MIN / 1000, Q_MAX / 1000)
    ax.set_ylim(DT_MIN, DT_MAX)
    ax.set_title("Mechanism Competition — Red River UHS\n"
                 "Blue = RMG binds  |  Red = PGN binds  |  Yellow = frontier",
                 color="white", fontsize=13)
    _save("envelope_competition.png")

    # ── Figure 5: CSV table (already saved in run_envelope) ───────────────────
    print("  → envelope_results.csv  (operator rate-limit table)")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    results = run_envelope()
    plot_all(*results)
    print("\n" + "=" * 70)
    print("DONE — Output figures:")
    print("  envelope_binding_map.png  ← binding map + Ca-M-G SOE")
    print("  envelope_qmax_curve.png   ← q_max vs ΔT with confidence band")
    print("  envelope_p90_margin.png   ← P90 safety margin surface")
    print("  envelope_competition.png  ← mechanism competition map")
    print("  envelope_results.npz      ← raw arrays")
    print("  envelope_results.csv      ← operator rate-limit table")
    print("=" * 70)