"""
corpus_generator.py
===================
Generates the UHS surrogate training corpus.

HDF5 Storage — 3D TENSOR FORMAT
---------------------------------
  corpus.h5
  ├── meta/
  │   └── params_lhs  (N_SOBOL_MAX, N_PARAMS) float32  — global Sobol design
  └── stage_MEGA/  (or stage_A, stage_B, stage_C)
      ├── attrs: n_q, n_dT, n_sobol        — stage-specific dimensions
      ├── q_axis   (n_q,)          float32  — stage-specific grid
      ├── dT_axis  (n_dT,)         float32  — stage-specific grid
      ├── m_pgn    (n_q, n_dT, n_sobol) float32
      ├── m_aig    (n_q, n_dT, n_sobol) float32
      ├── m_rmg    (n_q, n_dT, n_sobol) float32
      └── done     (n_q, n_dT)     bool

  Each stage stores its OWN q_axis, dT_axis, and n_sobol count.
  params_lhs is global and sliced to [:n_sobol] per stage at read time.
  m_joint = min(m_pgn, m_aig, m_rmg)  — recomputed at read time, not stored
  binding = argmin(...)                — recomputed at read time, not stored

Worker design
-------------
  params_lhs is loaded ONCE per worker process via ProcessPoolExecutor
  initializer — not pickled per task. Each task receives only (gi, gj, q, dT).
  Eliminates ~1.3 TB of IPC traffic for MEGA stage.

Usage
-----
  python corpus_generator.py --stage MEGA
  python corpus_generator.py --stage A
  python corpus_generator.py --validate MEGA
  python corpus_generator.py --summary
  python corpus_generator.py --stage MEGA --smoke
"""

from __future__ import annotations
import sys
import os
import time
import math
import argparse
import logging
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Set

try:
    import h5py
except ImportError:
    sys.exit("h5py required: pip install h5py")

try:
    from scipy.stats import qmc
except ImportError:
    sys.exit("scipy required: pip install scipy")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from kernels import AIG_margin, RMG_margin, PGN_margin_caprock, Bg_rm3_per_Sm3
from warren_root_pressure import compute_transient_multipliers
from surrogate_utils import (
    PARAM_NAMES, N_PARAMS, PARAM_IDX,
    HALO_LO, HALO_HI,
    Q_MIN_SM3, Q_MAX_SM3, DT_MIN_C, DT_MAX_C,
    FIXED_FOR_SURROGATE, FIXED_MIDPOINTS,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

CORPUS_FILE = Path(__file__).parent / "corpus.h5"

STAGE_GRIDS = {
    "A":    (65,   65),
    "B":    (129,  129),
    "C":    (257,  257),
    "MEGA": (1000, 1000),
}

# Sobol sample count per stage — must be powers of 2
N_SOBOL_PER_STAGE = {
    "A":    256,
    "B":    512,
    "C":    1024,
    "MEGA": 8192,
}
N_SOBOL_MAX = 8192   # global design stored in meta — all stages slice from this

BATCH_SIZE   = 64     # grid points per HDF5 write batch
N_WORKERS    = max(1, min(os.cpu_count() - 1, 15))
MAX_INFLIGHT = 1024

SMOKE_GRIDS   = {"A": (9,9),  "B": (17,17), "C": (33,33), "MEGA": (11,11)}
SMOKE_N_SOBOL = {"A": 16,     "B": 16,      "C": 32,      "MEGA": 64}

VAL_P90_RMSE_MPa = 0.05
VAL_BINDING_ACC  = 0.92

# HDF5 chunk: one grid point's full Sobol slice — optimal for per-point reads
# For MEGA: chunk = (1, 1, 8192) → each chunk is one (i,j) point = 32 KB
CHUNK_SOBOL = N_SOBOL_MAX

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("corpus")

# =============================================================================
# WORKER GLOBAL STATE — loaded once per worker process via initializer
# =============================================================================

_WORKER_PARAMS_LHS: np.ndarray = None   # set by _worker_init


def _worker_init(params_lhs_arr: np.ndarray):
    """
    Load params_lhs into worker-global state once per worker process.
    Receives the array directly from the main process — no file I/O.
    Avoids Windows HDF5 file locking when main process holds the file open.
    """
    global _WORKER_PARAMS_LHS
    _WORKER_PARAMS_LHS = params_lhs_arr

# =============================================================================
# GRID + SOBOL DESIGN
# =============================================================================

def make_grid(n_q: int, n_dT: int) -> Tuple[np.ndarray, np.ndarray]:
    return (np.linspace(Q_MIN_SM3, Q_MAX_SM3, n_q),
            np.linspace(DT_MIN_C,  DT_MAX_C,  n_dT))


def make_sobol_params(n: int = N_SOBOL_MAX, seed: int = 42) -> np.ndarray:
    """
    Generate n Sobol points mapped to halo parameter bounds.
    n must be a power of 2. Returns float32 (n, N_PARAMS).
    Fixed-for-surrogate params set to midpoint values.
    """
    sampler = qmc.Sobol(d=N_PARAMS, scramble=True, seed=seed)
    unit    = sampler.random_base2(m=int(math.log2(n)))
    params  = (HALO_LO + unit * (HALO_HI - HALO_LO)).astype(np.float32)
    for pname, midval in FIXED_MIDPOINTS.items():
        params[:, PARAM_IDX[pname]] = np.float32(midval)
    return params


def new_grid_indices(stage: str) -> np.ndarray:
    n_q, n_dT = STAGE_GRIDS[stage]
    all_ij    = set((i, j) for i in range(n_q) for j in range(n_dT))

    if stage in ("A", "MEGA"):
        return np.array(sorted(all_ij))

    prev_stages = {"B": ["A"], "C": ["A", "B"]}[stage]
    covered: Set[Tuple[int, int]] = set()
    for prev in prev_stages:
        pn_q, pn_dT = STAGE_GRIDS[prev]
        scale_q  = (n_q  - 1) // (pn_q  - 1)
        scale_dT = (n_dT - 1) // (pn_dT - 1)
        for pi in range(pn_q):
            for pj in range(pn_dT):
                covered.add((pi * scale_q, pj * scale_dT))

    return np.array(sorted(all_ij - covered))

# =============================================================================
# PHYSICS KERNEL HELPERS
# =============================================================================

def _f_transient(omega, tau_days, t_eval=30.0, t_anchor=182.5):
    try:
        mults = compute_transient_multipliers(
            omega, tau_days,
            t_eval_days=[t_eval, t_anchor],
            t_anchor_days=t_anchor,
        )
        return float(mults[t_eval])
    except Exception:
        return 1.0


def build_params_from_row(row: np.ndarray):
    p = dict(zip(PARAM_NAMES, row))
    P_res   = p["P_res_MPa"]
    T_res   = p["T_res_C"]
    Bg      = Bg_rm3_per_Sm3(P_res, T_res)
    f_trans = _f_transient(p["omega"], p["tau_exchange_days"])

    cap = {
        "E_GPa":                        p["cap_E_GPa"],
        "nu":                           p["cap_nu"],
        "alpha_T_perK":                 p["cap_alpha_T_perK"],
        "UCS_MPa":                      p["cap_UCS_MPa"],
        "T_tensile_MPa":                p["cap_T_tensile_MPa"],
        "alpha_m2s":                    p["cap_alpha_m2s"],
        "chi_confinement":              p["cap_chi"],
        "pressure_transmission_frac":   p["cap_p_trans"],
        "cap_thickness_m":              p["cap_thickness_m"],
        "reservoir_standoff_m":         p["reservoir_standoff_m"],
        "tau_cool_days":                p["cap_tau_cool"],
        "tau_warm_days":                p["cap_tau_warm"],
        "exposure_days":                30.0,
        "FoS":                          p["FoS_cap"],
        "dE_dT_GPa_per_C":             p["cap_dE_dT"],
        "embrittlement_factor_per_10C": p["cap_embrittle"],
        "PI_m3day_per_MPa":             p["PI_m3day_per_MPa"],
        "skin":                         p["skin"],
        "f_plume":                      p["f_plume"],
        "Bg_rm3_per_Sm3":               Bg,
        "H2_stiffening_frac":           p["cap_H2_stiff"],
        "H2_strength_reduction_frac":   p["cap_H2_weaken"],
    }

    is_damaged = p["damage_prob"] < p["damage_thresh"]
    res = {
        "E_GPa":                        p["res_E_GPa"],
        "nu":                           p["res_nu"],
        "alpha_T_perK":                 p["res_alpha_T_perK"],
        "chi_confinement":              p["res_chi"],
        "T_tensile_MPa":                p["res_T_tensile_MPa"],
        "T_tensile_damaged_MPa":        p["res_T_tensile_dmg_MPa"],
        "is_damaged_zone":              is_damaged,
        "FoS":                          p["FoS_res"],
        "dE_dT_GPa_per_C":             p["res_dE_dT"],
        "embrittlement_factor_per_10C": p["res_embrittle"],
        "PI_m3day_per_MPa":             p["PI_m3day_per_MPa"],
        "skin":                         p["skin"],
        "f_plume":                      p["f_plume"],
        "Bg_rm3_per_Sm3":               Bg,
        "tau_cool_days":                p["res_tau_cool"],
        "tau_warm_days":                p["res_tau_warm"],
        "H2_stiffening_frac":           p["res_H2_stiff"],
        "H2_strength_reduction_frac":   p["res_H2_weaken"],
    }

    pgn = {
        "IFT_mN_per_m":              p["pgn_IFT"],
        "IFT_scale_dT_perC":         p["pgn_IFT_dT"],
        "IFT_scale_dSal_perM":       p["pgn_IFT_dSal"],
        "contact_angle_deg":         p["pgn_theta"],
        "dtheta_H2_deg":             p["pgn_dtheta_H2"],
        "salinity_M":                p["pgn_salinity"],
        "log10_pore_throat_radius":  p["pgn_log10r"],
        "vug_weakening_factor":      p["pgn_vug"],
        "f_plume":                   p["f_plume"],
        "fracture_weakening_factor": p["pgn_frac"],
        "bc_lambda":                 p["pgn_bc_lambda"],
        "Sw_irr":                    p["pgn_Sw_irr"],
        "Sw_face":                   p["pgn_Sw_face"],
    }

    return cap, res, pgn, f_trans

# =============================================================================
# SINGLE-POINT WORKER
# =============================================================================

def _eval_point(args):
    """
    Evaluate all Sobol parameter rows at one (q, dT) grid point.
    Uses worker-global _WORKER_PARAMS_LHS — not passed per task.
    Returns (grid_i, grid_j, m_pgn, m_aig, m_rmg) float32 arrays.
    """
    grid_i, grid_j, q, dT = args
    lhs_rows = _WORKER_PARAMS_LHS   # loaded once by _worker_init
    N        = len(lhs_rows)

    m_pgn = np.empty(N, dtype=np.float32)
    m_aig = np.empty(N, dtype=np.float32)
    m_rmg = np.empty(N, dtype=np.float32)

    for k, row in enumerate(lhs_rows):
        try:
            cap, res, pgn, f_t = build_params_from_row(row)
            mp = PGN_margin_caprock(q, dT, pgn, cap, f_transient=f_t)
            ma = AIG_margin(q, dT, cap, f_transient=f_t)
            mr = RMG_margin(q, dT, res, f_transient=f_t)
        except Exception:
            mp = ma = mr = float("nan")
        m_pgn[k] = mp
        m_aig[k] = ma
        m_rmg[k] = mr

    return grid_i, grid_j, m_pgn, m_aig, m_rmg

# =============================================================================
# HDF5 I/O — TENSOR FORMAT
# =============================================================================

def _open_corpus(path: Path, mode: str = "a") -> h5py.File:
    return h5py.File(path, mode, rdcc_nbytes=256*1024*1024)


def _init_meta(f: h5py.File, params_lhs: np.ndarray):
    """Write global Sobol design to meta/ once. Skip if already exists."""
    if "meta" in f:
        return
    meta = f.create_group("meta")
    meta.create_dataset("params_lhs", data=params_lhs,
                        compression="lzf", shuffle=True)
    log.info(f"  Wrote meta/params_lhs {params_lhs.shape}")


def _init_stage(f: h5py.File, stage: str,
                n_q: int, n_dT: int, n_sobol: int,
                q_arr: np.ndarray, dT_arr: np.ndarray):
    """
    Pre-allocate 3D margin datasets for a stage.
    Stores stage-specific q_axis, dT_axis, and n_sobol as attrs.
    HDF5 chunked datasets: only chunks with actual data hit disk.
    """
    grp_name = f"stage_{stage}"
    if grp_name in f:
        return

    grp = f.create_group(grp_name)
    grp.attrs["n_q"]     = n_q
    grp.attrs["n_dT"]    = n_dT
    grp.attrs["n_sobol"] = n_sobol

    # Store stage-specific axes inside the stage group
    grp.create_dataset("q_axis",  data=q_arr.astype(np.float32))
    grp.create_dataset("dT_axis", data=dT_arr.astype(np.float32))

    opts  = dict(compression="lzf", shuffle=True)
    chunk = (1, 1, min(n_sobol, CHUNK_SOBOL))

    for name in ("m_pgn", "m_aig", "m_rmg"):
        grp.create_dataset(name,
                           shape=(n_q, n_dT, n_sobol),
                           dtype=np.float32,
                           chunks=chunk,
                           **opts)

    grp.create_dataset("done",
                       shape=(n_q, n_dT),
                       dtype=bool,
                       data=np.zeros((n_q, n_dT), dtype=bool))

    raw_gb = 3 * n_q * n_dT * n_sobol * 4 / 1e9
    log.info(f"  Initialised stage_{stage}: "
             f"({n_q}, {n_dT}, {n_sobol}), raw≈{raw_gb:.1f} GB")


def _get_completed_keys(f: h5py.File, stage: str) -> Set[Tuple[int, int]]:
    grp_name = f"stage_{stage}"
    if grp_name not in f:
        return set()
    done = f[grp_name]["done"][:]
    ii, jj = np.where(done)
    return set(zip(ii.tolist(), jj.tolist()))


def _write_batch(f: h5py.File, stage: str, results: list):
    grp = f[f"stage_{stage}"]
    for grid_i, grid_j, m_pgn, m_aig, m_rmg in results:
        grp["m_pgn"][grid_i, grid_j, :] = m_pgn
        grp["m_aig"][grid_i, grid_j, :] = m_aig
        grp["m_rmg"][grid_i, grid_j, :] = m_rmg
        grp["done"][grid_i, grid_j]      = True

# =============================================================================
# MAIN STAGE RUNNER
# =============================================================================

def run_stage(stage: str, smoke: bool = False):
    n_q, n_dT  = (SMOKE_GRIDS if smoke else STAGE_GRIDS)[stage]
    n_sobol    = (SMOKE_N_SOBOL if smoke else N_SOBOL_PER_STAGE)[stage]
    q_arr, dT_arr = make_grid(n_q, n_dT)
    new_ij     = new_grid_indices(stage)

    log.info(f"Stage {stage}: grid={n_q}×{n_dT}, "
             f"n_sobol={n_sobol}, workers={N_WORKERS}")

    # Generate Sobol design in main process first — workers receive it as array
    # (avoids Windows HDF5 file locking when workers try to open the corpus file)
    params_lhs_full = make_sobol_params(N_SOBOL_MAX)
    params_lhs      = params_lhs_full[:n_sobol]   # stage-specific slice

    # Ensure global Sobol design exists in corpus file
    with _open_corpus(CORPUS_FILE) as f:
        if "meta" not in f:
            log.info(f"  Generating global Sobol design ({N_SOBOL_MAX} pts) ...")
            _init_meta(f, params_lhs_full)
        _init_stage(f, stage, n_q, n_dT, n_sobol, q_arr, dT_arr)
        completed = _get_completed_keys(f, stage)

    remaining_ij = [(int(i), int(j)) for i, j in new_ij
                    if (int(i), int(j)) not in completed]
    log.info(f"  {len(completed)} already done, {len(remaining_ij)} to compute")

    if not remaining_ij:
        log.info(f"Stage {stage} already complete.")
        return

    t0      = time.time()
    n_done  = 0
    n_total = len(remaining_ij)

    # Tasks are just (gi, gj, q, dT) — no params_lhs passed per task
    def _make_task(gi, gj):
        return (gi, gj, float(q_arr[gi]), float(dT_arr[gj]))

    with _open_corpus(CORPUS_FILE) as f:
        batch_results = []

        # Pass params_lhs array directly to workers — avoids Windows HDF5 file locking
        # (main process holds file open in write mode; workers cannot open it simultaneously)
        with ProcessPoolExecutor(
            max_workers  = N_WORKERS,
            initializer  = _worker_init,
            initargs     = (params_lhs,),
        ) as pool:
            from collections import deque
            pending   = deque()
            task_iter = iter(remaining_ij)

            for gi, gj in task_iter:
                pending.append(pool.submit(_eval_point, _make_task(gi, gj)))
                if len(pending) >= MAX_INFLIGHT:
                    break

            while pending:
                fut = pending.popleft()
                batch_results.append(fut.result())
                n_done += 1

                if len(batch_results) >= BATCH_SIZE:
                    _write_batch(f, stage, batch_results)
                    f.flush()
                    batch_results.clear()

                if n_done % 20000 == 0 or n_done == n_total:
                    elapsed = time.time() - t0
                    rate    = n_done / max(elapsed, 1.0)
                    eta     = (n_total - n_done) / max(rate, 1e-6)
                    log.info(f"  {n_done}/{n_total}  "
                             f"elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min")

                nxt = next(task_iter, None)
                if nxt is not None:
                    gi, gj = nxt
                    pending.append(pool.submit(_eval_point, _make_task(gi, gj)))

        if batch_results:
            _write_batch(f, stage, batch_results)
            f.flush()

    elapsed = time.time() - t0
    log.info(f"Stage {stage} complete in {elapsed/60:.1f} min. Corpus: {CORPUS_FILE}")

# =============================================================================
# VALIDATION
# =============================================================================

def validate_stage(stage: str):
    log.info(f"Validating Stage {stage} corpus ...")

    try:
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
    except ImportError:
        log.warning("xgboost/sklearn not available — skipping validation")
        return

    with _open_corpus(CORPUS_FILE, "r") as f:
        grp_name = f"stage_{stage}"
        if grp_name not in f:
            log.error(f"No {grp_name} in corpus")
            return

        grp     = f[grp_name]
        n_sobol = int(grp.attrs["n_sobol"])
        # Load stage-specific axes and matching params slice
        q_arr      = grp["q_axis"][:]
        dT_arr     = grp["dT_axis"][:]
        params_lhs = f["meta"]["params_lhs"][:n_sobol].astype(np.float32)

        done    = grp["done"][:]
        ii, jj  = np.where(done)
        n_pts   = min(500, len(ii))
        idx     = np.random.choice(len(ii), n_pts, replace=False)
        ii, jj  = ii[idx], jj[idx]

        X_list, yj_list, yb_list = [], [], []
        for i, j in zip(ii, jj):
            mp = grp["m_pgn"][i, j, :].astype(np.float32)
            ma = grp["m_aig"][i, j, :].astype(np.float32)
            mr = grp["m_rmg"][i, j, :].astype(np.float32)
            mj = np.minimum(mp, np.minimum(ma, mr))
            bd = np.argmin(np.stack([mp, ma, mr], axis=1),
                           axis=1).astype(np.int32)
            valid = np.isfinite(mj) & (bd >= 0)
            if valid.sum() < 2:
                continue
            q_col  = np.full(n_sobol, q_arr[i],  dtype=np.float32)
            dT_col = np.full(n_sobol, dT_arr[j], dtype=np.float32)
            X = np.column_stack([q_col[valid], dT_col[valid], params_lhs[valid]])
            X_list.append(X)
            yj_list.append(mj[valid])
            yb_list.append(bd[valid])

    if not X_list:
        log.error("No valid data found")
        return

    X     = np.vstack(X_list).astype(np.float32)
    y_jnt = np.concatenate(yj_list).astype(np.float32)
    y_bnd = np.concatenate(yb_list).astype(np.int32)
    log.info(f"  Validation rows: {len(X):,}")

    X_tr, X_va, yj_tr, yj_va, yb_tr, yb_va = train_test_split(
        X, y_jnt, y_bnd, test_size=0.2, random_state=42)

    reg = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                            tree_method="hist", device="cpu", verbosity=0, n_jobs=-1)
    reg.fit(X_tr, yj_tr)
    yj_pred  = reg.predict(X_va)
    rmse     = math.sqrt(np.mean((yj_va - yj_pred) ** 2))
    near_bnd = np.abs(yj_va) < 0.5
    bnd_rmse = math.sqrt(np.mean((yj_va[near_bnd] - yj_pred[near_bnd]) ** 2)) \
               if near_bnd.sum() > 0 else float("nan")

    clf = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                             tree_method="hist", device="cpu", verbosity=0,
                             n_jobs=-1, use_label_encoder=False,
                             eval_metric="mlogloss")
    clf.fit(X_tr, yb_tr)
    acc = accuracy_score(yb_va, clf.predict(X_va))

    log.info(f"  Overall RMSE      : {rmse:.4f} MPa")
    log.info(f"  P90-boundary RMSE : {bnd_rmse:.4f} MPa  "
             f"({'PASS' if bnd_rmse < VAL_P90_RMSE_MPa else 'FAIL'})")
    log.info(f"  Binding accuracy  : {acc:.3f}  "
             f"({'PASS' if acc >= VAL_BINDING_ACC else 'FAIL'})")

# =============================================================================
# CORPUS SUMMARY
# =============================================================================

def corpus_summary():
    if not CORPUS_FILE.exists():
        print("Corpus file not found.")
        return
    with _open_corpus(CORPUS_FILE, "r") as f:
        if "meta" in f and "params_lhs" in f["meta"]:
            print(f"meta/params_lhs: {f['meta']['params_lhs'].shape}")
        for stage in ["A", "B", "C", "MEGA"]:
            grp_name = f"stage_{stage}"
            if grp_name in f:
                grp     = f[grp_name]
                n_sobol = int(grp.attrs["n_sobol"])
                shape   = grp["m_pgn"].shape
                done    = grp["done"][:]
                n_done  = done.sum()
                n_total = shape[0] * shape[1]
                raw_gb  = 3 * shape[0] * shape[1] * n_sobol * 4 / 1e9
                print(f"Stage {stage}: {shape}, done={n_done}/{n_total} "
                      f"({100*n_done/max(n_total,1):.1f}%), raw≈{raw_gb:.1f} GB")
            else:
                print(f"Stage {stage}: not generated")

# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="UHS Corpus Generator")
    parser.add_argument("--stage",    choices=["A","B","C","MEGA"])
    parser.add_argument("--validate", choices=["A","B","C","MEGA"])
    parser.add_argument("--summary",  action="store_true")
    parser.add_argument("--smoke",    action="store_true")
    args = parser.parse_args()

    if args.summary:
        corpus_summary()
    elif args.validate:
        validate_stage(args.validate)
    elif args.stage:
        run_stage(args.stage, smoke=args.smoke)
        if not args.smoke:
            validate_stage(args.stage)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
