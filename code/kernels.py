#!/usr/bin/env python
"""
kernels.py - Safe operating envelope margin calculations

Mechanisms
----------
  AIG_margin   : Caprock thermal fracture (erfc attenuation, embrittlement)
  RMG_margin : Near-well thermal/mechanical failure
  PGN_margin_caprock   : Capillary seal entry pressure (H2 wettability shift included)
  CMG_stability        : Fingering/sweep stability screen (M + Ca; G stub ready)
  joint_margin         : min(PGN, AIG, RMG) + CMG soft derate

Temperature coupling
--------------------
  E stiffening  : E_eff = E_virgin × (1 + (dE/dT) × ΔT)
  Embrittlement : T_tensile_eff = T_tensile × (1 - embrittlement_factor × |ΔT|/10)

H2 wettability
--------------
  theta_eff = theta_base + dtheta_H2  (dtheta_H2 sampled as uncertain prior ~[5, 25] deg)
  Increasing theta -> lower cos(theta) -> lower Pe -> tighter PGN margin
"""

import math
import numpy as np

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def Bg_rm3_per_Sm3(P_res_MPa, T_res_C, Z=1.0):
    """Gas formation volume factor."""
    Pstd_MPa = 0.101325
    Tstd_K = 288.15
    Tres_K = float(T_res_C) + 273.15
    Pres = max(float(P_res_MPa), 1e-6)
    return (Pstd_MPa / Pres) * (Tres_K / Tstd_K) * Z


def cycle_thermal_factor(t_inj_days=182.5, t_with_days=182.5,
                         tau_cool_days=60.0, tau_warm_days=90.0):
    """
    Lumped-capacitance cyclic thermal memory factor.

    Returns f_end in (0, 1] such that ΔT_rock ≈ f_end × ΔT_perf at
    end of injection, accounting for partial re-heating during withdrawal.

    Model (periodic steady state of inject/withdraw cycle):
      During injection:  dΔT/dt = (ΔT_perf - ΔT) / tau_cool
      During withdrawal: dΔT/dt = (0       - ΔT) / tau_warm

    f_end < 1 means the rock never fully cools to ΔT_perf because
    residual heat from the previous warm-back partially offsets each
    new cooling cycle.

    Args:
        t_inj_days:   injection half-cycle length (default 182.5 = 6 months)
        t_with_days:  withdrawal half-cycle length (default 182.5)
        tau_cool_days: thermal time constant for cooling (days)
        tau_warm_days: thermal time constant for re-heating (days)

    Returns:
        f_end: fraction of ΔT_perf realised at end of injection [0, 1]
    """
    tau_cool = max(float(tau_cool_days), 1e-6)
    tau_warm = max(float(tau_warm_days), 1e-6)

    a = math.exp(-float(t_inj_days) / tau_cool)
    b = math.exp(-float(t_with_days) / tau_warm)

    denom = max(1.0 - a * b, 1e-12)
    f_start = (b * (1.0 - a)) / denom
    f_end   = (1.0 - a) + a * f_start
    return float(np.clip(f_end, 0.0, 1.0))

# ============================================================================
# AIG CAPROCK (Thermal Fracture)
# ============================================================================

def AIG_margin(q, dT_perf, cap, f_transient=1.0):
    """
    Caprock thermal fracture margin.
    
    Args:
        q: Injection rate (Sm³/day)
        dT_perf: Temperature differential at perforations (°C)
        cap: Caprock parameter dict
        f_transient: Transient pressure multiplier (default 1.0)
    
    Returns:
        Margin (MPa) - positive = safe, negative = failure
    """
    
    # Extract parameters
    E = cap["E_GPa"]
    nu = cap["nu"]
    alpha_T = cap["alpha_T_perK"]
    UCS = cap["UCS_MPa"]
    FoS = cap.get("FoS", 1.75
)   # Caprock: conservative FoS
    # cap_thickness_m: physical caprock thickness for erfc thermal attenuation (1-5 m)
    # Distinct from reservoir_standoff_m (perf-to-caprock distance, 10-20 m) used in PGN.
    h = cap.get("cap_thickness_m", cap.get("h_m", 3.0))
    alpha_diff = cap["alpha_m2s"]
    exposure_days = cap.get("exposure_days", 30.0)
    chi = cap["chi_confinement"]
    p_trans = cap.get("pressure_transmission_frac", 0.1)
    
    # Temperature coupling parameters
    dE_dT = cap.get("dE_dT_GPa_per_C", -0.02)
    embrittle_rate = cap.get("embrittlement_factor_per_10C", 0.05)
    
    # ── H2 chemical exposure effects (Aluah et al. 2025) ────────────────
    # H2 stiffens anhydrite (~11-17% E increase) and embrittles it.
    # Applied to baseline E BEFORE thermal coupling.
    # H2_stiffening_frac: fractional E increase from H2 exposure (e.g. 0.14 = 14%)
    # H2_strength_reduction_frac: fractional UCS/tensile reduction (e.g. 0.10 = 10%)
    H2_stiff = cap.get("H2_stiffening_frac", 0.0)
    H2_weaken = cap.get("H2_strength_reduction_frac", 0.0)
    
    E = E * (1.0 + H2_stiff)                # stiffer → more thermal stress (worse)
    UCS = UCS * (1.0 - H2_weaken)           # weaker → lower allowable (worse)
    
    # Thermal attenuation (erfc) at standoff distance
    t_seconds = exposure_days * 86400.0
    z_norm = h / (2.0 * np.sqrt(alpha_diff * t_seconds))
    
    from scipy.special import erfc
    attenuation = erfc(z_norm)
    
    dT_cap = dT_perf * attenuation
    
    # ── Cycle-aware thermal memory (lumped capacitance) ──────────────────
    # Caprock rock partially re-heats during withdrawal season.
    # Applied AFTER erfc attenuation: dT_cap_eff = dT_cap × f_end.
    # f_end = 1.0 if tau values absent (backward-compatible, conservative).
    tau_cool = cap.get("tau_cool_days", 0.0)
    tau_warm = cap.get("tau_warm_days", 0.0)
    if tau_cool > 0.0 and tau_warm > 0.0:
        f_end = cycle_thermal_factor(182.5, 182.5, tau_cool, tau_warm)
    else:
        f_end = 1.0
    
    dT_cap = dT_cap * f_end
    
    # Temperature coupling
    # E stiffening
    E_eff = E * (1.0 + dE_dT * dT_cap / E)
    E_eff = max(E_eff, 0.5 * E)  # Safety clip
    
    # Embrittlement
    embrittle_factor = 1.0 - embrittle_rate * abs(dT_cap) / 10.0
    embrittle_factor = max(embrittle_factor, 0.7)  # Max 30% reduction
    
    # Thermal stress
    sigma_thermal = E_eff * alpha_T * abs(dT_cap) / (1.0 - nu)
    
    # Pressure transmission
    # Estimate pressure from simple radial flow
    PI = cap.get("PI_m3day_per_MPa", 380.0)
    skin = cap.get("skin", 1.5)
    Bg = cap.get("Bg_rm3_per_Sm3", 0.005)
    
    q_res = q * Bg
    f_plume = cap.get("f_plume", 1.0)
    Delta_p = (q_res / (PI * f_plume)) * (1.0 + skin) * f_transient
    
    sigma_pressure = chi * p_trans * Delta_p
    
    # Total stress
    sigma_total = sigma_thermal + sigma_pressure
    
    # Allowable stress (with embrittlement)
    # If T_tensile_MPa is provided directly (lab-measured or sampled prior),
    # use it. Otherwise fall back to Brazilian test correlation UCS / 10.
    if "T_tensile_MPa" in cap:
        T_tensile = cap["T_tensile_MPa"] * (1.0 - H2_weaken)
    else:
        T_tensile = UCS / 10.0  # Brazilian test correlation
    T_allow = (T_tensile * embrittle_factor) / FoS

    # Margin
    margin = T_allow - sigma_total

    return margin

# ============================================================================
# AIG RESERVOIR (Near-Well Thermal/Mechanical)
# ============================================================================

def RMG_margin(q, dT_perf, res, f_transient=1.0):
    """
    Reservoir near-wellbore thermal/mechanical margin.
    
    Args:
        q: Injection rate (Sm³/day)
        dT_perf: Temperature differential at perforations (°C)
        res: Reservoir parameter dict
        f_transient: Transient pressure multiplier (default 1.0)
    
    Returns:
        Margin (MPa) - positive = safe, negative = failure
    """
    
    # Extract parameters
    E = res["E_GPa"]
    nu = res["nu"]
    alpha_T = res["alpha_T_perK"]
    chi = res["chi_confinement"]
    T_tensile = res["T_tensile_MPa"]
    FoS = res.get("FoS", 1.5)   # Reservoir: FoS=1.5 consistent with AIG caprock
    
    # Damage zone check
    is_damaged = res.get("is_damaged_zone", False)
    if is_damaged:
        T_tensile = res.get("T_tensile_damaged_MPa", T_tensile * 0.6)
    
    # Well parameters
    PI = res["PI_m3day_per_MPa"]
    skin = res.get("skin", 1.0)
    Bg = res.get("Bg_rm3_per_Sm3", 0.005)
    
    # Temperature coupling parameters
    dE_dT = res.get("dE_dT_GPa_per_C", -0.02)
    embrittle_rate = res.get("embrittlement_factor_per_10C", 0.05)
    
    # ── H2 chemical exposure effects (Aluah et al. 2025) ────────────────
    # H2 stiffens carbonate (~2-11% E increase) and may reduce strength.
    # Applied to baseline E and T_tensile BEFORE thermal coupling.
    H2_stiff = res.get("H2_stiffening_frac", 0.0)
    H2_weaken = res.get("H2_strength_reduction_frac", 0.0)
    
    E = E * (1.0 + H2_stiff)                # stiffer → more thermal stress (worse)
    T_tensile = T_tensile * (1.0 - H2_weaken)  # weaker → lower allowable (worse)
    
    # ── Cycle-aware thermal memory (lumped capacitance) ──────────────────
    # Rock partially re-heats during withdrawal; f_end < 1 at periodic SS.
    # f_end = 1.0 if tau values absent (backward-compatible, conservative).
    tau_cool = res.get("tau_cool_days", 0.0)
    tau_warm = res.get("tau_warm_days", 0.0)
    if tau_cool > 0.0 and tau_warm > 0.0:
        f_end = cycle_thermal_factor(182.5, 182.5, tau_cool, tau_warm)
    else:
        f_end = 1.0
    
    dT_eff = dT_perf * f_end          # signed, for E coupling
    dT_cooling = max(-dT_eff, 0.0)     # only cooling creates tensile stress
    
    # Temperature coupling
    # E stiffening (cooling stiffens rock: dE/dT < 0, dT < 0 → E increases)
    E_eff = E * (1.0 + dE_dT * dT_eff / E)
    E_eff = max(E_eff, 0.5 * E)  # Safety clip
    
    # Embrittlement (only cooling embrittles)
    embrittle_factor = 1.0 - embrittle_rate * dT_cooling / 10.0
    embrittle_factor = max(embrittle_factor, 0.7)  # Max 30% reduction
    
    # Thermal stress (near wellbore, no attenuation)
    # Only cooling (contraction) creates tensile stress; heating is safe
    sigma_thermal = E_eff * alpha_T * dT_cooling / (1.0 - nu)
    
    # Pressure-induced stress
    q_res = q * Bg
    f_plume = res.get("f_plume", 1.0)
    Delta_p = (q_res / (PI * f_plume)) * (1.0 + skin) * f_transient
    
    sigma_pressure = chi * Delta_p
    
    # Total stress
    sigma_total = sigma_thermal + sigma_pressure
    
    # Allowable stress (with embrittlement)
    T_allow = (T_tensile * embrittle_factor) / FoS
    
    # Margin
    margin = T_allow - sigma_total
    
    return margin

# ============================================================================
# PGN CAPILLARY ENTRY
# ============================================================================

def PGN_margin_caprock(q, dT_perf, pgn, cap=None, f_transient=1.0):
    """
    Capillary seal entry pressure margin.
    
    Args:
        q: Injection rate (Sm³/day)
        dT_perf: Temperature differential at perforations (°C)
        pgn: PGN parameter dict
        cap: Optional caprock dict for pressure calculation
        f_transient: Transient pressure multiplier (default 1.0)
    
    Returns:
        Margin (MPa) - positive = safe, negative = failure
    """
    
    # ── IFT and wettability ─────────────────────────────────────────────────────
    IFT            = pgn["IFT_mN_per_m"]
    IFT_scale_dT   = pgn.get("IFT_scale_dT_perC",   -0.2)
    IFT_scale_dSal = pgn.get("IFT_scale_dSal_perM",  2.0)
    salinity       = pgn.get("salinity_M",            1.0)
    theta_deg      = pgn["contact_angle_deg"]

    # H2 wettability alteration (Aluah et al. 2025):
    # advancing θ used for drainage (injection) — conservative vs static θ
    dtheta_H2     = pgn.get("dtheta_H2_deg", 0.0)
    theta_eff_deg = min(theta_deg + dtheta_H2, 89.0)

    # Temperature and salinity effects on IFT (additive, mN/m per unit)
    IFT_eff = IFT + IFT_scale_dT * dT_perf + IFT_scale_dSal * salinity
    IFT_eff = max(IFT_eff, 10.0)
    sigma     = IFT_eff * 1e-3
    cos_theta = np.cos(np.deg2rad(theta_eff_deg))

    # ── Brooks-Corey capillary entry pressure ────────────────────────────────
    # Pe = p_d × Sw*^(-1/λ)
    # p_d: displacement pressure anchored to reference throat radius (Young-Laplace)
    # λ:   pore-size distribution index — calibratable from NMR T2
    # Sw*: normalised wetting saturation at caprock face
    # Ref: Brooks & Corey (1964); Hildenbrand et al. (2004)
    log10_r = pgn["log10_pore_throat_radius"]
    r_ref   = 10.0 ** log10_r
    p_d_ref = (2.0 * sigma * cos_theta / r_ref) / 1e6   # MPa

    bc_lambda = pgn.get("bc_lambda",  2.0)   # prior [1.0, 4.0]
    Sw_irr    = pgn.get("Sw_irr",    0.20)   # prior [0.10, 0.35]
    Sw_face   = pgn.get("Sw_face",   0.90)   # prior [0.70, 0.95]

    Sw_star = max((Sw_face - Sw_irr) / max(1.0 - Sw_irr, 1e-6), 1e-4)
    Pe_bc   = p_d_ref * (Sw_star ** (-1.0 / bc_lambda))

    # Weakening (lamination in anhydrite seal, [0.90, 1.0])
    vug_factor  = pgn.get("vug_weakening_factor",      1.0)
    frac_factor = pgn.get("fracture_weakening_factor", 1.0)
    weakening   = min(vug_factor, frac_factor)

    Pe = Pe_bc * weakening
    
    # Pressure buildup (use cap dict if available, else estimate)
    if cap is not None:
        PI = cap.get("PI_m3day_per_MPa", 380.0)
        skin = cap.get("skin", 1.0)
        Bg = cap.get("Bg_rm3_per_Sm3", 0.005)
    else:
        # Fallback estimates
        PI = 380.0
        skin = 1.0
        Bg = 0.005
    
    q_res = q * Bg
    f_plume = cap.get("f_plume", 1.0) if cap is not None else 1.0

    # p_trans_pgn: fraction of wellbore ΔP reaching the caprock face.
    # Computed geometrically from reservoir_standoff_m (perf-to-caprock distance)
    # via log-radial attenuation: p_trans = ln(r_s/r_w) / ln(r_e/r_w)
    # This physically represents how much pressure the caprock face sees relative
    # to the wellbore drawdown across a reservoir of thickness ~reservoir_standoff_m.
    # Falls back to sampled p_trans_pgn if reservoir_standoff_m not provided.
    if cap is not None and "reservoir_standoff_m" in cap:
        r_w = cap.get("r_w_pgn_m", 0.1)
        r_e = cap.get("r_e_m", 500.0)        # drainage radius ~500 m
        r_s = max(cap["reservoir_standoff_m"], r_w * 1.01)
        ln_re_rw = max(np.log(r_e / r_w), 1e-6)
        ln_rs_rw = max(np.log(r_s / r_w), 1e-6)
        p_trans_pgn = min(ln_rs_rw / ln_re_rw, 1.0)
    else:
        p_trans_pgn = cap.get("p_trans_pgn", 0.5) if cap is not None else 0.5

    Delta_p = (q_res / (PI * f_plume)) * (1.0 + skin) * f_transient * p_trans_pgn

    # Margin
    margin = Pe - Delta_p
    
    return margin

# ============================================================================
# CMG STABILITY SCREEN (Fingering / Sweep)
# ============================================================================

def CMG_stability(q, cmg, pgn=None):
    """
    Mobility + Capillary number stability screen — H2/brine calibrated.

    Design principles
    -----------------
    1. Continuous derate: f_stab = f_M * f_Ca blended smoothly, no step bins.
    2. H2-relative M scoring: M is normalized to a baseline M_ref (~100 for
       H2/brine) so that the inherently high H2 mobility ratio doesn't
       automatically pin everything to worst-case. What matters operationally
       is how much WORSE than baseline a given realisation is, not the
       absolute value.
    3. Ca provides spatial variation with q: at low q Ca is small (capillary-
       dominated, relatively stable sweep); at high q Ca grows and adds a
       rate-dependent penalty on top of the M term.
    4. G stub ready — wire in dip/Δρ when confirmed from Friday's data.

    Scoring
    -------
    M_score  = sigmoid((M/M_ref - 1) / w_M)        in [0, 1]
                 → 0 when M << M_ref (better than baseline)
                 → 0.5 when M = M_ref (at baseline)
                 → 1 when M >> M_ref (much worse than baseline)

    Ca_score = sigmoid((log10(Ca) - log10(Ca_ref)) / w_Ca)  in [0, 1]
                 → 0 when Ca << Ca_ref (capillary-dominated, stable)
                 → 0.5 at Ca_ref transition
                 → 1 when Ca >> Ca_ref (viscous-dominated)

    Combined instability index:
      I = w_blend * M_score + (1 - w_blend) * Ca_score

    Derate:
      f_stab = f_min + (1 - f_min) * (1 - I)
             = 1.0 when fully stable, f_min when fully unstable

    Tunable priors (all in cmg dict)
    ---------------------------------
    M_ref      : H2/brine baseline mobility ratio       default 100
    w_M        : M sigmoid width (decades)              default 0.5
    Ca_ref     : Ca transition reference                default 1e-20
    w_Ca       : Ca sigmoid width (log10 decades)       default 1.0
    w_blend    : weight of M vs Ca in combined index    default 0.6
    f_min      : minimum derate (floor)                 default 0.40

    Args
    ----
    q   : Injection rate (Sm³/day)
    cmg : Parameter dict (see above)
    pgn : Optional PGN dict — reuses IFT if provided

    Returns
    -------
    f_stab  : Derate factor [f_min, 1.0]
    M       : Mobility ratio
    Ca      : Capillary number
    regime  : Descriptive string for diagnostics
    """
    k_md        = cmg["k_md"]
    mu_H2       = cmg["mu_H2_cP"]
    mu_brine    = cmg["mu_brine_cP"]
    kr_H2       = cmg["kr_H2_end"]
    kr_brine    = cmg["kr_brine_end"]
    r_w         = cmg.get("r_w_m", 0.1)
    h_perf      = cmg.get("h_perf_m", 10.0)

    # Tunable scaling parameters
    M_ref   = cmg.get("M_ref",    100.0)   # H2/brine baseline M
    w_M     = cmg.get("w_M",        0.5)   # M sigmoid width (fraction of M_ref)
    Ca_ref  = cmg.get("Ca_ref",   1e-20)   # Ca transition — calibrated to H2/brine/mD range
    w_Ca    = cmg.get("w_Ca",       1.0)   # Ca sigmoid width (log10 decades)
    w_blend = cmg.get("w_blend",    0.6)   # M weight in combined index
    f_min   = cmg.get("f_min",     0.40)   # minimum derate floor

    # IFT
    if pgn is not None:
        IFT_mNm = pgn.get("IFT_mN_per_m", cmg.get("IFT_mN_per_m", 40.0))
    else:
        IFT_mNm = cmg.get("IFT_mN_per_m", 40.0)

    # ── Mobility ratio M ──────────────────────────────────────────────────────
    lambda_H2    = kr_H2    / max(mu_H2,    1e-9)
    lambda_brine = kr_brine / max(mu_brine, 1e-9)
    M = lambda_H2 / max(lambda_brine, 1e-10)

    # M score: sigmoid centred at M_ref, width w_M * M_ref
    # Argument: how many "widths" above baseline we are
    M_arg   = (M / M_ref - 1.0) / max(w_M, 1e-6)
    M_score = 1.0 / (1.0 + np.exp(-M_arg))   # sigmoid in [0,1]

    # ── Capillary number Ca ───────────────────────────────────────────────────
    q_m3s   = q / 86400.0
    A_perf  = 2.0 * np.pi * r_w * h_perf
    v_darcy = q_m3s / max(A_perf, 1e-6)

    sigma_Nm  = IFT_mNm * 1e-3
    mu_H2_Pa  = mu_H2   * 1e-3
    k_SI      = k_md * 9.869e-16   # mD → m²
    Ca = (mu_H2_Pa * v_darcy * k_SI) / max(sigma_Nm, 1e-6)

    # Ca score: sigmoid in log10 space centred at log10(Ca_ref)
    Ca_safe = max(Ca, 1e-30)
    Ca_arg  = (np.log10(Ca_safe) - np.log10(Ca_ref)) / max(w_Ca, 1e-6)
    Ca_score = 1.0 / (1.0 + np.exp(-Ca_arg))

    # ── Combined instability index ────────────────────────────────────────────
    I = w_blend * M_score + (1.0 - w_blend) * Ca_score   # in [0, 1]

    # ── Derate ────────────────────────────────────────────────────────────────
    f_stab = f_min + (1.0 - f_min) * (1.0 - I)
    f_stab = float(np.clip(f_stab, f_min, 1.0))

    # ── Regime label (diagnostic) ─────────────────────────────────────────────
    if I < 0.35:
        regime = "stable"
    elif I < 0.65:
        regime = "mildly_unstable"
    else:
        regime = "unstable"

    # ── Gravity stub ─────────────────────────────────────────────────────────
    # G = k·Δρ·g·sin(dip) / (μ_H2·v)
    # TODO: wire in when dip and Δρ confirmed from Friday's characterisation

    return f_stab, M, Ca, regime


# ============================================================================
# JOINT MARGIN (Minimum of All Mechanisms + CMG Derate)
# ============================================================================

def joint_margin(q, dT_perf, cap, res, pgn, f_transient=1.0):
    """
    Joint margin = minimum of PGN, AIG, RMG at the actual rate q.

    CMG is intentionally NOT applied here. It is a post-processing derate
    on q_max_base, not a modifier on the physics margins. See CMG_stability
    and apply as: q_max_derated = f_stab * q_max_base.

    Returns
    -------
    min_margin   : Minimum margin across all mechanisms (MPa)
    binding_mech : Index of binding mechanism (0=PGN, 1=AIG, 2=RMG)
    """
    margin_pgn     = PGN_margin_caprock(q, dT_perf, pgn, cap, f_transient)
    margin_aig = AIG_margin(q, dT_perf, cap, f_transient)
    margin_rmg = RMG_margin(q, dT_perf, res, f_transient)

    margins = [margin_pgn, margin_aig, margin_rmg]
    min_margin   = min(margins)
    binding_mech = margins.index(min_margin)

    return min_margin, binding_mech
