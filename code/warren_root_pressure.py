#!/usr/bin/env python
"""
warren_root_pressure.py - Warren-Root dual-porosity transient pressure

Computes transient pressure multipliers f(t) using Stehfest Laplace inversion.
"""

import numpy as np
import math
from scipy.special import k0, k1

# ============================================================================
# STEHFEST ALGORITHM
# ============================================================================

def stehfest_weights(N=12):
    """
    Compute Stehfest weights for Laplace inversion.
    
    Args:
        N: Number of terms (must be even, typical: 10-14)
    
    Returns:
        Array of N weights
    """
    
    assert N % 2 == 0, "N must be even"
    
    V = np.zeros(N)
    
    for i in range(1, N + 1):
        sum_val = 0.0
        
        k_min = int((i + 1) / 2)
        k_max = min(i, N // 2)
        
        for k in range(k_min, k_max + 1):
            term = (k ** (N // 2)) * math.factorial(2 * k)
            denom = (math.factorial(N // 2 - k) * 
                    math.factorial(k) * 
                    math.factorial(k - 1) * 
                    math.factorial(i - k) * 
                    math.factorial(2 * k - i))
            sum_val += term / denom
        
        V[i - 1] = ((-1) ** (i + N // 2)) * sum_val
    
    return V

# Precompute weights
STEHFEST_WEIGHTS = stehfest_weights(N=12)

def stehfest_invert(laplace_func, t, weights=STEHFEST_WEIGHTS):
    """
    Invert Laplace transform using Stehfest algorithm.
    
    Args:
        laplace_func: Function f(s) in Laplace domain
        t: Time value (or array) to evaluate at
        weights: Precomputed Stehfest weights
    
    Returns:
        f(t) in time domain
    """
    
    N = len(weights)
    ln2_t = np.log(2.0) / t
    
    result = 0.0
    for i in range(N):
        s = (i + 1) * ln2_t
        result += weights[i] * laplace_func(s)
    
    return result * ln2_t

# ============================================================================
# WARREN-ROOT SOLUTION
# ============================================================================

def warren_root_laplace(s, lambda_val, omega):
    """
    Warren-Root dual-porosity solution in Laplace domain.
    
    Args:
        s: Laplace parameter
        lambda_val: Interporosity flow coefficient
        omega: Storativity ratio (fracture/total)
    
    Returns:
        Dimensionless pressure in Laplace domain
    """
    
    # Characteristic function f(s)
    f_s = omega * s + (1.0 - omega) * lambda_val * s / (s + lambda_val)
    
    # Modified Bessel functions
    x = np.sqrt(s * f_s)
    
    if x < 1e-10:
        # Near-zero limit
        return 0.5 / s
    
    K0_x = k0(x)
    K1_x = k1(x)
    
    # Dimensionless pressure at wellbore (rD = 1)
    pD = K0_x / (x * K1_x)
    
    return pD / s

def compute_transient_multipliers(omega, tau_exchange_days, t_eval_days, t_anchor_days=182.5):
    """
    Compute transient pressure multipliers f(t) for given times.
    
    Args:
        omega: Storativity ratio [0.05, 0.50]
        tau_exchange_days: Exchange time scale [10, 500] days
        t_eval_days: Array of times to evaluate (days)
        t_anchor_days: Anchor time where f = 1.0 (default 182.5)
    
    Returns:
        dict: {time: multiplier} with f(t_anchor) = 1.0
    """
    
    # Convert to dimensionless parameters
    lambda_val = 1.0 / tau_exchange_days
    
    # Evaluate at requested times
    multipliers = {}
    
    # Reference value at anchor time
    def laplace_ref(s):
        return warren_root_laplace(s, lambda_val, omega)
    
    p_anchor = stehfest_invert(laplace_ref, t_anchor_days)
    
    # Evaluate at all times and normalize
    for t in t_eval_days:
        if t <= 0:
            multipliers[t] = 1.0
        else:
            p_t = stehfest_invert(laplace_ref, t)
            multipliers[t] = p_t / p_anchor
    
    return multipliers

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: typical parameters
    omega = 0.2  # 20% in fractures
    tau_exchange = 300.0  # 300 days exchange time
    
    times = [1.0, 7.0, 30.0, 90.0, 182.5]
    
    multipliers = compute_transient_multipliers(omega, tau_exchange, times)
    
    print("Warren-Root Transient Multipliers:")
    print("=" * 40)
    for t, f in sorted(multipliers.items()):
        print(f"t = {t:6.1f} days: f(t) = {f:.4f}")
