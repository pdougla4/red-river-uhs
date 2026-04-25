#!/usr/bin/env python
"""
uq_runner.py - Uncertainty quantification helpers

Contains Morris screening implementation and LHS sampling utilities.
"""

import numpy as np

# ============================================================================
# MORRIS SCREENING - FIXED VERSION
# ============================================================================

def morris_fixed(func, bounds, r=10, p=4, max_retries=10):
    """
    Morris (OAT) global sensitivity screening with boundary handling.
    
    FIXED: Handles case where parameter is at boundary and can't move.
    
    Args:
        func: Model function taking [0,1]^D input, returning scalar
        bounds: List of (lo, hi) tuples for each parameter (normalized to [0,1])
        r: Number of trajectories (higher = more accurate, typical: 4-20)
        p: Number of levels for grid (typical: 4-8)
        max_retries: Maximum retries if trajectory generation fails
    
    Returns:
        dict with "mu_star" and "sigma" arrays (length D)
    """
    
    D = len(bounds)
    
    # Storage for elementary effects
    all_EE = []
    
    for traj in range(r):
        # Generate Morris trajectory with retry logic
        retry_count = 0
        trajectory_valid = False
        
        while not trajectory_valid and retry_count < max_retries:
            try:
                B = generate_morris_trajectory_fixed(D, p)
                trajectory_valid = True
            except ValueError:
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"WARNING: Could not generate valid trajectory after {max_retries} attempts")
                    continue
        
        if not trajectory_valid:
            continue
        
        # Evaluate along trajectory
        for i in range(D):
            x0 = B[i, :]
            x1 = B[i + 1, :]
            
            # Find which parameter changed
            changed_mask = np.abs(x0 - x1) > 1e-10
            
            if not np.any(changed_mask):
                # No change - skip this step
                continue
            
            changed_idx = np.where(changed_mask)[0][0]
            delta = x1[changed_idx] - x0[changed_idx]
            
            if abs(delta) < 1e-10:
                # Delta too small - skip
                continue
            
            # Evaluate
            y0 = func(x0)
            y1 = func(x1)
            
            # Elementary effect
            EE = (y1 - y0) / delta
            
            # Store (parameter_index, EE)
            all_EE.append((changed_idx, EE))
    
    # Compute statistics
    mu_star = np.zeros(D)
    sigma = np.zeros(D)
    mu = np.zeros(D)
    
    for i in range(D):
        # Extract EEs for parameter i
        EE_i = [ee for idx, ee in all_EE if idx == i]
        
        if len(EE_i) > 0:
            mu_star[i] = np.mean(np.abs(EE_i))
            sigma[i] = np.std(EE_i)
            mu[i] = np.mean(EE_i)
    
    return {
        "mu_star": mu_star,
        "sigma": sigma,
        "mu": mu
    }

def generate_morris_trajectory_fixed(D, p=4):
    """
    Generate a single Morris trajectory with better boundary handling.
    
    Returns (D+1) × D matrix where each row differs from previous in exactly one dimension.
    """
    
    # Step size
    delta = p / (2.0 * (p - 1))
    
    # Generate base point ensuring room to move
    grid_points = np.linspace(delta, 1.0 - delta, p - 2)  # Avoid exact boundaries
    x_base = np.random.choice(grid_points, size=D, replace=True)
    
    # Random order to perturb dimensions
    order = np.random.permutation(D)
    
    # Build trajectory
    trajectory = np.zeros((D + 1, D))
    trajectory[0, :] = x_base
    
    for i, dim in enumerate(order):
        trajectory[i + 1, :] = trajectory[i, :].copy()
        
        current_val = trajectory[i, dim]
        
        # Determine which directions are valid
        can_increase = (current_val + delta) <= 1.0
        can_decrease = (current_val - delta) >= 0.0
        
        if can_increase and can_decrease:
            # Both directions valid - choose randomly
            direction = np.random.choice([-1, 1])
        elif can_increase:
            # Only increase valid
            direction = 1
        elif can_decrease:
            # Only decrease valid
            direction = -1
        else:
            # Stuck at boundary (shouldn't happen with new grid_points)
            raise ValueError(f"Parameter {dim} stuck at boundary")
        
        new_val = current_val + direction * delta
        trajectory[i + 1, dim] = np.clip(new_val, 0.0, 1.0)
    
    return trajectory

# Keep old Morris for compatibility
def morris(func, bounds, r=10, p=4):
    """Original Morris - use morris_fixed for production."""
    return morris_fixed(func, bounds, r, p)
    """
    Morris (OAT) global sensitivity screening.
    
    Args:
        func: Model function taking [0,1]^D input, returning scalar
        bounds: List of (lo, hi) tuples for each parameter (normalized to [0,1])
        r: Number of trajectories (higher = more accurate, typical: 4-20)
        p: Number of levels for grid (typical: 4-8)
    
    Returns:
        dict with "mu_star" and "sigma" arrays (length D)
    """
    
    D = len(bounds)
    
    # Storage for elementary effects
    all_EE = []
    
    for traj in range(r):
        # Generate Morris trajectory (r × (D+1) design points)
        B = generate_morris_trajectory(D, p)
        
        # Evaluate along trajectory
        for i in range(D):
            # Base point and perturbed point
            x0 = B[i, :]
            x1 = B[i + 1, :]
            
            # Evaluate
            y0 = func(x0)
            y1 = func(x1)
            
            # Elementary effect for parameter that changed
            changed_idx = np.where(x0 != x1)[0][0]
            delta = x1[changed_idx] - x0[changed_idx]
            
            EE = (y1 - y0) / delta
            
            # Store (parameter_index, EE)
            all_EE.append((changed_idx, EE))
    
    # Compute statistics
    mu_star = np.zeros(D)
    sigma = np.zeros(D)
    
    for i in range(D):
        # Extract EEs for parameter i
        EE_i = [ee for idx, ee in all_EE if idx == i]
        
        if len(EE_i) > 0:
            mu_star[i] = np.mean(np.abs(EE_i))
            sigma[i] = np.std(EE_i)
    
    return {
        "mu_star": mu_star,
        "sigma": sigma,
        "mu": np.array([np.mean([ee for idx, ee in all_EE if idx == i]) for i in range(D)])
    }

def generate_morris_trajectory(D, p=4):
    """
    Generate a single Morris trajectory.
    
    Returns (D+1) × D matrix where each row differs from previous in exactly one dimension.
    """
    
    # Step size
    delta = p / (2.0 * (p - 1))
    
    # Random base point on grid
    x_base = np.random.choice(np.linspace(0, 1 - delta, p), size=D)
    
    # Random order to perturb dimensions
    order = np.random.permutation(D)
    
    # Build trajectory
    trajectory = np.zeros((D + 1, D))
    trajectory[0, :] = x_base
    
    for i, dim in enumerate(order):
        trajectory[i + 1, :] = trajectory[i, :].copy()
        
        # Perturb dimension 'dim' by +delta or -delta
        direction = np.random.choice([-1, 1])
        new_val = trajectory[i + 1, dim] + direction * delta
        
        # Clip to [0, 1]
        new_val = np.clip(new_val, 0.0, 1.0)
        trajectory[i + 1, dim] = new_val
    
    return trajectory

# ============================================================================
# LATIN HYPERCUBE SAMPLING
# ============================================================================

def lhs(n_samples, n_dims, seed=None):
    """
    Simple Latin Hypercube Sampling.
    
    Args:
        n_samples: Number of samples
        n_dims: Number of dimensions
        seed: Random seed
    
    Returns:
        n_samples × n_dims array with values in [0, 1]
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    # Create grid
    samples = np.zeros((n_samples, n_dims))
    
    for j in range(n_dims):
        # Divide [0, 1] into n_samples intervals
        intervals = np.linspace(0, 1, n_samples + 1)
        
        # Random point within each interval
        for i in range(n_samples):
            lo = intervals[i]
            hi = intervals[i + 1]
            samples[i, j] = np.random.uniform(lo, hi)
        
        # Shuffle to break diagonal correlation
        np.random.shuffle(samples[:, j])
    
    return samples
