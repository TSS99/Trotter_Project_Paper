from __future__ import annotations

import os
import numpy as np
from typing import Tuple

def evolve_exact_via_eigenstates(
    H: np.ndarray,
    psi0: np.ndarray,
    total_time: float,
    n_steps: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Computes reference dynamics using the eigenstate expansion:
      H φ_n = E_n φ_n
      ψ(t_k) = Σ_n c_n e^{-i E_n t_k} φ_n,  c_n = <φ_n|ψ0>
    Returns: times, probs_over_time, states_over_time, dt, evals
    """
    H = np.asarray(H, dtype=complex)
    psi0 = np.asarray(psi0, dtype=complex)
    psi0 = psi0 / np.linalg.norm(psi0)

    dt = float(total_time) / int(n_steps)
    times = np.linspace(0.0, float(total_time), int(n_steps) + 1)

    # Hermitian eigendecomposition: H = V diag(E) V^†
    evals, evecs = np.linalg.eigh(H)

    # Expansion coefficients in energy basis
    c = evecs.conj().T @ psi0  # (N,)

    # Vectorized construction over all time points:
    phases = np.exp(-1j * evals[:, None] * times[None, :])         # (N, T)
    coeffs_t = c[:, None] * phases                                 # (N, T)
    states = (evecs @ coeffs_t).T                                  # (T, N)

    # Norm drift check
    norms = np.linalg.norm(states, axis=1)
    max_norm_dev = float(np.max(np.abs(norms - 1.0)))
    print("Max | ||psi(t)|| - 1 | over time =", f"{max_norm_dev:.3e}")

    # Hard renormalization (matches notebook)
    states = states / norms[:, None]
    probs = np.abs(states) ** 2
    return times, probs, states, dt, evals

def save_reference_run_eigen(
    tag: str,
    x_grid: np.ndarray,
    H: np.ndarray,
    psi0: np.ndarray,
    total_time: float,
    n_steps: int,
    out_dir: str = "data",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
    os.makedirs(out_dir, exist_ok=True)
    times, probs, states, dt, evals = evolve_exact_via_eigenstates(H, psi0, total_time, n_steps)

    outpath = os.path.join(out_dir, f"ref_eigen_{tag}.npz")
    np.savez_compressed(
        outpath,
        x_grid=np.asarray(x_grid),
        times=times,
        dt=dt,
        evals=evals,
        psi0=np.asarray(psi0),
        probs=probs,
        states=states,
    )
    print(f"Saved eigen-expansion reference -> {outpath}")
    return times, probs, states, dt, outpath

def reference_state_at_t(H: np.ndarray, psi0: np.ndarray, t: float) -> np.ndarray:
    """Reference state from eigen expansion at a single time (matches notebook sweep)."""
    E, V = np.linalg.eigh(np.asarray(H, dtype=complex))
    c = V.conj().T @ np.asarray(psi0, dtype=complex)
    phase = np.exp(-1j * E * float(t))
    psi_t = V @ (c * phase)
    return psi_t / np.linalg.norm(psi_t)
