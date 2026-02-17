from __future__ import annotations

import numpy as np
from typing import Tuple

from .utils import json_error

# -------------------------
# Infinite well (matches your well notebook)
# -------------------------
def build_position_grid_infinite_well(n: int, L: float) -> Tuple[np.ndarray, float]:
    """
    Interior grid points for the infinite well on (0, L) with Dirichlet walls at 0 and L.
    x_i = (i+1) dx, dx = L/(n+1), i=0..n-1.
    """
    dx = float(L) / (int(n) + 1)
    x = dx * np.arange(1, int(n) + 1)
    return x, dx

def build_infinite_well_hamiltonian(
    n: int,
    L: float,
    hbar: float = 1.0,
    mass: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    H = p^2/(2m) with Dirichlet walls implemented via finite differences on interior points.
    """
    x, dx = build_position_grid_infinite_well(n, L)

    diag_main = -2.0 * np.ones(n)
    diag_off = 1.0 * np.ones(n - 1)
    D2 = (
        np.diag(diag_main, 0)
        + np.diag(diag_off, +1)
        + np.diag(diag_off, -1)
    ) / (dx**2)

    H = - (hbar**2) / (2.0 * mass) * D2
    H = 0.5 * (H + H.T)  # enforce Hermitian numerically
    return H.astype(complex), x, dx

def gaussian_wavepacket_well(
    x: np.ndarray,
    x0: float,
    sigma: float,
    k0: float,
    enforce_walls: bool = True,
) -> np.ndarray:
    """
    Exactly the well-notebook construction:
    envelope * plane-wave phase, then optional clamp psi[0]=psi[-1]=0, then normalize by l2 norm.
    """
    x = np.asarray(x, dtype=float)
    envelope = np.exp(-0.5 * ((x - float(x0)) / float(sigma)) ** 2)
    phase = np.exp(1j * float(k0) * (x - float(x0)))
    psi = (envelope * phase).astype(complex)

    if enforce_walls:
        psi[0] = 0.0
        psi[-1] = 0.0

    norm = np.linalg.norm(psi)
    if norm == 0:
        json_error("Initial state has zero norm. Try different x0/sigma.", param="initial_state")
    return psi / norm

# -------------------------
# Harmonic oscillator (matches your HO notebook)
# -------------------------
def build_position_grid_ho(n: int, x_max: float) -> Tuple[np.ndarray, float]:
    """Return equally spaced grid points x_j in [-x_max, x_max) with endpoint=False."""
    x = np.linspace(-float(x_max), float(x_max), int(n), endpoint=False)
    dx = float(x[1] - x[0])
    return x, dx

def build_ho_hamiltonian(n: int, x_max: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    In units ħ=m=ω=1:
      H = -1/2 d^2/dx^2 + x^2/2
    """
    hbar = 1.0
    m = 1.0
    w = 1.0

    x, dx = build_position_grid_ho(n, x_max)

    diag_main = -2.0 * np.ones(n)
    diag_off = 1.0 * np.ones(n - 1)
    D2 = (
        np.diag(diag_main, 0)
        + np.diag(diag_off, +1)
        + np.diag(diag_off, -1)
    ) / (dx**2)

    T = - (hbar**2) / (2.0 * m) * D2
    V = np.diag(0.5 * m * (w**2) * x**2)

    H = T + V
    H = 0.5 * (H + H.T)  # enforce Hermitian numerically
    return H.astype(complex), x, dx

def gaussian_wavepacket_ho(
    x: np.ndarray,
    x0: float,
    sigma: float,
    k0: float,
) -> np.ndarray:
    """Exactly the HO-notebook construction: envelope * phase, then l2 normalize."""
    x = np.asarray(x, dtype=float)
    envelope = np.exp(-0.5 * ((x - float(x0)) / float(sigma)) ** 2)
    phase = np.exp(1j * float(k0) * (x - float(x0)))
    psi = (envelope * phase).astype(complex)
    norm = np.linalg.norm(psi)
    if norm == 0:
        json_error("Gaussian normalization failed, adjust x0/sigma", param="initial_state")
    return psi / norm

# -------------------------
# Quick sanity checks (matches notebook prints)
# -------------------------
def sanity_check(H: np.ndarray, psi0: np.ndarray, label: str) -> None:
    herm_err = np.linalg.norm(H - H.conj().T)
    norm_psi = np.linalg.norm(psi0)
    evals = np.linalg.eigvalsh(H.real)
    print(f"[{label}] H shape: {H.shape}")
    print(f"[{label}] Hermiticity ||H-H^†||: {herm_err:.3e}")
    print(f"[{label}] ||psi0||2: {norm_psi:.12f}")
    print(f"[{label}] lowest 5 eigvals(H): {evals[:5]}")
