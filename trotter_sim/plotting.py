from __future__ import annotations

import os
import numpy as np

# IMPORTANT: use non-interactive backend so "python run_all.py" never hangs
import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from typing import Dict, List, Sequence, Tuple

def ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def plot_initial_density(x: np.ndarray, psi0: np.ndarray, title: str, outpath: str) -> None:
    plt.figure(figsize=(10, 3.5))
    plt.plot(x, np.abs(psi0) ** 2, lw=2)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("|psi|^2")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print("Saved:", outpath)

def plot_fidelity(t: np.ndarray, F: np.ndarray, title: str, outpath: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(t, F, lw=2)
    plt.ylim(0, 1.5)
    plt.xlabel("t")
    plt.ylabel("F(t)")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print("Saved:", outpath)

def labels_to_state_dict(labels: Sequence[str], snapshots: np.ndarray) -> Dict[int, np.ndarray]:
    d: Dict[int, np.ndarray] = {}
    for lbl, sv in zip(labels, snapshots):
        k = int(str(lbl).split("_")[-1])
        d[k] = np.asarray(sv, dtype=complex)
    return d

def choose_snapshot_indices(r_steps: int, num: int = 4) -> List[int]:
    if num < 2:
        return [0, int(r_steps)]
    raw = np.linspace(0, int(r_steps), num=int(num))
    idx = sorted(set(int(round(x)) for x in raw))
    idx[0] = 0
    idx[-1] = int(r_steps)
    return idx

def plot_snapshots_overlay_vertical(
    x: np.ndarray,
    probs_ref: np.ndarray,
    state_dict_trot: Dict[int, np.ndarray],
    dt: float,
    indices: Sequence[int],
    title: str,
    outpath: str,
    fig_width: float = 5.6,
    row_height: float = 2.4,
) -> None:
    indices = list(indices)
    for k in indices:
        if k not in state_dict_trot:
            raise ValueError(
                f"Missing circuit snapshot for k={k}. "
                f"Available ks: {min(state_dict_trot)}..{max(state_dict_trot)}"
            )

    n_panels = len(indices)
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(fig_width, row_height * n_panels),
        sharex=True,
        sharey=True
    )
    if n_panels == 1:
        axes = [axes]

    for ax, k in zip(axes, indices):
        p_ref = probs_ref[int(k)]
        psi_t = state_dict_trot[int(k)]
        p_trot = np.abs(psi_t) ** 2

        t = int(k) * float(dt)
        ax.plot(x, p_ref, lw=2, label="reference")
        ax.plot(x, p_trot, lw=2, ls="--", label="circuit")
        ax.set_title(f"t = {t:.3g}")
        ax.set_ylabel(r"$|\psi(x,t)|^2$")
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("x")
    axes[0].legend(loc="best", frameon=False)

    fig.suptitle(title, y=1.01, fontsize=13)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print("Saved:", outpath)

def plot_error_vs_r(r_list: Sequence[int], infids: np.ndarray, title: str, outpath: str) -> None:
    plt.figure(figsize=(6.2, 4.2))
    plt.plot(list(r_list), infids, marker="o", lw=2)
    plt.yscale("log")
    plt.xlabel("Trotter steps r")
    plt.ylabel(r"Final-time infidelity  $1 - F(t_{\max})$")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print("Saved:", outpath)

def plot_acc_vs_resources(cnot_totals: np.ndarray, infids: np.ndarray, title: str, outpath: str) -> None:
    plt.figure(figsize=(6.2, 4.2))
    plt.plot(cnot_totals, infids, marker="o", lw=2)
    plt.yscale("log")
    plt.xlabel("Total CNOT count (r × CNOT/step)")
    plt.ylabel(r"Final-time infidelity  $1 - F(t_{\max})$")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print("Saved:", outpath)

def plot_ho_xexp(
    x_ho: np.ndarray,
    probs_ho_ref: np.ndarray,
    dt_ho: float,
    labels_ho: Sequence[str],
    snaps_ho: np.ndarray,
    outpath: str,
) -> None:
    """
    Fig. ho_xexp. Expectation value <x>(t) for HO. Reference vs circuit.
    Matches your final notebook cell (larger text, thicker lines).
    """
    x = np.asarray(x_ho, dtype=float)
    probs_ref_all = np.asarray(probs_ho_ref, dtype=float)

    if labels_ho is not None and len(labels_ho) > 0:
        k_idx = np.array([int(str(lbl).split("_")[-1]) for lbl in labels_ho], dtype=int)
        probs_ref = probs_ref_all[k_idx]
        snaps = snaps_ho
        t = k_idx * float(dt_ho)
    else:
        probs_ref = probs_ref_all
        snaps = snaps_ho
        t = np.arange(probs_ref.shape[0]) * float(dt_ho)

    xexp_ref = probs_ref @ x

    snaps = list(snaps)
    if len(snaps) != len(t):
        raise ValueError(
            f"Time points mismatch. len(snaps_ho)={len(snaps)} but len(t)={len(t)}. "
            "Make sure snaps_ho is saved at the same time indices as the reference."
        )

    xexp_trot = np.array([np.dot(np.abs(psi) ** 2, x) for psi in snaps], dtype=float)

    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
    })

    plt.figure(figsize=(9.5, 5.5))
    plt.plot(t, xexp_ref, lw=3.2, label="reference")
    plt.plot(t, xexp_trot, lw=3.2, ls="--", label="circuit")
    plt.xlabel("t")
    plt.ylabel(r"$\langle x \rangle(t)$")
    plt.title("Harmonic oscillator: turning points and period via expectation value")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=450, bbox_inches="tight")
    plt.close()
    print("Saved:", outpath)
