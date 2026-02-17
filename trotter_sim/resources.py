from __future__ import annotations

import os
import numpy as np
from typing import Dict, List, Tuple

# plotting backend is controlled by plotting.py
import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from qiskit.quantum_info import Operator, SparsePauliOp

def pauli_decompose_terms(H: np.ndarray, coeff_tol: float = 1e-12) -> List[Tuple[str, float]]:
    op = SparsePauliOp.from_operator(Operator(np.asarray(H, dtype=complex)))
    labels = op.paulis.to_labels()
    coeffs = op.coeffs

    terms: List[Tuple[str, float]] = []
    for lab, c in zip(labels, coeffs):
        if abs(c) > coeff_tol:
            terms.append((lab, float(np.real(c))))
    return terms

def mk_sk_from_label(label: str) -> Tuple[int, int]:
    mk = sum(ch != "I" for ch in label)
    sk = sum(ch in ("X", "Y") for ch in label)
    return mk, sk

def cnot_per_term(mk: int) -> int:
    return 0 if mk <= 1 else 2 * (mk - 1)

def oneq_per_term(sk: int) -> int:
    return 2 * sk + 1  # basis changes in/out + one Rz

def depth_per_term(mk: int, sk: int) -> int:
    basis = 1 if sk > 0 else 0
    cnot_depth = 0 if mk <= 1 else 2 * (mk - 1)
    return basis + cnot_depth + 1 + basis

def summarize_terms(terms: List[Tuple[str, float]]) -> Dict[str, object]:
    labels = [lab for lab, _ in terms]
    mk = np.array([mk_sk_from_label(lab)[0] for lab in labels], dtype=int)
    sk = np.array([mk_sk_from_label(lab)[1] for lab in labels], dtype=int)

    cnot_k = np.array([cnot_per_term(int(m)) for m in mk], dtype=int)
    oneq_k = np.array([oneq_per_term(int(s)) for s in sk], dtype=int)
    depth_k = np.array([depth_per_term(int(m), int(s)) for m, s in zip(mk, sk)], dtype=int)

    info: Dict[str, object] = {
        "M": len(terms),
        "mk": mk,
        "sk": sk,
        "cnot_k": cnot_k,
        "oneq_k": oneq_k,
        "depth_k": depth_k,
        "cnot_step": int(cnot_k.sum()),
        "oneq_step": int(oneq_k.sum()),
        "depth_step": int(depth_k.sum()),
        "mk_min": int(mk.min()) if len(mk) else 0,
        "mk_max": int(mk.max()) if len(mk) else 0,
        "mk_mean": float(mk.mean()) if len(mk) else 0.0,
    }
    return info

def plot_pauli_stats(info_well: Dict[str, object], info_ho: Dict[str, object], N_well: int, N_ho: int, outpath: str) -> None:
    mk_w = np.asarray(info_well["mk"])
    mk_h = np.asarray(info_ho["mk"])

    max_m = max(int(mk_w.max()), int(mk_h.max()))
    bins = np.arange(0, max_m + 2) - 0.5

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.2))
    ax.hist(mk_w, bins=bins, alpha=0.6, label=f"Well (N={N_well}, M={info_well['M']})")
    ax.hist(mk_h, bins=bins, alpha=0.6, label=f"HO (N={N_ho}, M={info_ho['M']})")
    ax.set_xticks(range(0, max_m + 1))
    ax.set_xlabel(r"Active-qubit count  $m_k$")
    ax.set_ylabel("Number of Pauli terms")
    ax.set_title("Pauli-term active-qubit count distribution")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print("Saved:", outpath)

def plot_resources_vs_r(info_well: Dict[str, object], info_ho: Dict[str, object], r_list_well: List[int], r_list_ho: List[int], outpath: str) -> None:
    r_w = np.array(r_list_well, dtype=int)
    r_h = np.array(r_list_ho, dtype=int)

    cnot_w = int(info_well["cnot_step"]) * r_w
    depth_w = int(info_well["depth_step"]) * r_w

    cnot_h = int(info_ho["cnot_step"]) * r_h
    depth_h = int(info_ho["depth_step"]) * r_h

    fig, ax1 = plt.subplots(1, 1, figsize=(7.2, 4.3))
    ax2 = ax1.twinx()

    ax1.plot(r_w, cnot_w, marker="o", lw=2, label="Well: total CNOT")
    ax1.plot(r_h, cnot_h, marker="o", lw=2, ls="--", label="HO: total CNOT")
    ax2.plot(r_w, depth_w, marker="s", lw=2, label="Well: depth")
    ax2.plot(r_h, depth_h, marker="s", lw=2, ls="--", label="HO: depth")

    ax1.set_xlabel("Trotter steps r")
    ax1.set_ylabel("Total CNOT count")
    ax2.set_ylabel("Estimated circuit depth")
    ax1.grid(True, alpha=0.25)
    ax1.set_title("Resource scaling with r")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print("Saved:", outpath)

def write_resource_table_tex(path: str, system_name: str, n: int, N: int, r: int, info: Dict[str, object]) -> None:
    M = int(info["M"])
    mk_mean = float(info["mk_mean"])
    cnot_step = int(info["cnot_step"])
    oneq_step = int(info["oneq_step"])
    depth_step = int(info["depth_step"])

    cnot_total = cnot_step * int(r)
    oneq_total = oneq_step * int(r)
    depth_total = depth_step * int(r)

    tex = rf"""\begin{{table}}[t]
\centering
\caption{{Resource summary for {system_name}.}}
\label{{tab:resources_{system_name.lower().replace(' ','_')}}}
\begin{{tabular}}{{lrrrrrr}}
\toprule
$n$ & $N$ & $r$ & $M$ & $\langle m_k\rangle$ & CNOT/step & Depth/step \\
\midrule
{n} & {N} & {r} & {M} & {mk_mean:.2f} & {cnot_step} & {depth_step} \\
\midrule
\multicolumn{{4}}{{l}}{{Totals over full evolution}} & & & \\
\multicolumn{{2}}{{l}}{{Total CNOT}} & \multicolumn{{5}}{{l}}{{{cnot_total}}} \\
\multicolumn{{2}}{{l}}{{Total 1q rotations}} & \multicolumn{{5}}{{l}}{{{oneq_total}}} \\
\multicolumn{{2}}{{l}}{{Estimated total depth}} & \multicolumn{{5}}{{l}}{{{depth_total}}} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(tex)
    print("Wrote:", path)

def make_r_list(r0: int, num: int = 8) -> List[int]:
    vals = np.unique(np.round(np.geomspace(max(5, int(r0) // 4), int(r0) * 2, num=int(num))).astype(int))
    return vals.tolist()
