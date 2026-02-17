from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict, Any

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Operator, SparsePauliOp

from .pauli_blocks import pauli_exp_gateblock

def pauli_decompose_terms(H: np.ndarray, coeff_tol: float = 1e-10) -> List[Tuple[str, float]]:
    """
    Returns list [(label, alpha)] for H = Σ alpha P(label).
    label format: rightmost char is qubit 0 (Qiskit convention).

    Ordering is exactly the ordering returned by Qiskit.
    """
    op = SparsePauliOp.from_operator(Operator(np.asarray(H, dtype=complex)))
    labels = op.paulis.to_labels()
    coeffs = op.coeffs

    terms: List[Tuple[str, float]] = []
    for lab, c in zip(labels, coeffs):
        if abs(c) > coeff_tol:
            if abs(c.imag) > 1e-10:
                print("Warning: non-negligible imag coeff:", lab, c)
            terms.append((lab, float(np.real(c))))
    return terms

def active_qubit_count(label: str) -> int:
    return sum(ch != "I" for ch in label)

def build_trotter_step_circuit(num_qubits: int, terms: List[Tuple[str, float]], dt: float) -> QuantumCircuit:
    """
    Builds a circuit for U_TS(dt) = Π_k exp(-i alpha_k dt P_k)
    using pauli_exp_gateblock for each term in the given order.
    """
    step = QuantumCircuit(int(num_qubits), name="U_TS(dt)")
    for lab, alpha in terms:
        theta = float(alpha) * float(dt)
        block = pauli_exp_gateblock(lab, theta)
        step.append(block.to_instruction(), list(range(int(num_qubits))))
    return step

def run_trotter_with_snapshots_aer(
    psi0: np.ndarray,
    step_circuit: QuantumCircuit,
    n_steps: int,
    stride: int = 1,
) -> Tuple[QuantumCircuit, List[str], np.ndarray]:
    """
    Notebook-faithful execution path:
      initialize(psi0)
      save_statevector sv_0
      repeat: append(step_circuit) n_steps times
      save_statevector every 'stride' steps (and final)
    Returns: qc, labels, snapshots
    """
    n = step_circuit.num_qubits
    backend = AerSimulator(method="statevector")

    step_inst = step_circuit.to_instruction()

    qc = QuantumCircuit(n)
    qc.initialize(np.asarray(psi0, dtype=complex), qc.qubits)
    qc.save_statevector(label="sv_0")
    labels = ["sv_0"]

    for k in range(1, int(n_steps) + 1):
        qc.append(step_inst, qc.qubits)
        if (k % int(stride)) == 0 or k == int(n_steps):
            lbl = f"sv_{k}"
            qc.save_statevector(label=lbl)
            labels.append(lbl)

    tqc = transpile(qc, backend, optimization_level=0)
    result = backend.run(tqc).result()
    data = result.data(0)

    snapshots = [np.asarray(data[lbl], dtype=complex) for lbl in labels]
    return qc, labels, np.array(snapshots)

def run_trotter_with_snapshots_operator(
    psi0: np.ndarray,
    step_circuit: QuantumCircuit,
    n_steps: int,
    stride: int = 1,
) -> Tuple[List[str], np.ndarray]:
    """
    Fast, deterministic execution path (recommended):
    compute U_step = Operator(step_circuit).data, then iterate psi <- U_step psi.

    This follows the exact same circuit construction as the notebook,
    but avoids building/simulating a huge circuit with r copies.
    """
    U_step = Operator(step_circuit).data
    psi = np.asarray(psi0, dtype=complex)
    psi = psi / np.linalg.norm(psi)

    labels: List[str] = ["sv_0"]
    snaps: List[np.ndarray] = [psi.copy()]

    for k in range(1, int(n_steps) + 1):
        psi = U_step @ psi
        psi = psi / np.linalg.norm(psi)
        if (k % int(stride)) == 0 or k == int(n_steps):
            labels.append(f"sv_{k}")
            snaps.append(psi.copy())

    return labels, np.array(snaps)

def fidelity_curve_from_snapshots(
    states_ref: np.ndarray,
    labels: List[str],
    snapshots: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    states_ref: array shape (r+1, N) at every step k=0..r from eigen-reference
    labels/snapshots: only at k in {0, stride, 2*stride, ..., r}
    Returns: k_indices, fidelities
    """
    k_list: List[int] = []
    F_list: List[float] = []
    for lbl, psi_trot in zip(labels, snapshots):
        k = int(str(lbl).split("_")[-1])
        k_list.append(k)

        psi_ref = np.asarray(states_ref[k], dtype=complex)
        inner = np.vdot(psi_ref, np.asarray(psi_trot, dtype=complex))
        F_list.append(float(np.abs(inner) ** 2))
    return np.array(k_list, dtype=int), np.array(F_list, dtype=float)

def infidelity_vs_r(
    H: np.ndarray,
    psi0: np.ndarray,
    tmax: float,
    r_list: List[int],
    coeff_tol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Final-time infidelity sweep vs r (fixed N).
    Matches your notebook sweep logic:
      - reference final state from eigen-expansion
      - terms fixed, dt changes
      - build step circuit, take Operator(step), apply r times
    """
    from .reference import reference_state_at_t

    H = np.asarray(H, dtype=complex)
    psi0 = np.asarray(psi0, dtype=complex)
    psi0 = psi0 / np.linalg.norm(psi0)

    N = H.shape[0]
    nq = int(np.log2(N))

    psi_ref = reference_state_at_t(H, psi0, float(tmax))

    terms = pauli_decompose_terms(H, coeff_tol=float(coeff_tol))
    M = len(terms)
    mk = np.array([active_qubit_count(lab) for lab, _ in terms], dtype=int)

    def cnot_cost_per_term(m: int) -> int:
        return 0 if m <= 1 else 2 * (m - 1)

    cnot_step = int(np.sum([cnot_cost_per_term(int(m)) for m in mk]))

    infids: List[float] = []
    cnot_totals: List[int] = []

    for r in r_list:
        r = int(r)
        dt = float(tmax) / r

        step_circ = build_trotter_step_circuit(nq, terms, dt)
        U_step = Operator(step_circ).data

        psi = psi0.copy()
        for _ in range(r):
            psi = U_step @ psi
        psi = psi / np.linalg.norm(psi)

        F = float(np.abs(np.vdot(psi_ref, psi)) ** 2)
        infids.append(1.0 - F)
        cnot_totals.append(r * cnot_step)

    info = {"terms": terms, "M": M, "mk": mk, "cnot_step": cnot_step}
    return np.array(infids, dtype=float), np.array(cnot_totals, dtype=int), info

def make_r_list(r_min: int, r_max: int, num: int = 8) -> List[int]:
    vals = np.unique(np.round(np.geomspace(int(r_min), int(r_max), num=int(num))).astype(int))
    vals[0] = max(1, int(vals[0]))
    return vals.tolist()
