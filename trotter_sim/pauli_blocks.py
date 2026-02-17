from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, SparsePauliOp
from scipy.linalg import expm

# -----------------------------
# Pauli exponential block: exp(-i theta P)
# using basis-change + parity gadget + uncompute
# Qiskit label convention: rightmost char is qubit 0
# -----------------------------
def _basis_change_into_z(qc: QuantumCircuit, q: int, p: str) -> None:
    if p == "X":
        qc.ry(-np.pi / 2, q)
    elif p == "Y":
        qc.rx(np.pi / 2, q)  # Fixed in your notebook
    elif p in ("Z", "I"):
        pass
    else:
        raise ValueError(f"Invalid Pauli letter {p}")

def _basis_change_out_of_z(qc: QuantumCircuit, q: int, p: str) -> None:
    if p == "X":
        qc.ry(+np.pi / 2, q)
    elif p == "Y":
        qc.rx(-np.pi / 2, q)  # Fixed in your notebook
    elif p in ("Z", "I"):
        pass
    else:
        raise ValueError(f"Invalid Pauli letter {p}")

def pauli_exp_gateblock(label: str, theta: float) -> QuantumCircuit:
    """
    Returns a circuit implementing exp(-i theta P(label)) using:
      basis changes -> parity gadget -> Rz(2 theta) -> uncompute -> undo basis changes

    label like "IXYZ" with length n.
    Rightmost char corresponds to qubit 0.
    """
    n = len(label)
    qc = QuantumCircuit(n, name=f"exp(-iθ{label})")

    # Active qubits in increasing q order
    active = []
    for q in range(n):
        p = label[n - 1 - q]  # char for qubit q
        if p != "I":
            active.append(q)

    if len(active) == 0:
        return qc

    # Stage 1: basis changes
    for q in active:
        p = label[n - 1 - q]
        _basis_change_into_z(qc, q, p)

    # Stage 2: parity gadget onto last active qubit
    tgt = active[-1]
    if len(active) >= 2:
        for q in active[:-1]:
            qc.cx(q, tgt)

    qc.rz(2.0 * float(theta), tgt)

    if len(active) >= 2:
        for q in reversed(active[:-1]):
            qc.cx(q, tgt)

    # Stage 3: undo basis changes
    for q in reversed(active):
        p = label[n - 1 - q]
        _basis_change_out_of_z(qc, q, p)

    return qc

# -----------------------------
# Verification helpers (matches notebook tests)
# -----------------------------
def pauli_matrix_from_label(label: str) -> np.ndarray:
    return Operator(SparsePauliOp.from_list([(label, 1.0)])).data

def verify_pauli_block(label: str, theta: float, atol: float = 1e-10) -> float:
    qc = pauli_exp_gateblock(label, theta)
    U_circ = Operator(qc).data

    P = pauli_matrix_from_label(label)
    U_exact = expm(-1j * float(theta) * P)

    err = float(np.linalg.norm(U_circ - U_exact))
    print(f"label={label}, theta={theta:.4f}, ||U_circ - U_exact||_F = {err:.3e}")
    if err > atol:
        raise RuntimeError(f"Verification failed for {label}. Error {err} > {atol}")
    return err

def run_quick_tests() -> None:
    np.random.seed(0)
    tests = [("Z", 0.37), ("X", 0.21), ("Y", -0.4), ("XZ", 0.33), ("XYI", 0.17), ("IYZ", -0.28)]
    for lab, th in tests:
        verify_pauli_block(lab, th)
    print("All Pauli exponential block tests passed.")
