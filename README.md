# Circuit-Level Trotterization for Discretized 1D Schrödinger Dynamics

Author: **Tilock Sadhukahn**  
LinkedIn: https://www.linkedin.com/in/tilock-sadhukhan/  
License: MIT

## What this repository contains

This repo implements a complete, reproducible pipeline for simulating **1D quantum dynamics** on an **n-qubit register** by:

1. Discretizing position on a uniform grid with **N = 2^n** points and encoding the sampled wavefunction as amplitudes of $|j\rangle$.
2. Building discretized Hamiltonians (finite difference) for:
   - the infinite potential well
   - the harmonic oscillator
3. Expanding the discretized Hamiltonian in the **n-qubit Pauli basis**.
4. Implementing real-time evolution using a **first-order (Lie) product formula** (Trotterization).
5. Compiling each Pauli-string exponential into an explicit, readable gate block.
6. Computing an **exact reference evolution** in the same discretized Hilbert space via eigen-decomposition.
7. Comparing reference vs circuit evolution using fidelity and producing resource summaries (Pauli-term statistics, CNOT counts, depth proxies) plus plots.

The goal is to make the accuracy vs resources tradeoffs visible at the circuit level while keeping the physics and numerics consistent end to end.

## Core equations (GitHub renders these)

Schrödinger evolution:
$$
i\hbar\,\frac{\partial}{\partial t}|\psi(t)\rangle = \hat{H}|\psi(t)\rangle,
\qquad
|\psi(t)\rangle = e^{-i\hat{H}t/\hbar}|\psi(0)\rangle
$$

Grid encoding with $N=2^n$:
$$
x_j = x_{\min} + j\,\Delta x,\quad j=0,\dots,N-1
$$
$$
|\psi\rangle \approx \sum_{j=0}^{N-1} \psi(x_j)\,|j\rangle
$$

Pauli decomposition of the discretized Hamiltonian:
$$
H = \sum_{k=1}^{M} \alpha_k P_k,
\qquad
P_k \in \{I,X,Y,Z\}^{\otimes n}
$$

First-order product formula (Trotterization):
$$
U(t)=e^{-iHt}\approx\left(\prod_{k=1}^{M} e^{-i\alpha_k P_k\Delta t}\right)^{r},
\qquad
\Delta t = \frac{t}{r}
$$

Fidelity against the reference evolution:
$$
F(t) = \left|\langle \psi_{\text{ref}}(t)\,|\,\psi_{\text{trot}}(t)\rangle\right|^2
$$

## Project structure

```text
.
├── run_all.py                 # Main entrypoint. Runs both systems and generates artifacts
├── requirements.txt
├── trotter_sim/
│   ├── __init__.py
│   ├── utils.py               # config dataclasses + validation + helpers
│   ├── systems.py             # discretized Hamiltonians + initial states
│   ├── reference.py           # eigen-expansion reference evolution (saves NPZ)
│   ├── pauli_blocks.py        # exp(-iθP) gate block implementation + quick tests
│   ├── trotter.py             # Pauli decomposition + step-unitary + fidelity
│   ├── resources.py           # CNOT/depth proxies + Pauli-term stats
│   └── plotting.py            # non-interactive plotting helpers (saves to figures/)
├── config/
│   └── default_config.json    # parameters for both systems
└── scripts/                   # small helpers (optional)
```

## Run in a fresh VS Code folder

### 1) Create a virtual environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Run the full pipeline
```bash
python run_all.py
```

The run produces (created if missing):
- `data/` with reference trajectories saved as `ref_eigen_*.npz`
- `figures/` with plots (densities, fidelity curves, overlays, resource plots)
- `tables/` with text or CSV summaries when enabled by the scripts

## Configuration

Edit `config/default_config.json` to control:
- `n_qubits` (implied by `N = 2^n` or specified directly depending on the config)
- `n_steps` (Trotter steps $r$)
- `total_time` (final time)
- system parameters (well length, HO domain cutoff, Gaussian packet parameters)

## Notes on reproducibility

- Both reference and circuit evolution live in the same discretized Hilbert space. That keeps comparisons clean.
- Each Pauli exponential $e^{-i\theta P}$ is implemented with explicit basis changes and a parity gadget.
- The included quick tests in `pauli_blocks.py` validate the gate block against a matrix exponential for random cases.

## License

This repository is released under the MIT License. See `LICENSE`.
