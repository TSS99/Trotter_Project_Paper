# Circuit Level Trotterization for Discretized 1D Schrödinger Dynamics

Author: **Tilock Sadhukahn**  
LinkedIn: https://www.linkedin.com/in/tilock-sadhukhan/  
License: MIT


---

## What this repository does

This project is a reproducible pipeline for simulating 1D quantum dynamics on an $n$ qubit register by:

1. discretizing continuous space on a uniform grid with $N = 2^n$ points
2. encoding the sampled wavefunction on qubits using computational basis states $|j\rangle$
3. building finite difference Hamiltonians for the infinite well and the harmonic oscillator
4. expanding the discretized Hamiltonian in the $n$ qubit Pauli basis
5. implementing real time evolution with a first order (Lie) product formula
6. validating against a reference evolution computed in the same discretized Hilbert space
7. connecting accuracy to circuit resources such as Pauli term statistics, CNOT counts, and depth proxies

The main entrypoint is `run_all.py`. It generates figures, tables, and reference data in one run.

---

## Quickstart

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS or Linux:
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline

```bash
python run_all.py
```

Optional flags:

- `--use-aer` runs the repeated Trotter circuit on Aer statevector snapshots. Default uses Operator(step) propagation but follows the same circuit construction.
- `--no-pauli-tests` skips the Pauli block verification tests.

---

## What gets produced

After `python run_all.py`, outputs are written to:

- `figures/` (densities, fidelities, overlays, resource plots, and $\langle x\rangle(t)$ for the harmonic oscillator)
- `tables/` (LaTeX resource tables)
- `data/` (reference trajectories saved as NPZ files)

---

## Mathematical model and encoding

GitHub renders LaTeX math in Markdown. Use `$...$` for inline math and either `$$...$$` or a fenced `math` code block for block math.

### Schrödinger evolution

```math
i\hbar\,\frac{\partial}{\partial t}\,|\psi(t)\rangle = \hat{H}\,|\psi(t)\rangle
```

For time independent $\hat{H}$:

```math
|\psi(t)\rangle = e^{-i\hat{H}t/\hbar}\,|\psi(0)\rangle
```

### Grid to qubits mapping

Choose $N = 2^n$ grid points so the discretized state fits on $n$ qubits.

```math
|\psi\rangle \approx \sum_{j=0}^{N-1} \psi_j\,|j\rangle,
\qquad N = 2^n
```

Here $|j\rangle$ is the computational basis state whose integer value is the grid index $j$.

### Spatial grids used in this repo

Infinite well uses interior points (Dirichlet walls enforced by construction):

```math
x_j = (j+1)\,\Delta x,
\qquad \Delta x = \frac{L}{N+1},
\qquad j = 0,1,\dots,N-1
```

Harmonic oscillator uses an equally spaced grid on $[-x_{\max}, x_{\max})$ (endpoint excluded):

```math
x_j = -x_{\max} + j\,\Delta x,
\qquad \Delta x = \frac{2x_{\max}}{N},
\qquad j = 0,1,\dots,N-1
```

---

## Finite difference Hamiltonians

### Second derivative stencil

```math
\left.\frac{d^2\psi}{dx^2}\right|_{x_j}
\approx
\frac{\psi_{j+1} - 2\psi_j + \psi_{j-1}}{\Delta x^2}
```

### Infinite well

Inside the well, the Hamiltonian is purely kinetic:

```math
\hat{H}_{\mathrm{well}} = \frac{\hat{p}^2}{2m}
\approx -\frac{\hbar^2}{2m}\,D^{(2)}
```

### Harmonic oscillator

In the dimensionless units used in the code ($\hbar=m=\omega=1$):

```math
\hat{H}_{\mathrm{ho}}
=
-\frac{1}{2}\frac{d^2}{dx^2} + \frac{x^2}{2}
\approx
-\frac{1}{2}\,D^{(2)} + \mathrm{diag}\!\left(\frac{x_j^2}{2}\right)
```

---

## Reference evolution in the discretized Hilbert space

Reference trajectories are computed via eigen decomposition of the discretized Hamiltonian and phase evolution in that basis.

If $\hat{H}|\phi_\ell\rangle = E_\ell |\phi_\ell\rangle$ and $c_\ell = \langle \phi_\ell|\psi(0)\rangle$, then:

```math
|\psi_{\mathrm{ref}}(t)\rangle
=
\sum_{\ell=0}^{N-1} c_\ell\,e^{-iE_\ell t}\,|\phi_\ell\rangle
```

---

## Pauli basis expansion and first order Trotterization

### Pauli decomposition

The discretized Hamiltonian matrix is expanded in the $n$ qubit Pauli basis:

```math
H = \sum_{k=1}^{M}\alpha_k\,P_k,
\qquad
P_k \in \{I,X,Y,Z\}^{\otimes n}
```

### First order product formula

A single Trotter step is:

```math
U_{TS}(\Delta t)
=
\prod_{k=1}^{M}\exp\!\left(-i\,\alpha_k\,\Delta t\,P_k\right)
```

The total time is $t=r\Delta t$ and:

```math
U(t)=e^{-iHt}
\approx
\left(U_{TS}(\Delta t)\right)^r
```

---

## Gate block for a Pauli string exponential

Each term $\exp(-i\theta P)$ is compiled into a readable gate block:

1. basis changes mapping $X$ or $Y$ to $Z$
2. a parity gadget built from CNOTs onto a target qubit
3. a single $R_z(2\theta)$ on the target
4. uncompute parity and undo basis changes

Define the basis change maps:

```math
B(X)=R_y(-\pi/2),\qquad
B(Y)=R_x(\pi/2),\qquad
B(Z)=I
```

Let $P=\bigotimes_{q\in\mathcal{A}}\sigma_q$ where $\mathcal{A}$ are the active qubits (non identity letters). The compiled unitary is:

```math
e^{-i\theta P}
=
\left(\prod_{q\in\mathcal{A}} B(\sigma_q)^\dagger\right)
\cdot
U_{\mathrm{parity}}^\dagger\,
R_z(2\theta)\,
U_{\mathrm{parity}}
\cdot
\left(\prod_{q\in\mathcal{A}} B(\sigma_q)\right)
```

The parity gadget $U_{\mathrm{parity}}$ is implemented as a chain of CNOTs from all active qubits onto the last active qubit as the target, then uncomputed in reverse.

---

## Metrics used in the comparisons

### Fidelity

```math
F(t) = \left|\langle \psi_{\mathrm{ref}}(t)\,|\,\psi_{\mathrm{trot}}(t)\rangle\right|^2
```

Infidelity:

```math
\varepsilon(t) = 1 - F(t)
```

### Harmonic oscillator expectation value

```math
\langle x\rangle(t) = \sum_{j=0}^{N-1} x_j\,|\psi_j(t)|^2
```

---

## Resource accounting model

For each Pauli term $P_k$:

- $m_k$ is the number of non identity letters in $P_k$ (active qubits)
- $s_k$ is the number of letters in $\{X,Y\}$ in $P_k$ (basis changes required)

### Per term gate counts

CNOT count for one parity gadget:

```math
N_{\mathrm{CNOT}}(P_k)
=
\begin{cases}
0, & m_k \le 1 \\
2(m_k-1), & m_k \ge 2
\end{cases}
```

Single qubit rotation proxy:

```math
N_{1q}(P_k) = 2s_k + 1
```

Depth proxy (as implemented in this repo). Define $b_k = 1$ if $s_k>0$ else $0$.

```math
D(P_k) =
b_k
+
\begin{cases}
0, & m_k \le 1 \\
2(m_k-1), & m_k \ge 2
\end{cases}
+
1
+
b_k
```

### Per step and total resources

For one Trotter step:

```math
N_{\mathrm{CNOT}}^{(\mathrm{step})} = \sum_{k=1}^{M} N_{\mathrm{CNOT}}(P_k),
\qquad
N_{1q}^{(\mathrm{step})} = \sum_{k=1}^{M} N_{1q}(P_k),
\qquad
D^{(\mathrm{step})} = \sum_{k=1}^{M} D(P_k)
```

Total resources scale linearly with $r$:

```math
N_{\mathrm{CNOT}}^{(\mathrm{total})} = r\,N_{\mathrm{CNOT}}^{(\mathrm{step})},
\qquad
D^{(\mathrm{total})} = r\,D^{(\mathrm{step})}
```

---

## Configuration

The pipeline reads a JSON config (default: `config/default_config.json`) with top level keys `well` and `ho`.

For each system, the required keys are:

- `n_discretize` (must be a power of two, at least 4)
- `n_steps` (Trotter step count $r$)
- `total_time` (final time $t$)

System specific keys:

- Infinite well: `L`, `x0`, `sigma`, `k0`, and optional `enforce_walls`
- Harmonic oscillator: `x_max`, `x0`, `sigma`, `k0`

---

## Project structure

```text
.
├── run_all.py
├── requirements.txt
├── trotter_sim/
│   ├── __init__.py
│   ├── utils.py
│   ├── systems.py
│   ├── reference.py
│   ├── pauli_blocks.py
│   ├── trotter.py
│   ├── resources.py
│   └── plotting.py
├── config/
│   └── default_config.json
├── figures/
├── tables/
└── data/
```

---

## License

MIT License. See `LICENSE`.

## Contact

Tilock Sadhukahn  
https://www.linkedin.com/in/tilock-sadhukhan/
