# Trotterization notebook. Converted to a multi-file Python project

This project is a direct translation of the logic in your notebook:
- Build infinite well and harmonic oscillator Hamiltonians on an N=2^n grid
- Build Gaussian initial states
- Compute the reference dynamics via eigenstate expansion
- Construct the Pauli exponential gadget and build the first-order product-formula Trotter step
- Compare fidelity vs time, snapshot overlays, and final-time error vs r
- Produce resource accounting (CNOT count, 1q rotations, depth estimate) and LaTeX tables
- Produce the HO expectation-value plot ⟨x⟩(t)

All outputs go to:
- figures/
- tables/
- data/

## Setup (fresh VS Code folder)

Create a venv:

```bash
python -m venv .venv
```

Activate it:

Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

Windows cmd:
```bat
.\.venv\Scripts\activate.bat
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Pinned versions (for reproducibility):
- qiskit==2.3.0
- qiskit-aer==0.17.2
- scipy==1.17.0

## Run everything

```bash
python run_all.py
```

## About --use-aer

Your notebook saves statevectors using Aer (save_statevector). For large r (your well uses r=8000), building a giant circuit can get slow.

By default, run_all.py uses the exact same circuit construction for the Trotter step, then evaluates it as a matrix once (U_step = Operator(step).data) and propagates:

```text
|psi_{k+1}> = U_step |psi_k>
```

This is mathematically identical to repeating the step circuit r times and gives the same snapshots and fidelities, but runs much faster.

If you want the notebook-faithful Aer path anyway:

```bash
python run_all.py --use-aer
```

## What to expect in figures/

You should see (names match the notebook logic):
- well_fidelity_N32_r8000.pdf
- ho_fidelity_N32_r40.pdf
- well_snapshots_N32_r8000_vertical.pdf
- ho_snapshots_N32_r40_vertical.pdf
- well_error_vs_r_N32.pdf, acc_vs_resources_well_N32.pdf
- ho_error_vs_r_N32.pdf, acc_vs_resources_ho_N32.pdf
- pauli_stats_well_N32_ho_N32.pdf
- resources_vs_r_well_N32_ho_N32.pdf
- ho_xexp.png

and the initial density plots:
- well_initial.png
- ho_initial.png

## Config

Default config:
- config/default_config.json

## File map

- run_all.py . entrypoint that runs the full pipeline
- trotter_sim/systems.py . builds Hamiltonians and initial states
- trotter_sim/reference.py . eigenstate expansion reference simulation
- trotter_sim/pauli_blocks.py . Pauli exponential gadget (with the Y-rotation sign fix)
- trotter_sim/trotter.py . Pauli decomposition, Trotter step, snapshots, fidelity, error-vs-r sweep
- trotter_sim/resources.py . resource accounting plus LaTeX tables
- trotter_sim/plotting.py . all plots (saved to disk, no interactive windows)
