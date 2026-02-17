from __future__ import annotations

import argparse
import os
import numpy as np

from trotter_sim.utils import RunConfig, validate_common
from trotter_sim import systems, reference, pauli_blocks, trotter, plotting, resources

def build_systems(cfg: RunConfig):
    # Validate commons
    Nw, rw, tw = validate_common(cfg.well)
    Nh, rh, th = validate_common(cfg.ho)

    # Infinite well
    L = float(cfg.well["L"])
    H_well, x_well, dx_well = systems.build_infinite_well_hamiltonian(Nw, L=L)
    psi0_well = systems.gaussian_wavepacket_well(
        x_well,
        x0=float(cfg.well["x0"]),
        sigma=float(cfg.well["sigma"]),
        k0=float(cfg.well["k0"]),
        enforce_walls=bool(cfg.well.get("enforce_walls", True)),
    )
    systems.sanity_check(H_well, psi0_well, "Infinite well")

    # Harmonic oscillator
    x_max = float(cfg.ho["x_max"])
    H_ho, x_ho, dx_ho = systems.build_ho_hamiltonian(Nh, x_max=x_max)
    psi0_ho = systems.gaussian_wavepacket_ho(
        x_ho,
        x0=float(cfg.ho["x0"]),
        sigma=float(cfg.ho["sigma"]),
        k0=float(cfg.ho["k0"]),
    )
    systems.sanity_check(H_ho, psi0_ho, "Harmonic oscillator")

    return (H_well, x_well, psi0_well, Nw, rw, tw), (H_ho, x_ho, psi0_ho, Nh, rh, th)

def main():
    parser = argparse.ArgumentParser(description="Reproduce the notebook pipeline as a multi-file Python project.")
    parser.add_argument("--config", default="config/default_config.json", help="Path to config JSON.")
    parser.add_argument(
        "--use-aer",
        action="store_true",
        help="Run the large Trotter circuit using Aer statevector snapshots (not recommended for large r). "
             "Default uses Operator(step) propagation but follows the same circuit construction.",
    )
    parser.add_argument("--no-pauli-tests", action="store_true", help="Skip the Pauli block verification tests.")
    args = parser.parse_args()

    cfg = RunConfig.from_json(args.config)

    plotting.ensure_dirs("figures", "tables", "data")

    # 1) Build H and psi0 for both systems
    (H_well, x_well, psi0_well, Nw, rw, tw), (H_ho, x_ho, psi0_ho, Nh, rh, th) = build_systems(cfg)

    # 2) Save initial densities
    plotting.plot_initial_density(x_well, psi0_well, "Infinite well initial |psi(x,0)|^2", "figures/well_initial.png")
    plotting.plot_initial_density(x_ho, psi0_ho, "Harmonic oscillator initial |psi(x,0)|^2", "figures/ho_initial.png")

    # 3) Reference evolution (eigen expansion) + save
    tag_well = f"well_N{cfg.well['n_discretize']}_r{cfg.well['n_steps']}_t{cfg.well['total_time']}"
    times_well, probs_well_ref, states_well_ref, dt_well, _ = reference.save_reference_run_eigen(
        tag_well, x_well, H_well, psi0_well, cfg.well["total_time"], cfg.well["n_steps"]
    )

    tag_ho = f"ho_N{cfg.ho['n_discretize']}_r{cfg.ho['n_steps']}_t{cfg.ho['total_time']}"
    times_ho, probs_ho_ref, states_ho_ref, dt_ho, _ = reference.save_reference_run_eigen(
        tag_ho, x_ho, H_ho, psi0_ho, cfg.ho["total_time"], cfg.ho["n_steps"]
    )

    # 4) Pauli exponential block tests
    if not args.no_pauli_tests:
        pauli_blocks.run_quick_tests()

    # 5) Trotter evolution + fidelity (well + ho)
    coeff_tol = 1e-14

    # WELL
    nq_well = int(np.log2(Nw))
    dt_well_step = float(cfg.well["total_time"]) / int(cfg.well["n_steps"])
    terms_well = trotter.pauli_decompose_terms(H_well, coeff_tol=coeff_tol)
    print("WELL:", "N =", Nw, "nq =", nq_well, "r =", rw, "dt =", dt_well_step)
    print("WELL:", "M =", len(terms_well))

    step_well = trotter.build_trotter_step_circuit(nq_well, terms_well, dt_well_step)

    if args.use_aer:
        _, labels_well, snaps_well = trotter.run_trotter_with_snapshots_aer(psi0_well, step_well, n_steps=rw, stride=1)
    else:
        labels_well, snaps_well = trotter.run_trotter_with_snapshots_operator(psi0_well, step_well, n_steps=rw, stride=1)

    k_idx_well, F_well = trotter.fidelity_curve_from_snapshots(states_well_ref, labels_well, snaps_well)
    t_well = k_idx_well * dt_well_step
    plotting.plot_fidelity(
        t_well, F_well,
        "Infinite well: fidelity vs time (eigen-reference vs Trotter circuit)",
        f"figures/well_fidelity_N{Nw}_r{rw}.pdf"
    )

    # HO
    nq_ho = int(np.log2(Nh))
    dt_ho_step = float(cfg.ho["total_time"]) / int(cfg.ho["n_steps"])
    terms_ho = trotter.pauli_decompose_terms(H_ho, coeff_tol=coeff_tol)
    print("\nHO:", "N =", Nh, "nq =", nq_ho, "r =", rh, "dt =", dt_ho_step)
    print("HO:", "M =", len(terms_ho))

    step_ho = trotter.build_trotter_step_circuit(nq_ho, terms_ho, dt_ho_step)

    if args.use_aer:
        _, labels_ho, snaps_ho = trotter.run_trotter_with_snapshots_aer(psi0_ho, step_ho, n_steps=rh, stride=1)
    else:
        labels_ho, snaps_ho = trotter.run_trotter_with_snapshots_operator(psi0_ho, step_ho, n_steps=rh, stride=1)

    k_idx_ho, F_ho = trotter.fidelity_curve_from_snapshots(states_ho_ref, labels_ho, snaps_ho)
    t_ho = k_idx_ho * dt_ho_step
    plotting.plot_fidelity(
        t_ho, F_ho,
        "Harmonic oscillator: fidelity vs time (eigen-reference vs Trotter circuit)",
        f"figures/ho_fidelity_N{Nh}_r{rh}.pdf"
    )

    # 6) Snapshot overlays (vertical)
    well_state_dict = plotting.labels_to_state_dict(labels_well, snaps_well)
    idx_well = plotting.choose_snapshot_indices(rw, num=4)
    plotting.plot_snapshots_overlay_vertical(
        x=x_well,
        probs_ref=probs_well_ref,
        state_dict_trot=well_state_dict,
        dt=dt_well_step,
        indices=idx_well,
        title=f"Infinite well snapshots (N={Nw}, r={rw})",
        outpath=f"figures/well_snapshots_N{Nw}_r{rw}_vertical.pdf",
        fig_width=4.6,
        row_height=2.35,
    )

    ho_state_dict = plotting.labels_to_state_dict(labels_ho, snaps_ho)
    idx_ho = plotting.choose_snapshot_indices(rh, num=4)
    plotting.plot_snapshots_overlay_vertical(
        x=x_ho,
        probs_ref=probs_ho_ref,
        state_dict_trot=ho_state_dict,
        dt=dt_ho_step,
        indices=idx_ho,
        title=f"Harmonic oscillator snapshots (N={Nh}, r={rh})",
        outpath=f"figures/ho_snapshots_N{Nh}_r{rh}_vertical.pdf",
        fig_width=4.6,
        row_height=2.35,
    )

    # 7) Error vs r sweeps + acc-vs-resource plots (matches notebook sweep logic)
    r0_well = int(cfg.well["n_steps"])
    r_list_well = trotter.make_r_list(max(10, r0_well // 4), r0_well * 2, num=8)
    inf_well, cnot_well, info_well_sweep = trotter.infidelity_vs_r(H_well, psi0_well, float(cfg.well["total_time"]), r_list_well, coeff_tol=1e-12)
    print("WELL sweep:", "CNOT/step =", info_well_sweep["cnot_step"], "| r_list =", r_list_well)
    plotting.plot_error_vs_r(r_list_well, inf_well, f"Infinite well: error vs r (N={Nw})", f"figures/well_error_vs_r_N{Nw}.pdf")
    plotting.plot_acc_vs_resources(cnot_well, inf_well, f"Infinite well: accuracy vs resources (N={Nw})", f"figures/acc_vs_resources_well_N{Nw}.pdf")

    r0_ho = int(cfg.ho["n_steps"])
    r_list_ho = trotter.make_r_list(max(10, r0_ho // 4), r0_ho * 2, num=8)
    inf_ho, cnot_ho, info_ho_sweep = trotter.infidelity_vs_r(H_ho, psi0_ho, float(cfg.ho["total_time"]), r_list_ho, coeff_tol=1e-12)
    print("HO sweep:", "CNOT/step =", info_ho_sweep["cnot_step"], "| r_list =", r_list_ho)
    plotting.plot_error_vs_r(r_list_ho, inf_ho, f"Harmonic oscillator: error vs r (N={Nh})", f"figures/ho_error_vs_r_N{Nh}.pdf")
    plotting.plot_acc_vs_resources(cnot_ho, inf_ho, f"Harmonic oscillator: accuracy vs resources (N={Nh})", f"figures/acc_vs_resources_ho_N{Nh}.pdf")

    # 8) Resource accounting + LaTeX tables
    coeff_tol_res = 1e-12
    terms_well_res = resources.pauli_decompose_terms(H_well, coeff_tol=coeff_tol_res)
    info_well = resources.summarize_terms(terms_well_res)
    print("WELL resources:",
          "N", Nw, "n", nq_well, "r", rw,
          "| M", info_well["M"],
          "| mk(min/mean/max)", info_well["mk_min"], f"{info_well['mk_mean']:.2f}", info_well["mk_max"],
          "| CNOT/step", info_well["cnot_step"],
          "| 1q/step", info_well["oneq_step"],
          "| depth/step", info_well["depth_step"])

    terms_ho_res = resources.pauli_decompose_terms(H_ho, coeff_tol=coeff_tol_res)
    info_ho = resources.summarize_terms(terms_ho_res)
    print("HO resources:",
          "N", Nh, "n", nq_ho, "r", rh,
          "| M", info_ho["M"],
          "| mk(min/mean/max)", info_ho["mk_min"], f"{info_ho['mk_mean']:.2f}", info_ho["mk_max"],
          "| CNOT/step", info_ho["cnot_step"],
          "| 1q/step", info_ho["oneq_step"],
          "| depth/step", info_ho["depth_step"])

    resources.plot_pauli_stats(info_well, info_ho, Nw, Nh, f"figures/pauli_stats_well_N{Nw}_ho_N{Nh}.pdf")

    r_list_well_res = resources.make_r_list(rw, num=8)
    r_list_ho_res = resources.make_r_list(rh, num=8)
    resources.plot_resources_vs_r(info_well, info_ho, r_list_well_res, r_list_ho_res, f"figures/resources_vs_r_well_N{Nw}_ho_N{Nh}.pdf")

    resources.write_resource_table_tex(f"tables/resources_well_N{Nw}.tex", "Infinite well", nq_well, Nw, rw, info_well)
    resources.write_resource_table_tex(f"tables/resources_ho_N{Nh}.tex", "Harmonic oscillator", nq_ho, Nh, rh, info_ho)

    # 9) HO expectation value <x>(t)
    plotting.plot_ho_xexp(x_ho, probs_ho_ref, dt_ho_step, labels_ho, snaps_ho, "figures/ho_xexp.png")

    print("\nDone. Outputs are in ./figures, ./tables, ./data")

if __name__ == "__main__":
    main()
