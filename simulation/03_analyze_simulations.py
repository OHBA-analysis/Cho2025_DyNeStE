"""Script for analyzing model ability to learn long-range dependencies."""

# Import packages
import os
import numpy as np
from scipy.stats import ks_2samp
from osl_dynamics.analysis import modes
from utils import validate_nd_arrays
from utils import data as ud
from utils import plotting as up


if __name__ == "__main__":
    # -------------- [1] Settings -------------- #
    print("Step 1: Setting up ...")

    # Set user-defined parameters
    dyneste_run_ids = 0
    hmm_run_ids = 0

    # Validate user inputs
    if dyneste_run_ids != hmm_run_ids:
        raise ValueError(
            "Run IDs for DyNeStE and HMM must match to enable a valid comparison."
        )

    # Set directories
    BASE_DIR = "/well/woolrich/users/olt015/Cho2025_DyNeStE/simulation"
    MODEL_DIR = os.path.join(BASE_DIR, "results")
    FIG_DIR = os.path.join(BASE_DIR, "figures")
    os.makedirs(FIG_DIR, exist_ok=True)

    # Set colors for visualization
    cmap = ["#E69F00", "#56B4E9", "#009E73"]  # Okabe-Ito color palette

    # -------------- [2] Single Simulation -------------- #
    print("Step 2: Analyzing a representative simulation ...")

    # Load inferred parameters
    dyneste_inf_params = ud.load_inf_params(MODEL_DIR, "dyneste", 0)
    hmm_inf_params = ud.load_inf_params(MODEL_DIR, "hmm", 0)

    # Get state time courses
    dyneste_sim_stc = dyneste_inf_params["sim_stc"]
    dyneste_inf_stc = dyneste_inf_params["inf_stc"]
    dyneste_sam_stc = dyneste_inf_params["sam_stc"]
    hmm_sim_stc = hmm_inf_params["sim_stc"]
    hmm_inf_stc = hmm_inf_params["inf_stc"]
    hmm_sam_stc = hmm_inf_params["sam_stc"]
    # *_stc.shape = (n_samples, n_states)

    # Get dice coefficients
    dices = [
        dyneste_inf_params["dice_coefficient"],
        hmm_inf_params["dice_coefficient"],
    ]

    # Get state-specific covariance matrices
    dyneste_sim_cov = dyneste_inf_params["sim_cov"]
    dyneste_inf_cov = dyneste_inf_params["inf_cov"]
    hmm_sim_cov = hmm_inf_params["sim_cov"]
    hmm_inf_cov = hmm_inf_params["inf_cov"]
    # *_cov.shape = (n_states, n_channels, n_channels)

    # Validate simulation data
    validate_nd_arrays(dyneste_sim_stc, hmm_sim_stc)
    validate_nd_arrays(dyneste_sim_cov, hmm_sim_cov)
    sim_stc = dyneste_inf_params["sim_stc"]
    sim_cov = dyneste_inf_params["sim_cov"]
    n_states = sim_stc.shape[-1]

    # Plot state time courses
    print("(Step 2-1) Plotting state time courses ...")

    fig, axes = up.plot_alpha(
        sim_stc,
        dyneste_inf_stc,
        hmm_inf_stc,
        n_samples=4000,
        plot_kwargs={"alpha": 0.75},
        colors=cmap,
    )
    titles = ["Ground Truth", "DyNeStE", "HMM"]
    axes[-1].set_xlabel("Time (samples)", fontsize=14)
    for i, ax in enumerate(axes):
        title = f"{titles[i]}"
        if i > 0:
            title += f" (Dice Sørensen Coefficient: {dices[i - 1]:.4f})"
        ax.set_ylabel("Probabilities", fontsize=14)
        ax.set_title(title, fontsize=14)
        ax.tick_params(axis="both", which="both", labelsize=14)
    up.save(fig, os.path.join(FIG_DIR, "state_time_courses.png"))

    # Plot covariance matrices
    print("(Step 2-2) Plotting covariance matrices ...")

    covs = np.concatenate([
        sim_cov, dyneste_inf_cov, hmm_inf_cov,
    ], axis=0)
    # covs.shape = (n_total_states, n_channels, n_channels)

    fig, _ = up.plot_matrices(covs, cbar_label="Covariance")
    up.save(fig, os.path.join(FIG_DIR, "covariance_matrices.png"))

    # Plot state lifetime distributions
    print("(Step 2-3) Plotting covariance matrices ...")

    stcs = [sim_stc, dyneste_inf_stc, hmm_inf_stc, dyneste_sam_stc, hmm_sam_stc]
    stc_names = ["sim", "dyneste_inf", "hmm_inf", "dyneste_sam", "hmm_sam"]

    for stc, name in zip(stcs, stc_names):
        filename = os.path.join(FIG_DIR, f"{name}_lt.png")
        up.plot_state_lifetime_dist(
            stc,
            colors=cmap,
            gamma_shape=10,
            gamma_scale=5,
            filename=filename,
        )

    # Perform statistical testing on state lifetime distributions
    print("(Step 2-4) Two-Sample Kolmogorov-Smirnov test for goodness of fit")

    bonferroni_n_tests = n_states * 4
    alpha_thr = 0.05 / bonferroni_n_tests
    print(f"\tBonferroni-corrected alpha threshold: {alpha_thr:.4e}")

    sim_lt = modes.lifetimes(sim_stc)
    for name, stc in zip(stc_names[1:], stcs[1:]):
        print(f"\tComparing '{name}' with simulated data ...")
        stc_lt = modes.lifetimes(stc)
        # stc_lt.shape = (n_states, n_activations)
        for i, lt in enumerate(stc_lt):
            res = ks_2samp(sim_lt[i], lt, method="auto")
            print(
                f"\t\tState {i + 1}: test statistic = {res.statistic:.4f}, "
                f"p-value = {res.pvalue:.4e}, significance = {res.pvalue < alpha_thr}"
            )

    print("Analysis complete.")
