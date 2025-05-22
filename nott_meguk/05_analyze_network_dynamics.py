"""Post-hoc analysis of resting-state network dynamics."""

# Import packages
import os
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import trange
from osl_dynamics.inference import modes
from utils import analysis as ua
from utils import plotting as up
from utils.data import load, match_order


if __name__ == "__main__":
    # -------------- [1] Settings -------------- #
    print("Step 1: Setting up ...")

    # Set directory paths
    BASE_DIR = "/well/woolrich/users/olt015/Cho2025_DyNeStE/nott_meguk"
    DATA_DIR = os.path.join(BASE_DIR, "results/full/{0}/run{1}")
    FIG_DIR = os.path.join(BASE_DIR, "figures")
    os.makedirs(FIG_DIR, exist_ok=True)

    # Set user-defined parameters
    best_dyneste_run = 2
    best_hmm_run = 2

    # Set model directories (with best model runs)
    dyneste_dir = DATA_DIR.format("dyneste", best_dyneste_run)
    hmm_dir = DATA_DIR.format("hmm", best_hmm_run)
    
    # Load inferred parameters
    dyneste_inf_params = load(os.path.join(dyneste_dir, "inference/inf_params.pkl"))
    dyneste_alphas = dyneste_inf_params["alpha"]
    dyneste_covs = dyneste_inf_params["covariance"]

    hmm_inf_params = load(os.path.join(hmm_dir, "inference/inf_params.pkl"))
    hmm_alphas = hmm_inf_params["alpha"]
    hmm_covs = hmm_inf_params["covariance"]

    # Match state orders between two runs
    order = match_order(
        ref_info=("full", "dyneste", best_dyneste_run),
        cmp_info=("full", "hmm", best_hmm_run),
        method="covariances",
    )
    
    if order is not None:
        print("Reordering states ...")
        print(f"\tOrder: {order}")
        hmm_alphas = [a[:, order] for a in hmm_alphas]
        hmm_covs = hmm_covs[order]

    # Get state time courses
    dyneste_stc = modes.argmax_time_courses(dyneste_alphas)
    hmm_stc = modes.argmax_time_courses(hmm_alphas)

    # Get data dimensions
    n_subjects = len(dyneste_stc)
    n_states = dyneste_stc[0].shape[1]

    # -------------- [2] Network Dynamics Feature Computations -------------- #
    print("Step 2: Computing network dynamics features ...")

    # Compute summary statistics
    dyneste_fo, dyneste_lt, dyneste_intv, dyneste_sr = ua.calculate_summary_stats(
        dyneste_stc, sampling_frequency=250
    )
    hmm_fo, hmm_lt, hmm_intv, hmm_sr = ua.calculate_summary_stats(
        hmm_stc, sampling_frequency=250
    )
    # shape: (n_subjects, n_states)

    summary_stats = {
        "fo": (dyneste_fo, hmm_fo),
        "lt": (dyneste_lt, hmm_lt),
        "intv": (dyneste_intv, hmm_intv),
        "sr": (dyneste_sr, hmm_sr),
    }

    # Compute a group-level state-wise correlation matrix
    statewise_correlations = np.zeros((n_states, n_states))
    for n in trange(n_subjects):
        statewise_correlations += ua.compute_statewise_correlation(
            dyneste_stc[n], hmm_stc[n]
        )
    statewise_correlations /= n_subjects  # normalize by the number of subjects

    # Compute a group-level state-wise Riemannian distance matrix
    riemannian_distances = ua.compute_statewise_riemannian_distance(
        dyneste_covs, hmm_covs
    ) # shape: (n_states, n_states)

    # Compute a subject-level state-wise dice coefficients
    dice_coeffs = ua.compute_dice_coefficients(dyneste_stc, hmm_stc)
    # shape: (n_subjects,)
    print(f"Mean Dice Coefficient: {np.mean(dice_coeffs):.3f} ± {np.std(dice_coeffs):.3f}")

    # -------------- [3] Visualization -------------- #
    print("Step 3: Visualizing results ...")

    # Plot summary statistics
    stat_names = [
        "Fractional Occupancy",
        "Mean Lifetime (s)",
        "Mean Interval (s)",
        "Switching Rates (/s)",
    ]
    for n, (key, values) in enumerate(summary_stats.items()):
        # Construct a dataframe
        df = pd.DataFrame({
            f"{stat_names[n]}": np.vstack(values).ravel(),
            "Models": np.repeat(["DyNeStE", "HMM"], n_subjects * n_states),
            "States": np.tile(np.arange(n_states) + 1, n_subjects * 2),
        })

        # Plot grouped violins
        up.plot_violin(
            df, x_var="States", y_var=f"{stat_names[n]}",
            hue_var="Models", palette={"DyNeStE": "#6EB5C0", "HMM": "#B6443F"},
            ylim=None, figsize=(10, 6), fontsize=18,
            filename=os.path.join(FIG_DIR, f"summary_stats_{key}.png"),
        )

    # Plot a state-wise correlation matrix
    up.plot_statewise_matrix(
        statewise_correlations,
        colormap=sns.color_palette("crest_r", as_cmap=True),
        axis_labels=["HMM States", "DyNeStE States"],
        cbar_label="Correlation",
        filename=os.path.join(FIG_DIR, "stc_correlation.png"),
    )

    # Plot a state-wise Riemannian distance matrix
    up.plot_statewise_matrix(
        riemannian_distances,
        colormap=sns.color_palette("flare_r", as_cmap=True),
        axis_labels=["HMM States", "DyNeStE States"],
        cbar_label="Riemannian Distance",
        filename=os.path.join(FIG_DIR, "stc_riemannian_distance.png"),
    )

    print("Analysis complete.")
