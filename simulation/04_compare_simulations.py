"""Script for assessing consistency across simulations."""

# Import packages
import os
import numpy as np
import pandas as pd
from osl_dynamics.analysis import modes
from utils import validate_nd_arrays, flatten_nested_data
from utils import analysis as ua
from utils import data as ud
from utils import plotting as up
from utils.statistics import stat_ind_two_samples


if __name__ == "__main__":
    # -------------- [1] Settings -------------- #
    print("Step 1: Setting up ...")

    # Set user-defined parameters
    dyneste_run_ids = list(range(10))
    hmm_run_ids = list(range(10))
    n_states = 3

    # Validate user inputs
    if len(dyneste_run_ids) != len(hmm_run_ids):
        raise ValueError("Run IDs for DyNeStE and HMM must be the same length.")
    if dyneste_run_ids != hmm_run_ids:
        raise ValueError(
            "Run IDs for DyNeStE and HMM must match to enable a valid comparison."
        )
    n_runs = len(dyneste_run_ids)

    # Set directories
    BASE_DIR = "/well/woolrich/users/olt015/Cho2025_DyNeStE/simulation"
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    FIG_DIR = os.path.join(BASE_DIR, "figures")
    os.makedirs(FIG_DIR, exist_ok=True)

    # Set colors for visualization
    cmap = ["#E69F00", "#56B4E9", "#009E73"]  # Okabe-Ito color palette

    # -------------- [2] Multiple Simulations -------------- #
    print("Step 2: Analyzing multiple simulations ...")

    # Define model types and their corresponding run IDs
    model_configs = {
        "dyneste": dyneste_run_ids,
        "hmm": hmm_run_ids,
    }
    model_types = list(model_configs.keys())

    # Initialize results
    results = {
        model: {
            "dice": [],
            "rv_coeffs": [],
            "lts": [],
            "jsd": {"inferred": [], "sampled": []},
        }
        for model in model_types
    }

    # Define functions
    def process_model_data(model_dir, model_type, run_id):
        """Loads inferred parameters and calculates metrics for a given model type and run ID."""

        # Load inferred parameters
        inf_params = ud.load_inf_params(model_dir, model_type, run_id)

        # Get Dice coefficient
        dice = inf_params["dice_coefficient"]

        # Get RV coefficients
        rv_coeffs = ua.calculate_rv_coefficients(
            inf_params["sim_cov"], inf_params["inf_cov"]
        )

        # Compute state lifetimes
        sim_lt = modes.lifetimes(inf_params["sim_stc"])
        inf_lt = modes.lifetimes(inf_params["inf_stc"])
        sam_lt = modes.lifetimes(inf_params["sam_stc"])
        # *_lt.shape = (n_states, n_activations)
        lts = [sim_lt, inf_lt, sam_lt]

        return {
            "dice": dice,
            "rv_coeffs": rv_coeffs,
            "lts": lts,
            "sim_stc": inf_params["sim_stc"],
        }

    # Process each model type for the current run index 'n'
    for n in range(n_runs):
        run_results = {}
        for model_type in model_types:
            # Collect model-specific metric values
            run_id = model_configs[model_type][n]
            model_results = process_model_data(MODEL_DIR, model_type, run_id)
            run_results[model_type] = model_results

            # Store metric values to the main storage
            results[model_type]["dice"].append(
                model_results["dice"]
            )  # shape: (n_runs,)
            results[model_type]["rv_coeffs"].append(
                model_results["rv_coeffs"]
            )  # shape: (n_runs, n_states)

            # Accumulate state lifetimes
            results[model_type]["lts"].append(
                model_results["lts"]
            )  # shape: (n_runs, n_states, n_activations)

        # Validate if the simulations are identical
        validate_nd_arrays(
            run_results["dyneste"]["sim_stc"],
            run_results["hmm"]["sim_stc"],
        )

    # Compute Jensen-Shannon distance
    flattened_lts = flatten_nested_data(
        [
            results["dyneste"]["lts"],
            results["hmm"]["lts"],
        ]
    )

    min_lt = np.min(flattened_lts)
    max_lt = np.max(flattened_lts)

    dyneste_sim_lts = [lts[0] for lts in results["dyneste"]["lts"]]
    dyneste_inf_lts = [lts[1] for lts in results["dyneste"]["lts"]]
    dyneste_sam_lts = [lts[2] for lts in results["dyneste"]["lts"]]
    hmm_sim_lts = [lts[0] for lts in results["hmm"]["lts"]]
    hmm_inf_lts = [lts[1] for lts in results["hmm"]["lts"]]
    hmm_sam_lts = [lts[2] for lts in results["hmm"]["lts"]]
    # *_lts.shape = (n_runs, n_states, n_activations)

    for n in range(n_runs):
        for s in range(n_states):
            # Compute Jensen-Shannon distance
            dyneste_jsd_inf = ua.calculate_js_distance(
                dyneste_sim_lts[n][s],
                dyneste_inf_lts[n][s],
                bounds=[min_lt, max_lt],
            )
            dyneste_jsd_sam = ua.calculate_js_distance(
                dyneste_sim_lts[n][s],
                dyneste_sam_lts[n][s],
                bounds=[min_lt, max_lt],
            )
            hmm_jsd_inf = ua.calculate_js_distance(
                hmm_sim_lts[n][s],
                hmm_inf_lts[n][s],
                bounds=[min_lt, max_lt],
            )
            hmm_jsd_sam = ua.calculate_js_distance(
                hmm_sim_lts[n][s],
                hmm_sam_lts[n][s],
                bounds=[min_lt, max_lt],
            )

            # Store results
            results["dyneste"]["jsd"]["inferred"].append(dyneste_jsd_inf)
            results["dyneste"]["jsd"]["sampled"].append(dyneste_jsd_sam)
            results["hmm"]["jsd"]["inferred"].append(hmm_jsd_inf)
            results["hmm"]["jsd"]["sampled"].append(hmm_jsd_sam)
            # shape: (n_runs * n_states,)

    # -------------- [3] Visualization -------------- #
    print("Step 3: Visualizing results ...")

    # Plot Dice-Sørensen coefficients
    print("(Step 3-1) Plotting Dice coefficients ...")

    dices = results["dyneste"]["dice"] + results["hmm"]["dice"]
    df = pd.DataFrame({
        "Dice-Sørensen Coefficient": dices,
        "Model": np.repeat(["DyNeStE", "HMM"], n_runs),
    })

    up.plot_violin(
        df,
        x_var="Dice-Sørensen Coefficient",
        y_var="Model",
        hue_var="Model",
        ylbl=False,
        palette={"DyNeStE": "#6EB5C0", "HMM": "#B6443F"},
        filename=os.path.join(FIG_DIR, "dice_coefficients.png"),
    )

    # Plot RV coefficients
    print("(Step 3-2) Plotting RV coefficients ...")

    rv_coeffs = np.concatenate([
        flatten_nested_data(results["dyneste"]["rv_coeffs"]),
        flatten_nested_data(results["hmm"]["rv_coeffs"]),
    ])
    df = pd.DataFrame({
        "RV Coefficient": rv_coeffs,
        "Model": np.repeat(["DyNeStE", "HMM"], n_runs * n_states),
    })

    up.plot_violin(
        df,
        x_var="RV Coefficient",
        y_var="Model",
        hue_var="Model",
        ylbl=False,
        palette={"DyNeStE": "#6EB5C0", "HMM": "#B6443F"},
        filename=os.path.join(FIG_DIR, "rv_coefficients.png"),
    )

    # Plot JS distances
    print("(Step 3-3) Plotting JS distances ...")

    inferred_jsd = results["dyneste"]["jsd"]["inferred"] + results["hmm"]["jsd"]["inferred"]
    sampled_jsd = results["dyneste"]["jsd"]["sampled"] + results["hmm"]["jsd"]["sampled"]
    jsd = inferred_jsd + sampled_jsd
    df = pd.DataFrame({
        "Jensen-Shannon Distance": jsd,
        "Model": np.tile(np.repeat(["DyNeStE", "HMM"], n_runs * n_states), 2),
        "Type": np.repeat(["Inferred", "Sampled"], n_runs * n_states * 2),
    })

    up.plot_violin(
        df,
        x_var="Jensen-Shannon Distance",
        y_var="Type",
        hue_var="Model",
        figsize=(5, 2),
        palette={"DyNeStE": "#6EB5C0", "HMM": "#B6443F"},
        filename=os.path.join(FIG_DIR, "js_distances.png"),
    )

    # Test statistical differences between models
    print("(Step 3-4) Testing statistical differences ...")

    inf_stat, inf_pval, inf_sig = stat_ind_two_samples(
        results["dyneste"]["jsd"]["inferred"],
        results["hmm"]["jsd"]["inferred"],
        alpha=0.05,
    )

    sam_stat, sam_pval, sam_sig = stat_ind_two_samples(
        results["dyneste"]["jsd"]["sampled"],
        results["hmm"]["jsd"]["sampled"],
        alpha=0.05,
    )

    print("Analysis complete.")
