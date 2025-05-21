"""Post-hoc analysis of split-half reproducibility."""

# Import packages
import os
import numpy as np
from osl_dynamics.inference import modes, metrics
from utils.data import load, match_order


if __name__ == "__main__":
    # -------------- [1] Settings -------------- #
    print("Step 1: Setting up ...")

    # Set user-defined hyperparameters
    model_type = "dyneste"  # model type
    run_id_1 = 1  # run ID for first split-half
    run_id_2 = 4  # run ID for second split-half
    best_dyneste_run = 2  # best run ID for DyNeStE model (trained on full data)

    print(f"[INFO] Model Type: {model_type} | Run ID (1st): run{run_id_1}" +
          f" | Run ID (2nd): {run_id_2}")

    # Validate user inputs
    if model_type not in ["dyneste", "hmm"]:
        raise ValueError("Model type must be one of ['dyneste', 'hmm'].")

    # Set output directories
    BASE_DIR = "/well/woolrich/users/olt015/Cho2025_DyNeStE/nott_meguk"
    DATA_DIR = os.path.join(BASE_DIR, "results/split{0}/{1}/run{2}")
    split1_dir = DATA_DIR.format(1, model_type, run_id_1)
    split2_dir = DATA_DIR.format(2, model_type, run_id_2)

    # -------------- [2] Data Loading -------------- #
    print("Step 2: Loading inferred parameters ...")

    # Get inferred parameters
    inf_params_1 = load(os.path.join(split1_dir, "inference/inf_params.pkl"))
    alphas_1 = inf_params_1["alpha"]
    covs_1 = inf_params_1["covariance"]
    
    inf_params_2 = load(os.path.join(split2_dir, "inference/inf_params.pkl"))
    alphas_2 = inf_params_2["alpha"]
    covs_2 = inf_params_2["covariance"]

    # Get state orders for the specified model run
    ref_run = ("full", "dyneste", best_dyneste_run)
    
    if model_type == "hmm":
        order_1 = match_order(
            ref_info=ref_run,
            cmp_info=("split1", model_type, run_id_1),
        )
        order_2 = match_order(
            ref_info=ref_run,
            cmp_info=("split2", model_type, run_id_2),
        )
    else:
        # Use manual ordering
        order_1 = [7, 4, 11, 5, 2, 9, 8, 1, 3, 10, 0, 6] # run 1
        order_2 = [3, 10, 11, 6, 7, 0, 2, 4, 9, 1, 5, 8] # run 4

    if order_1 is not None:
        print(f"Reordering states for the first split half ...")
        print(f"\tOrder: {order_1}")
        alphas_1 = [a[:, order_1] for a in alphas_1]
        covs_1 = covs_1[order_1]
    if order_2 is not None:
        print(f"Reordering states for the second split half ...")
        print(f"\tOrder: {order_2}")
        alphas_2 = [a[:, order_2] for a in alphas_2]
        covs_2 = covs_2[order_2]

    # Get state time courses
    stc_1 = modes.argmax_time_courses(alphas_1)
    stc_2 = modes.argmax_time_courses(alphas_2)

    # -------------- [4] Split-half Comparisons -------------- #
    print("Step 3: Comparing split-halves ...")

    # Compute Riemannian distances between covariance matrices
    riemannian_distances = []
    for c1, c2 in zip(covs_1, covs_2):
        riemannian_distances.append(
            metrics.pairwise_riemannian_distances(np.array([c1, c2]))[0, 1]
        )
    print(f"Riemannian distances: {riemannian_distances}")

    # Compute RV coefficients between covariance matrices
    rv_coefficients = []
    for c1, c2 in zip(covs_1, covs_2):
        rv_coefficients.append(
            metrics.pairwise_rv_coefficient(np.array([c1, c2]))[0, 1]
        )
    print(f"RV coefficients: {rv_coefficients}")

    print("Analysis complete.")
