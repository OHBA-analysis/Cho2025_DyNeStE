"""Post-hoc analysis of split-half reproducibility."""

# Import packages
import os
import numpy as np
from scipy.spatial.distance import cosine
from osl_dynamics.analysis import power
from utils.analysis import compute_tv_covariances
from utils.data import load, match_order
from utils.statistics import split_half_permutation_test


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

    # Get state-specific power spectra
    psd_data_1 = load(os.path.join(split1_dir, "inference/psds.pkl"))
    freqs = psd_data_1["freqs"]
    psds_1 = psd_data_1["psds"]
    weights_1 = psd_data_1["weights"]
    
    psd_data_2 = load(os.path.join(split2_dir, "inference/psds.pkl"))
    psds_2 = psd_data_2["psds"]
    weights_2 = psd_data_2["weights"]

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

    # Load training data time series
    ts_1 = load(os.path.join(BASE_DIR, "data/split1/trimmed_ts.pkl"))
    ts_2 = load(os.path.join(BASE_DIR, "data/split2/trimmed_ts.pkl"))
    # shape: (n_subjects, n_samples, n_channels)

    # Validate time course lengths and shapes
    if len(alphas_1) != len(ts_1) or len(alphas_2) != len(ts_2):
        raise ValueError(
            "The number of subjects between alpha time courses " +
            "and data time series should be matched."
        )

    assert all(
        alphas_1[i].shape[0] == ts_1[i].shape[0] for i in range(len(alphas_1))
    ), "Inconsistent data shapes between alpha time courses and time series."
    assert all(
        alphas_2[i].shape[0] == ts_2[i].shape[0] for i in range(len(alphas_2))
    ), "Inconsistent data shapes between alpha time courses and time series."

    # -------------- [3] Split-half Comparisons -------------- #
    print("Step 3: Comparing split-halves ...")

    # Compute state-specific power maps
    power_maps_1 = power.variance_from_spectra(
        freqs,
        psds_1,
        frequency_range=[1.5, 20],
    )
    power_maps_2 = power.variance_from_spectra(
        freqs,
        psds_2,
        frequency_range=[1.5, 20],
    )

    # Average power maps across subjects
    power_maps_1 = np.average(power_maps_1, weights=weights_1, axis=0)
    power_maps_2 = np.average(power_maps_2, weights=weights_2, axis=0)
    # shape: (n_states, n_channels)

    # Subtract average across states
    power_maps_1 -= np.mean(power_maps_1, axis=0, keepdims=True)
    power_maps_2 -= np.mean(power_maps_2, axis=0, keepdims=True)

    # Compute cosine similarities between power maps
    cosine_similarities = []
    for p1, p2 in zip(power_maps_1, power_maps_2):
        cosine_similarities.append(1 - cosine(p1, p2))
    cosine_similaritys = np.array(cosine_similarities)
    print(f"Cosine similarities: {cosine_similarities}")

    # -------------- [4] Non-parametric Permutation Tests -------------- #
    print("Step 4: Running permutation tests ...")

    # Compute time-varying covarainces
    kwargs = {
        "window_length": 50,
        "step_size": 5,
        "shuffle_window_length": 250,
        "n_jobs": 16,
    }

    tv_covs_1 = compute_tv_covariances(
        ts_1, sampling_frequency=250, **kwargs
    )
    tv_covs_2 = compute_tv_covariances(
        ts_2, sampling_frequency=250, **kwargs
    )

    # Perform split-half permutation test
    cs_pvals, cs_sig = split_half_permutation_test(
        [tv_covs_1, tv_covs_2],
        [alphas_1, alphas_2],
        cosine_similarities=cosine_similarities,
        n_perms=1000,
        **kwargs,
    )

    print(f"Cosine similarity p-values: {cs_pvals}")
    print(f"Cosine similarity significance: {cs_sig}")

    print("Analysis complete.")
