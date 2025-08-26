"""Post-hoc visualization of long-term dependency analysis results."""

# Import packages
import os
import mne
import numpy as np
from utils import plotting as up
from utils.data import load


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

    # Load Fano factors
    print("(Step 1-1) Loading Fano factors ...")
    dyneste_fano = load(os.path.join(dyneste_dir, "inference/fano_factor.pkl"))
    hmm_fano = load(os.path.join(hmm_dir, "inference/fano_factor.pkl"))

    dyneste_inf_fano, dyneste_sam_fano = dyneste_fano.values()
    hmm_inf_fano, hmm_sam_fano = hmm_fano.values()
    # shape: (n_subjects, n_windows, n_states)

    # Load mutual information
    print("(Step 1-2) Loading mutual information ...")
    dyneste_mi = load(os.path.join(dyneste_dir, "inference/mutual_information.pkl"))
    hmm_mi = load(os.path.join(hmm_dir, "inference/mutual_information.pkl"))
    
    dyneste_inf_mi, dyneste_sam_mi = dyneste_mi.values()
    hmm_inf_mi, hmm_sam_mi = hmm_mi.values()
    # shape: (n_subjects, n_lags, n_states)

    # Set hyperparameters
    Fs = 250  # sampling frequency
    n_subjects = dyneste_inf_fano.shape[0]  # number of subjects
    n_states = dyneste_inf_fano.shape[-1]  # number of states
    n_jobs = 8  # number of CPUs to use for parallel processing

    # -------------- [2] Fano Factor -------------- #
    print("Step 2: Visualizing Fano factors ...")

    # Define window lengths to use
    window_lengths = np.unique(
        np.round(np.exp(np.linspace(np.log(2), np.log(1000), 30)))
    ) # unit: samples

    # Run cluster-level statistical permutation test
    inf_sig_clu_idx = np.zeros((len(window_lengths),))
    sam_sig_clu_idx = np.zeros((len(window_lengths),))

    thr = 0.05 / n_states  # Bonferroni-corrected threshold
    for n in range(n_states):
        # Test for inferred Fano factors of DyNeStE and HMM
        dyneste_inf_fano_n = dyneste_inf_fano[:, :, n]
        hmm_inf_fano_n = hmm_inf_fano[:, :, n]
        f_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_test(
            [dyneste_inf_fano_n, hmm_inf_fano_n],
            n_permutations=10000,
            threshold=3,
            stat_fun=mne.stats.ttest_ind_no_p,
            tail=0,
            n_jobs=n_jobs,
        )
        for i, p in enumerate(cluster_pv):
            if p < thr:
                inf_sig_clu_idx[clusters[i][0]] += 1

        # Test for sampled Fano factors of DyNeStE and HMM
        dyneste_sam_fano_n = dyneste_sam_fano[:, :, n]
        hmm_sam_fano_n = hmm_sam_fano[:, :, n]
        f_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_test(
            [dyneste_sam_fano_n, hmm_sam_fano_n],
            n_permutations=10000,
            threshold=3,
            stat_fun=mne.stats.ttest_ind_no_p,
            tail=0,
            n_jobs=n_jobs,
        )
        for i, p in enumerate(cluster_pv):
            if p < thr:
                sam_sig_clu_idx[clusters[i][0]] += 1

    # Find overlapping window lengths across states
    # (i.e., the intersection of significant clusters)
    inf_sig_clu_idx = np.where(np.array(inf_sig_clu_idx) == n_states)[0]
    sam_sig_clu_idx = np.where(np.array(sam_sig_clu_idx) == n_states)[0]
    print(f"Significant cluster indices for inferred Fano factors: {inf_sig_clu_idx}")
    print(f"Significant cluster indices for sampled Fano factors: {sam_sig_clu_idx}")

    # Plot the Fano factors
    up.plot_fano_factors(
        [dyneste_inf_fano, dyneste_sam_fano],
        window_lengths,
        sampling_frequency=Fs,
        sig_indices=[[inf_sig_clu_idx], [sam_sig_clu_idx]],
        ylims=[[0.5, 6.0], [0.5, 6.0]],
        filename=os.path.join(
            FIG_DIR, f"fano_factor_dyneste_run{best_dyneste_run}.png"
        )
    )

    up.plot_fano_factors(
        [hmm_inf_fano, hmm_sam_fano],
        window_lengths,
        sampling_frequency=Fs,
        sig_indices=[[inf_sig_clu_idx], [sam_sig_clu_idx]],
        ylims=[[0.5, 6.0], [0.5, 6.0]],
        filename=os.path.join(
            FIG_DIR, f"fano_factor_hmm_run{best_hmm_run}.png"
        )
    )

    # -------------- [3] Mutual Information -------------- #
    print("Step 3: Visualizing mutual information ...")

    # Define lags to use
    lags = np.linspace(-Fs, Fs, num=11, endpoint=True, dtype=int)
    lags = lags[lags != 0]  # remove zero lag

    # Run cluster-level statistical permutation test
    inf_sig_clu_idx = np.zeros((len(lags),))
    sam_sig_clu_idx = np.zeros((len(lags),))

    thr = 0.05 / n_states  # Bonferroni-corrected threshold
    for n in range(n_states):
        # Test for inferred mutual information of DyNeStE and HMM
        dyneste_inf_mi_n = dyneste_inf_mi[:, :, n]
        hmm_inf_mi_n = hmm_inf_mi[:, :, n]
        f_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_test(
            [dyneste_inf_mi_n, hmm_inf_mi_n],
            n_permutations=10000,
            threshold=3,
            tail=0,
            n_jobs=n_jobs,
            stat_fun=mne.stats.ttest_ind_no_p,
        )
        for i, p in enumerate(cluster_pv):
            if p < thr:
                inf_sig_clu_idx[clusters[i][0]] += 1

        # Test for generated mutual information of DyNeStE and HMM
        dyneste_sam_mi_n = dyneste_sam_mi[:, :, n]
        hmm_sam_mi_n = hmm_sam_mi[:, :, n]
        f_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_test(
            [dyneste_sam_mi_n, hmm_sam_mi_n],
            n_permutations=10000,
            threshold=3,
            stat_fun=mne.stats.ttest_ind_no_p,
            tail=0,
            n_jobs=n_jobs,
        )
        for i, p in enumerate(cluster_pv):
            if p < thr:
                sam_sig_clu_idx[clusters[i][0]] += 1

    # Find overlapping lags across states
    # (i.e., the intersection of significant clusters)
    inf_sig_clu_idx = np.where(np.array(inf_sig_clu_idx) == n_states)[0]
    sam_sig_clu_idx = np.where(np.array(sam_sig_clu_idx) == n_states)[0]
    print(f"Significant cluster indices for inferred mutual information: {inf_sig_clu_idx}")
    print(f"Significant cluster indices for generated mutual information: {sam_sig_clu_idx}")

    # Plot the mutual information
    up.plot_mutual_information(
        [dyneste_inf_mi, dyneste_sam_mi],
        lags,
        sampling_frequency=Fs,
        sig_indices=[[inf_sig_clu_idx], [sam_sig_clu_idx]],
        xticks=lags[[0, 2, 4, 5, 7, 9]],
        ylims=[[-5.6e-4, 9.3e-3], [-5.6e-4, 9.3e-3]],
        filename=os.path.join(
            FIG_DIR, f"mi_dyneste_run{best_dyneste_run}.png"
        )
    )

    up.plot_mutual_information(
        [hmm_inf_mi, hmm_sam_mi],
        lags,
        sampling_frequency=Fs,
        sig_indices=[[inf_sig_clu_idx], [sam_sig_clu_idx]],
        xticks=lags[[0, 2, 4, 5, 7, 9]],
        ylims=[[-5.6e-4, 9.3e-3], [-5.6e-4, 9.3e-3]],
        filename=os.path.join(
            FIG_DIR, f"mi_hmm_run{best_hmm_run}.png"
        )
    )

    print("Visualization complete.")
