"""Post-hoc visualization of the replay dataset analysis results."""

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
    BASE_DIR = "/well/woolrich/users/olt015/Cho2025_DyNeStE/replay"
    DATA_DIR = os.path.join(BASE_DIR, "results/{0}/{1}/run{2}")
    FIG_DIR = os.path.join(BASE_DIR, "figures")
    os.makedirs(FIG_DIR, exist_ok=True)

    # Set user-defined parameters
    data_type = "study1"
    dyneste_run = 0
    hmm_run = 0
    print(f"[INFO] Data type: {data_type} | DyNeStE Run ID: {dyneste_run} | HMM Run ID: {hmm_run}")

    # Set model directories (with selected model runs)
    dyneste_dir = DATA_DIR.format(data_type, "dyneste", dyneste_run)
    hmm_dir = DATA_DIR.format(data_type, "hmm", hmm_run)

    # Load replay-evoked state activations
    print("(Step 1-1) Loading replay-evoked state activations ...")

    dyneste_resa = np.load(os.path.join(dyneste_dir, "resa.npy"))
    hmm_resa = np.load(os.path.join(hmm_dir, "resa.npy"))
    # shape: (n_sessions, n_epoched_samples, n_states)

    # Load network-dependent replay intervals
    print("(Step 1-2) Loading network-dependent replay intervals ...")
    dyneste_ndri = load(os.path.join(dyneste_dir, "ndri.pkl"))
    hmm_ndri = load(os.path.join(hmm_dir, "ndri.pkl"))
    # shape: (n_states + 1, n_subjects)

    # Load network-dependent replay rates
    print("(Step 1-3) Loading network-dependent replay rates ...")
    dyneste_ndrr = load(os.path.join(dyneste_dir, "ndrr.pkl"))
    hmm_ndrr = load(os.path.join(hmm_dir, "ndrr.pkl"))
    # shape: (n_states + 1, n_subjects)

    # Load Fano factors
    print("(Step 1-4) Loading Fano factors ...")
    dyneste_fano = np.load(os.path.join(dyneste_dir, "fano_factor.npy"))
    hmm_fano = np.load(os.path.join(hmm_dir, "fano_factor.npy"))
    # shape: (n_sessions, n_windows, n_states)

    # Set hyperparameters
    Fs = 250  # sampling frequency
    n_sessions = dyneste_resa.shape[0]  # number of sessions
    n_subjects = int(n_sessions / 2)  # number of subjects
    n_states = dyneste_resa.shape[-1]  # number of states
    n_jobs = 8  # number of CPUs to use for parallel processing

    # -------------- [2] Replay-Evoked State Activations -------------- #
    print("Step 2: Visualizing replay-evoked state activations ...")

    # Plot replay-evoked state activations
    up.plot_replay_evoked_state_activations(
        dyneste_resa,
        filename=os.path.join(
            FIG_DIR, f"resa_{data_type}_dyneste_run{dyneste_run}.png"
        ),
    )

    up.plot_replay_evoked_state_activations(
        hmm_resa,
        filename=os.path.join(
            FIG_DIR, f"resa_{data_type}_hmm_run{hmm_run}.png"
        ),
    )

    # -------------- [3] Network-Dependent Replay Intervals -------------- #
    print("Step 3: Visualizing network-dependent replay intervals ...")

    # Plot network-dependent replay intervals
    up.plot_network_dependent_replay_intervals(
        dyneste_ndri,
        sampling_frequency=Fs,
        filename=os.path.join(
            FIG_DIR, f"ndri_{data_type}_dyneste_run{dyneste_run}.png"
        ),
    )

    up.plot_network_dependent_replay_intervals(
        hmm_ndri,
        sampling_frequency=Fs,
        filename=os.path.join(
            FIG_DIR, f"ndri_{data_type}_hmm_run{hmm_run}.png"
        ),
    )

    # -------------- [4] Network-Dependent Replay Rates -------------- #
    print("Step 4: Visualizing network-dependent replay rates ...")

    # Plot network-dependent replay rates
    up.plot_network_dependent_replay_rates(
        dyneste_ndrr,
        filename = os.path.join(
            FIG_DIR, f"ndrr_{data_type}_dyneste_run{dyneste_run}.png"
        ),
    )

    up.plot_network_dependent_replay_rates(
        hmm_ndrr,
        filename = os.path.join(
            FIG_DIR, f"ndrr_{data_type}_hmm_run{hmm_run}.png"
        ),
    )

    # -------------- [5] Fano Factors -------------- #
    print("Step 5: Visualizing Fano factors ...")

    # Define window lengths to use
    window_lengths = np.unique(
        np.round(np.exp(np.linspace(np.log(2), np.log(2500), 50)))
    ) # unit: samples

    # Run cluster-level statistical permutation test
    sig_clu_idx = np.zeros((len(window_lengths),))

    thr = 0.05 / n_states  # Bonferroni-corrected threshold
    for n in range(n_states):
        # Test for Fano factors of DyNeStE and HMM
        f_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_test(
            [dyneste_fano[:, :, n], hmm_fano[:, :, n]],
            n_permutations=10000,
            threshold=3,
            stat_fun=mne.stats.ttest_ind_no_p,
            tail=0,
            n_jobs=n_jobs,
        )
        for i, p in enumerate(cluster_pv):
            if p < thr:
                sig_clu_idx[clusters[i][0]] += 1

    # Find overlapping window lengths across states
    # (i.e., the intersection of significant clusters)
    sig_clu_idx = np.where(np.array(sig_clu_idx) == n_states)[0]
    print(f"Significant cluster indices for Fano factors: {sig_clu_idx}")

    # Plot the Fano factors
    up.plot_fano_factors(
        dyneste_fano,
        window_lengths=window_lengths,
        sampling_frequency=Fs,
        sig_indices=sig_clu_idx,
        ylims=[0.5, 5.5],
        filename=os.path.join(
            FIG_DIR, f"fano_factor_{data_type}_dyneste_run{dyneste_run}.png"
        ),
    )

    up.plot_fano_factors(
        hmm_fano,
        window_lengths=window_lengths,
        sampling_frequency=Fs,
        sig_indices=sig_clu_idx,
        ylims=[0.5, 5.5],
        filename=os.path.join(
            FIG_DIR, f"fano_factor_{data_type}_hmm_run{hmm_run}.png"
        ),
    )

    up.plot_fano_factor_effect_size(
        [dyneste_fano, hmm_fano],
        window_lengths=window_lengths,
        sampling_frequency=Fs,
        sig_indices=sig_clu_idx,
        ylims=None,
        filename=os.path.join(
            FIG_DIR, f"fano_factor_effect_{data_type}.png"
        ),
    )

    print("Visualization complete.")
