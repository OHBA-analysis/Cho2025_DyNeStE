"""Post-hoc analysis of the replay dataset."""

# Import packages
import os
import numpy as np
from sys import argv
from tqdm import trange
from osl_dynamics.inference import modes
from utils.array_ops import find_on_off_indices
from utils.data import load, save


if __name__ == "__main__":
    # -------------- [1] Settings -------------- #
    print("Step 1: Setting up ...")

    # Set user defined arguments
    if len(argv) != 4:
        raise ValueError(
            "Need to pass three arguments: model type, run ID, and data type" +
            f" (e.g., python {argv[0]} dyneste 0 study1)"
        )
    model_type = argv[1]  # model type
    run_id = int(argv[2])  # run ID
    data_type = argv[3]  # data type
    print(f"[INFO] Model Type: {model_type} | Run ID: run{run_id} | Data Type: {data_type}")

    # Validate user inputs
    if model_type not in ["dyneste", "hmm"]:
        raise ValueError("Model type must be one of ['dyneste', 'hmm'].")
    if data_type not in ["study1", "study2"]:
        raise ValueError("Data type must be one of ['study1', 'study2'].")
    
    # Set output directories
    BASE_DIR = "/well/woolrich/users/olt015/Cho2025_DyNeStE/replay"
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODEL_DIR = os.path.join(BASE_DIR, f"results/{data_type}/{model_type}/run{run_id}")
    FIG_DIR = os.path.join(BASE_DIR, "figures")
    os.makedirs(FIG_DIR, exist_ok=True)

    # Set hyperparameters
    Fs = 250  # sampling frequency

    # Load replay indices
    replay_indices = load(os.path.join(DATA_DIR, f"{data_type}/replay_indices.pkl"))
    # shape: (n_sessions, n_replays)
    # NOTE: There are two resting-state scan sessions per subject.

    # Load inferred parameters
    inf_params = load(os.path.join(MODEL_DIR, "inf_params.pkl"))
    alphas = inf_params["alpha"]

    if len(alphas) != len(replay_indices):
        raise ValueError("Number of sessions for replay indices and alphas do not match.")
    else:
        n_sessions = len(alphas)  # number of sessions
    n_states = alphas[0].shape[1]  # number of states

    # -------------- [2] Preprocessing -------------- #
    print("Step 2: Preprocessing data and inferred parameters ...")

    # Preprocess replay indices
    replay_indices = [indices.astype(int) for indices in replay_indices]
    for n in range(n_sessions):
        n_samples = alphas[n].shape[0]
        up_bound_cond = (replay_indices[n] + (Fs // 2) + 1) < n_samples
        low_bound_cond = (replay_indices[n] - (Fs // 2)) > 0
        replay_indices[n] = replay_indices[n][up_bound_cond & low_bound_cond]
    # NOTE: Inferred alphas can be shorter than the original time series as we time-embed and segment the data 
    #       before fitting the model. Therefore, we need to filter out replay indices that are out of bounds.
    #       These bounds should account for how we epoch the data (see below).

    # Reorder states if necessary
    if model_type == "dyneste":
        order = None
    if model_type == "hmm":
        order = [1, 0, 2, 3, 4, 10, 8, 11, 5, 6, 9, 7]  # adjust accordingly
    # NOTE: Since we are performing inference with the pre-trained model, we can use the
    #       order from the pre-trained model.

    if order is not None:
        print(f"Reordering states ...")
        print(f"\tOrder: {order}")
        alphas = [a[:, order] for a in alphas]
        # shape: (n_sessions, n_samples, n_states)

    # Get state time courses
    stc = modes.argmax_time_courses(alphas)
    # shape: (n_sessions, n_samples, n_states)

    # -------------- [3] Replay-Evoked State Activations -------------- #
    print("Step 3: Computing replay-evoked state activations ...")

    # Get the replay-evoked state activations (RESAs)
    resa_path = os.path.join(MODEL_DIR, "resa.npy")

    if os.path.exists(resa_path):
        # Load the replay-evoked state activations
        print("Replay-evoked state activations already exist. Loading ...")
        resa = np.load(resa_path)
    else:
        # Compute the replay-evoked state activations
        print("Computing replay-evoked state activations ...")
        resa = []  # replay-evoked state activations
        for n in trange(n_sessions):
            bc_alpha = alphas[n] - alphas[n].mean(axis=0)  # baseline correction across time
            sliced_alphas = []
            for r_idx in replay_indices[n]:
                epoch_indices = slice(r_idx - (Fs // 2), r_idx + (Fs // 2) + 1)  # 0.5s before and after replay
                sliced_alphas.append(bc_alpha[epoch_indices, :])
            resa.append(np.mean(sliced_alphas, axis=0))  # for each session, get the mean activation across replays
        resa = np.array(resa)
        # shape: (n_sessions, n_epoched_samples, n_states)

        # Save the replay-evoked state activations
        np.save(resa_path, resa)

    # -------------- [4] Network-Dependent Replay Intervals -------------- #
    print("Step 4: Computing network-dependent replay intervals ...")

    # Get the network-dependent replay intervals (NDRIs)
    ndri_path = os.path.join(MODEL_DIR, "ndri.pkl")

    if os.path.exists(ndri_path):
        # Load the network-dependent replay intervals
        print("Network-dependent replay intervals already exist. Loading ...")
        network_dependent_replay_intervals = load(ndri_path)
    else:
        # Compute the network-dependent replay intervals
        print("Computing network-dependent replay intervals ...")

        # Get replay intervals given active RSN states per session
        ndris = []  # network-dependent replay intervals
        for n in range(n_sessions):
            replay_intervals = np.diff(replay_indices[n])
            active_states = np.argmax(alphas[n][replay_indices[n][:-1]], axis=1)
            ndri = dict()
            for s, r in zip(active_states, replay_intervals):
                ndri[s + 1] = ndri.get(s + 1, []) + [r]
            ndris.append(ndri)
        # shape: (n_sessions, n_states, n_replays)
        # NOTE: The number of replays might differ across states and sessions.

        # Get replay intervals given active RSN states per subject
        ndris_subjects = []
        for i in range(0, len(ndris), 2):
            d1 = ndris[i]
            d2 = ndris[i + 1] if i + 1 < len(ndris) else {}

            merged = {}
            keys = set(d1.keys()) | set(d2.keys())  # union of keys
            for key in keys:
                merged[key] = d1.get(key, []) + d2.get(key, [])
            ndris_subjects.append(merged)
        # shape: (n_subjects, n_states, n_replays)

        # Compute mean replay intervals for each active RSN state
        for i, ndri in enumerate(ndris_subjects):
            for key, value in ndri.items():
                if len(ndri[key]) >= 10:
                    ndri[key] = np.mean(value)  # mean across replays
                else: ndri[key] = []
                # NOTE: We omit mean replay intervals with less than 10 replay instances 
                #       to prevent high variance estimates.
            ndri["Mean"] = np.sum([val for val in ndri.values() if val]) / len(ndri)  # mean across replays over all states
            ndris_subjects[i] = ndri
        # shape: (n_subjects, n_states + 1); one added state for the mean across all states

        # Combine the network-dependent replay intervals
        network_dependent_replay_intervals = {}
        for ndri in ndris_subjects:
            for key, value in ndri.items():
                network_dependent_replay_intervals.setdefault(key, []).append(value)
        # shape: (n_states + 1, n_subjects)

        # Save the network-dependent replay intervals
        save(network_dependent_replay_intervals, ndri_path)

    # -------------- [5] Network-Dependent Replay Rates -------------- #
    print("Step 5: Computing network-dependent replay rates ...")

    # Get the network-dependent replay rates (NDRRs)
    ndrr_path = os.path.join(MODEL_DIR, "ndrr.pkl")

    if os.path.exists(ndrr_path):
        # Load the network-dependent replay rates
        print("Network-dependent replay rates already exist. Loading ...")
        ndrr = load(ndrr_path)
    else:
        # Compute the network-dependent replay rates
        print("Computing network-dependent replay rates ...")

        # Get replay rates given active RSN states per subject
        ndrrs_subjects = []
        for n in range(0, n_sessions, 2):
            ndrr_binary_sess1 = np.zeros(stc[n].shape[0])
            ndrr_binary_sess2 = np.zeros(stc[n + 1].shape[0])
            ndrr_binary_sess1[replay_indices[n]] = 1
            ndrr_binary_sess2[replay_indices[n + 1]] = 1

            ndrr = np.zeros((n_states,))
            for s in range(n_states):
                onoff_idx_sess1 = find_on_off_indices(stc[n][:, s])
                onoff_idx_sess2 = find_on_off_indices(stc[n + 1][:, s])
                
                mask_sess1 = np.zeros(stc[n].shape[0])
                mask_sess2 = np.zeros(stc[n + 1].shape[0])
                for onset, offset in onoff_idx_sess1:
                    mask_sess1[onset:offset + 1] = 1
                for onset, offset in onoff_idx_sess2:
                    mask_sess2[onset:offset + 1] = 1
                
                replay_sum = np.sum(ndrr_binary_sess1 * mask_sess1) + np.sum(ndrr_binary_sess2 * mask_sess2)
                active_state_sum = np.sum(mask_sess1) + np.sum(mask_sess2)
                rate = (replay_sum / active_state_sum) if active_state_sum > 0 else 0  # replay per samples
                ndrr[s] = rate * Fs  # convert to replay per second
            ndrrs_subjects.append(ndrr)
        ndrrs_subjects = np.array(ndrrs_subjects)
        # shape: (n_subjects, n_states)

        # Compute mean replay rates for each active RSN state
        ndrrs_mean = np.mean(ndrrs_subjects, axis=1)  # average across states
        # shape: (n_subjects,)

        # Combine the network-dependent replay rates
        network_dependent_replay_rates = {}
        for s in range(n_states):
            network_dependent_replay_rates[s + 1] = ndrrs_subjects[:, s].tolist()
        network_dependent_replay_rates["Mean"] = ndrrs_mean.tolist()  # mean across states
        # shape: (n_states + 1, n_subjects); one added state for the mean across all states
        
        # Save the network-dependent replay rates
        save(network_dependent_replay_rates, ndrr_path)

    # -------------- [6] Fano Factor -------------- #
    print("Step 6: Computing Fano factors ...")

    # Define window lengths to use
    window_lengths = np.unique(
        np.round(np.exp(np.linspace(np.log(2), np.log(2500), 50)))
    ) # unit: samples

    # Get the Fano factors
    fano_path = os.path.join(MODEL_DIR, "fano_factor.npy")

    if os.path.exists(fano_path):
        # Load the Fano factors
        print("Fano factors already exist. Loading ...")
        fano_factors = np.load(fano_path)
    else:
        # Compute the Fano factors
        print("Computing the Fano factor...")
        fano_factors = np.array(modes.fano_factor(stc, window_lengths))
        # shape: (n_sessions, n_windows, n_states)
        np.save(fano_path, fano_factors)

    print("Analysis complete.")
