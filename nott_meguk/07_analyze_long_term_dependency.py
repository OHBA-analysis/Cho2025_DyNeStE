"""Post-hoc analysis of the long-term dependencies."""

# Import packages
import os
import numpy as np
import seaborn as sns

from sys import argv
from tqdm import trange

from osl_dynamics.analysis import modes
from osl_dynamics.inference.modes import argmax_time_courses

from utils import analysis as ua
from utils import plotting as up
from utils.data import load, save, match_order
from utils.model import load_model


if __name__ == "__main__":
    # -------------- [1] Settings -------------- #
    print("Step 1: Setting up ...")

    # Set user defined arguments
    if len(argv) != 4:
        raise ValueError(
            "Need to pass three arguments: model type, run ID, and data type" +
            f" (e.g., python {argv[0]} dyneste 0 full)"
        )
    model_type = argv[1]  # model type
    run_id = int(argv[2])  # run ID
    data_type = argv[3]  # data type
    print(f"[INFO] Model Type: {model_type} | Run ID: run{run_id} | Data Type: {data_type}")

    # Validate user inputs
    if model_type not in ["dyneste", "hmm"]:
        raise ValueError("Model type must be one of ['dyneste', 'hmm'].")
    if data_type not in ["full", "split1", "split2"]:
        raise ValueError("Data type must be one of ['full', 'split1', 'split2'].")

    # Set output directories
    BASE_DIR = "/well/woolrich/users/olt015/Cho2025_DyNeStE/nott_meguk"
    DATA_DIR = os.path.join(BASE_DIR, f"results/{data_type}/{model_type}/run{run_id}")
    FIG_DIR = os.path.join(DATA_DIR, "figures")
    os.makedirs(FIG_DIR, exist_ok=True)

    # Load inferred parameters
    print("(Step 1-1) Loading inferred parameters ...")
    
    inf_params = load(os.path.join(DATA_DIR, "inference/inf_params.pkl"))
    alphas = inf_params["alpha"]
    covs = inf_params["covariance"]
    stc = argmax_time_courses(alphas)

    # Set hyperparameters
    Fs = 250  # sampling frequency
    n_subjects = len(alphas)  # number of subjects
    n_states = alphas[0].shape[1]  # number of states
    best_dyneste_run = 2  # best run ID for DyNeStE model

    # Sample state time courses
    print("(Step 1-2) Sampling state time courses ...")
    
    sample_path = os.path.join(DATA_DIR, "inference/samples.pkl")
    
    if os.path.exists(sample_path):
        print("Sampled state time courses already exist. Loading ...")
        output = load(sample_path)
        if model_type == "dyneste":
            sam_alphas = output["sam_alphas"]
        sam_stc = output["sam_stc"]
    else:
        sampling_time = 300  # unit: seconds (5 minutes)
        n_samples = int(Fs * sampling_time)  # number of samples

        model = load_model(os.path.join(DATA_DIR, "model/trained_model"))
        if model_type == "hmm":
            sam_stc = []
            for _ in trange(n_subjects):
                sam_stc.append(model.sample_state_time_course(n_samples))
        if model_type == "dyneste":
            sam_alphas, sam_stc = [], []
            for n in trange(n_subjects):
                sam_alphas.append(model.sample_alpha(n_samples))
                sam_stc.append(argmax_time_courses(sam_alphas[n]))
        # shape (sam_stc/sam_alphas): (n_subjects, n_samples, n_states)

        outputs = {"sam_stc": sam_stc}
        if model_type == "dyneste":
            outputs["sam_alphas"] = sam_alphas
        save(outputs, sample_path)

    # Match state orders between two runs
    print("(Step 1-3) Matching state orders ...")
    
    order = match_order(
        ref_info=("full", "dyneste", 2),
        cmp_info=(data_type, model_type, run_id),
        method="covariances",
    )
    
    if order is not None:
        print("Reordering states ...")
        print(f"\tOrder: {order}")
        reorder = lambda data: [d[:, order] for d in data]
        alphas = reorder(alphas)
        stc = reorder(stc)
        covs = covs[order]
        sam_stc = reorder(sam_stc)
        if model_type == "dyneste":
            sam_alphas = reorder(sam_alphas)

    # -------------- [2] State Time Courses -------------- #
    print("Step 2: Visualizing state time courses ...")

    n_samples = int(Fs * 5)  # 5 seconds to plot
    y_labels = [
        "Prob. (Inferred)",
        "STC (Inferred)",
        "Prob. (Generated)",
        "STC (Generated)",
    ]

    # Plot alphas and state time courses for a single subject
    if model_type == "dyneste":
        alpha_list = [
            alphas[0][:n_samples, :],
            stc[0][:n_samples, :],
            sam_alphas[0][:n_samples, :],
            sam_stc[0][:n_samples, :],
        ]
    if model_type == "hmm":
        alpha_list = [
            alphas[0][:n_samples, :],
            stc[0][:n_samples, :],
            sam_stc[0][:n_samples, :],
        ]
        del y_labels[2]

    up.plot_alpha(
        *alpha_list,
        n_samples=n_samples,
        cmap="Set3",
        sampling_frequency=Fs,
        y_labels=y_labels,
        title=f"{model_type.upper()}",
        filename=os.path.join(FIG_DIR, f"alpha_stc_{model_type}_run{run_id}.png"),
    )

    # -------------- [3] Fano Factor -------------- #
    print("Step 3: Computing Fano factors ...")

    # Define window lengths to use
    window_lengths = np.unique(
        np.round(np.exp(np.linspace(np.log(2), np.log(1000), 30)))
    ) # unit: samples

    # Get the Fano factors
    fano_path = os.path.join(DATA_DIR, "inference/fano_factor.pkl")
    
    if os.path.exists(fano_path):
        # Load the fano factors
        print("Fano factors already exist. Loading ...")
        output = load(fano_path)
        inf_fano, sam_fano = output.values()
    else:
        print("Computing the Fano factor...")

        # Compute the fano factors
        inf_fano = np.array(
            modes.fano_factor(stc, window_lengths)
        )
        sam_fano = np.array(
            modes.fano_factor(sam_stc, window_lengths)
        )
        # shape (*_fano): (n_subjects, n_windows, n_states)

        # Save the fano factors
        save(
            {"inf_fano": inf_fano, "sam_fano": sam_fano},
            fano_path,
        )

    # -------------- [4] Mutual Information -------------- #
    print("Step 4: Computing mutual information ...")

    # Get the mutual information
    lags = np.linspace(-Fs, Fs, num=11, endpoint=True, dtype=int)
    lags = lags[lags != 0]  # remove zero lag

    mi_path = os.path.join(DATA_DIR, "inference/mutual_information.pkl")
    if os.path.exists(mi_path):
        print("Mutual information already exists. Loading ...")
        output = load(mi_path)
        inf_mi, sam_mi = output.values()
    else:
        print("Computing mutual information...")

        # Compute the mutual information
        inf_mi, inf_mi_matrix = ua.compute_mutual_information(stc, lags)
        sam_mi, sam_mi_matrix = ua.compute_mutual_information(sam_stc, lags)
        # shape (*_mi): (n_subjects, n_lags, n_states)
        # shape (*_mi_matrix): (n_subjects, n_lags, n_states, n_states)

        # Save the mutual information
        save({"inf_mi": inf_mi, "sam_mi": sam_mi}, mi_path)

    # -------------- [5] Transition Probability Matrix -------------- #
    print("Step 5: Computing transition probability matrix ...")

    tp_path = os.path.join(DATA_DIR, "inference/transition_probability.pkl")
    if os.path.exists(tp_path):
        print("Transition probability matrix already exists. Loading ...")
        output = load(tp_path)
        inf_tp, sam_tp = output.values()
    else:
        print("Computing transition probability matrix...")

        # Compute the transition probability matrices
        if model_type == "dyneste":
            inf_tp = ua.compute_transition_probability(stc)
        else:
            inf_tp = inf_params["transition_probability"]
        
        if order is not None:
            inf_tp = inf_tp[np.ix_(order, order)]
        
        sam_tp = ua.compute_transition_probability(sam_stc)

        # Save the transition probability matrices
        save({"inf_tp": inf_tp, "sam_tp": sam_tp}, tp_path)

    print("Analysis complete.")
