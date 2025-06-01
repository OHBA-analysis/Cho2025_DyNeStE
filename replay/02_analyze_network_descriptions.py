"""Post-hoc analysis of dynamic MEG resting-state networks."""

# Import packages
import os
import numpy as np
from sys import argv
from osl_dynamics import analysis
from osl_dynamics.inference import modes
from osl_dynamics.utils import plotting
from utils.data import load, save
from utils.plotting import plot_alpha, DynamicVisualizer


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
    DATA_DIR = os.path.join(BASE_DIR, f"results/{data_type}/{model_type}/run{run_id}")
    FIG_DIR = os.path.join(DATA_DIR, "figures")
    os.makedirs(FIG_DIR, exist_ok=True)

    # Set hyperparameters
    best_dyneste_run = 2  # best run ID for DyNeStE model
    Fs = 250  # sampling frequency
    n_jobs = 8  # number of CPUs to use for parallel processing
    vis_method = "manual"  # visualization method ("manual" or "nnmf")
    wb_freqs = [1.5, 20]  # wide-band frequency range (only for "manual" method)

    # Load inferred parameters
    inf_params = load(os.path.join(DATA_DIR, "inf_params.pkl"))
    alphas = inf_params["alpha"]

    # -------------- [2] Parameter Preprocessing -------------- #
    print("Step 2: Preprocessing inferred parameters ...")
    
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
        alphas = [a[:, order] for a in alphas] # shape: (n_subjects, n_samples, n_states)
    
    # Get state time courses
    stc = modes.argmax_time_courses(alphas)

    # -------------- [3] PSD Computations -------------- #
    print("Step 3: Computing power spectral densities ...")
    
    # Set up a file path for saving results
    save_path = os.path.join(DATA_DIR, "psds.pkl")

    # Get state-specific PSDs
    if os.path.exists(save_path):
        # Load power spectra
        print("(Step 3-1) Loading state-specific PSDs ...")
        outputs = load(save_path)
        freqs, psds, cohs, weights = outputs.values()
    else:
        # Load time series training data
        ts = load(os.path.join(DATA_DIR, f"trimmed_ts.pkl"))
        print(f"Number of subjects: {len(ts)}")

        # Validation
        for a, x in zip(alphas, ts):
            assert a.shape[0] == x.shape[0], "Inconsistent data shapes between alpha and time series."
        
        # Compute state-specific power spectra
        print("(Step 3-1) Computing state-specific PSDs ...")
        freqs, psds, cohs, weights = analysis.spectral.multitaper_spectra(
            data=ts,
            alpha=alphas,
            sampling_frequency=Fs,
            time_half_bandwidth=4,
            n_tapers=7,
            frequency_range=[1, 45],
            return_weights=True,
            standardize=True,
            n_jobs=n_jobs,
        )
        # NOTE: State time course is computed within the function using alpha.
        # shape (psd): (n_subjects, n_states, n_channels, n_freqs)
        # shape (coh): (n_subjects, n_states, n_channels, n_channels, n_freqs)

        # Save computed PSDs
        print("(Step 3-2) Saving computed PSDs ...")
        outputs = {
            "freqs": freqs,
            "psds": psds,
            "coherences": cohs,
            "weights": weights,
        }
        save(outputs, save_path)

    # -------------- [4] Dynamic Network Feature Computations -------------- #
    print("Step 4: Computing dynamic network features ...")

    # Perform data-driven spectral decomposition
    if vis_method == "nnmf":
        wb_comp = analysis.spectral.decompose_spectra(cohs, n_components=2)
        plotting.plot_line(
            [freqs] * 2,
            wb_comp,
            x_label="Frequency (Hz)",
            y_label="Spectral Component",
            filename=os.path.join(FIG_DIR, "analysis/nnmf.png"),
        )
        # NOTE: Here, we use non-negative matrix factorization (NNMF) to decompose the data 
        #       into wide-band and noise components.

    # Compute state-specific power maps
    if vis_method == "nnmf":
        power_maps = analysis.power.variance_from_spectra(
            freqs,
            (psds[:, 0, :, :, :] if model_type == "dynemo" else psds),
            components=wb_comp,
        )
        # dim: (n_subjects, n_components, n_states, n_channels)
    if vis_method == "manual":
        power_maps = analysis.power.variance_from_spectra(
            freqs,
            (psds[:, 0, :, :, :] if model_type == "dynemo" else psds),
            frequency_range=wb_freqs,
        )
        # dim: (n_subjects, n_states, n_channels)

    # Compute state-specific coherence maps
    if vis_method == "nnmf":
        conn_maps = analysis.connectivity.mean_coherence_from_spectra(
            freqs, cohs, components=wb_comp,
        )
        # dim: (n_subjects, n_components, n_states, n_channels, n_channels)
    if vis_method == "manual":
        conn_maps = analysis.connectivity.mean_coherence_from_spectra(
            freqs, cohs, frequency_range=wb_freqs,
        )
        # dim: (n_subjects, n_states, n_channels, n_channels)

    # Compute fractional occupancies to be used as weights
    fo = modes.fractional_occupancies(stc) # shape: (n_subjects, n_states)
    gfo = np.average(fo, weights=weights, axis=0)

    # -------------- [5] Visualization -------------- #
    print("Step 5: Visualizing dynamic network features ...")

    # Set figure subdirectories
    os.makedirs(os.path.join(FIG_DIR, "analysis"), exist_ok=True)
    os.makedirs(os.path.join(FIG_DIR, "maps"), exist_ok=True)

    # Set up visualization tools
    DV = DynamicVisualizer()

    # Plot wide-band power maps (averaged over all subjects)
    print("(Step 5-1) Plotting power maps ...")
    
    gpower = np.average(power_maps, weights=weights, axis=0)
    if vis_method == "nnmf":
        gpower = gpower[0]
    # shape: (n_states, n_channels)

    DV.plot_power_map(
        power_map=gpower,
        filename=os.path.join(FIG_DIR, "maps/power_map.png"),
        subtract_mean=(False if model_type == "dynemo" else True),
        mean_weights=gfo,
        fontsize=30,
        plot_kwargs={"symmetric_cbar": True},
    )

    # Plot wide-band coherence maps (averaged over all subjects)
    print("(Step 5-2) Plotting coherence maps ...")
    
    gconn = np.average(conn_maps, weights=weights, axis=0)
    if vis_method == "nnmf":
        gconn = gconn[0]
    gconn -= np.average(gconn, weights=gfo, axis=0, keepdims=True)
    # shape: (n_states, n_channels, n_channels)
    
    gconn = analysis.connectivity.threshold(
        gconn, percentile=97, absolute_value=True,
    )  # select top 3%

    DV.plot_coh_conn_map(
        connectivity_map=gconn,
        filename=os.path.join(FIG_DIR, "maps/conn_map.png"),
    )

    # Plot power spectral densities (averaged over all subjects)
    print("(Step 5-3) Plotting PSDs ...")
    
    gpsds = np.average(psds, weights=weights, axis=0)
    # shape: (n_states, n_channels, n_freqs)
    mgpsds = np.average(gpsds, weights=gfo, axis=0)  # average over states
    # shape: (n_channels, n_freqs)
    
    # Average over channels
    p = np.mean(gpsds, axis=1)
    mp = np.mean(mgpsds, axis=0)

    DV.plot_psd(
        freqs=freqs, psd=p, mean_psd=mp,
        filename=os.path.join(FIG_DIR, "maps/psd.png"),
    )

    # Plot alpha/state time courses (for one subject)
    print("(Step 5-4) Plotting alpha/state time courses ...")
    n_samples = int(Fs * 5) # 5 seconds
    plot_alpha(
        alphas[0][:n_samples, :],
        stc[0][:n_samples, :],
        n_samples=n_samples,
        sampling_frequency=Fs,
        cmap="Set3",
        y_labels=["State Probabilities", "State Time Course"],
        title=f"{model_type.upper()}",
        fig_kwargs={"figsize": (11, 5)},
        filename=os.path.join(FIG_DIR, "analysis/alpha_stc.png"),
    )  # plots for one example subject

    print("Visualization complete.")
