"""TINDA analysis on the Nottingham MEGUK dataset."""

# Import packages
import os
import pandas as pd
from osl_dynamics.inference import modes
from utils import analysis as ua
from utils import plotting as up
from utils import data as ud
from utils.statistics import stat_ind_two_samples


if __name__ == "__main__":
    # -------------- [1] Settings -------------- #
    print("Step 1: Setting up ...")

    # Set directory paths
    BASE_DIR = "/well/woolrich/users/olt015/Cho2025_DyNeStE/nott_meguk"
    DATA_DIR = os.path.join(BASE_DIR, "results/{0}/{1}/run{2}")
    FIG_DIR = os.path.join(BASE_DIR, "figures/tinda")
    os.makedirs(FIG_DIR, exist_ok=True)

    # Set user-defined parameters
    data_type = "full"
    dyneste_run = 2  # best run ID for DyNeStE model
    hmm_run = 2  # best run ID for HMM model
    Fs = 250  # sampling frequency (unit: Hz)
    print(f"[INFO] Data type: {data_type} | DyNeStE Run ID: {dyneste_run} | HMM Run ID: {hmm_run}")

    # Set model directories (with selected model runs)
    dyneste_dir = DATA_DIR.format(data_type, "dyneste", dyneste_run)
    hmm_dir = DATA_DIR.format(data_type, "hmm", hmm_run)

    # Load inferred parameters
    print("(Step 1-1) Loading inferred parameters ...")
    dyneste_inf_params = ud.load(os.path.join(dyneste_dir, "inference/inf_params.pkl"))
    hmm_inf_params = ud.load(os.path.join(hmm_dir, "inference/inf_params.pkl"))

    # Get inferred and sampled state time courses
    dyneste_inf_stc = modes.argmax_time_courses(
        dyneste_inf_params["alpha"]
    )
    hmm_inf_stc = modes.argmax_time_courses(
        hmm_inf_params["alpha"]
    )
    # shape (*_inf_stc): (n_subjects, n_samples, n_states)

    dyneste_sam_stc = ud.load(os.path.join(dyneste_dir, "inference/samples.pkl"))["sam_stc"]
    hmm_sam_stc = ud.load(os.path.join(hmm_dir, "inference/samples.pkl"))["sam_stc"]
    # shape (*_sam_stc): (n_subjects, n_samples, n_states)

    # Validate state time courses
    stcs = [dyneste_inf_stc, hmm_inf_stc, dyneste_sam_stc, hmm_sam_stc]
    if len(set(len(l) for l in stcs)) != 1:
        raise ValueError(
            "state time courses must have the same number of subjects."
        )
    n_subjects = len(dyneste_inf_stc)  # number of subjects
    n_states = dyneste_inf_stc[0].shape[1]  # number of states
    print(f"[INFO] Number of subjects: {n_subjects} | Number of states: {n_states}")

    # Match state orders between two models
    print("(Step 1-2) Matching state orders ...")
    
    dyneste_order = ud.match_order(
        ref_info=("full", "dyneste", 2),
        cmp_info=(data_type, "dyneste", dyneste_run),
        method="covariances",
    )
    hmm_order = ud.match_order(
        ref_info=("full", "dyneste", 2),
        cmp_info=(data_type, "hmm", hmm_run),
        method="covariances",
    )

    reorder = lambda data, order: [d[:, order] for d in data]
    if dyneste_order is not None:
        print("Reordering states ...")
        print(f"\tOrder: {dyneste_order}")
        dyneste_inf_stc = reorder(dyneste_inf_stc, dyneste_order)
        dyneste_sam_stc = reorder(dyneste_sam_stc, dyneste_order)
    if hmm_order is not None:
        print("Reordering states ...")
        print(f"\tOrder: {hmm_order}")
        hmm_inf_stc = reorder(hmm_inf_stc, hmm_order)
        hmm_sam_stc = reorder(hmm_sam_stc, hmm_order)

    # -------------- [2] TINDA Analysis -------------- #
    print("Step 2: Performing TINDA analysis ...")

    # Run TINDA on the DyNeStE model
    print("(Step 2-1) Running TINDA on DyNeStE model ...")
    dyneste_tinda_path = os.path.join(dyneste_dir, "inference/tinda.pkl")

    if os.path.exists(dyneste_tinda_path):
        print("TINDA results already exist for DyNeStE. Loading ...")
        tinda_results = ud.load(dyneste_tinda_path)
        
        dyneste_inf_fo_density, dyneste_inf_tinda_stats, dyneste_inf_best_sequence, \
        dyneste_inf_asym_matrix = tinda_results["inference"].values()

        dyneste_sam_fo_density, dyneste_sam_tinda_stats, dyneste_sam_best_sequence, \
        dyneste_sam_asym_matrix = tinda_results["sample"].values()
    
    else:
        dyneste_inf_fo_density, dyneste_inf_tinda_stats, dyneste_inf_best_sequence, \
        dyneste_inf_asym_matrix = ua.run_tinda(dyneste_inf_stc)

        dyneste_sam_fo_density, dyneste_sam_tinda_stats, dyneste_sam_best_sequence, \
        dyneste_sam_asym_matrix = ua.run_tinda(dyneste_sam_stc)

        # Save results
        tinda_results = dict()
        tinda_results["inference"] = {
            "fo_density": dyneste_inf_fo_density,
            "tinda_stats": dyneste_inf_tinda_stats,
            "best_sequence": dyneste_inf_best_sequence,
            "asymmetry_matrix": dyneste_inf_asym_matrix,
            
        }
        tinda_results["sample"] = {
            "fo_density": dyneste_sam_fo_density,
            "tinda_stats": dyneste_sam_tinda_stats,
            "best_sequence": dyneste_sam_best_sequence,
            "asymmetry_matrix": dyneste_sam_asym_matrix,
        }
        ud.save(tinda_results, dyneste_tinda_path)

    # Run TINDA on the HMM model
    print("(Step 2-2) Running TINDA on HMM model ...")
    hmm_tinda_path = os.path.join(hmm_dir, "inference/tinda.pkl")

    if os.path.exists(hmm_tinda_path):
        print("TINDA results already exist for HMM. Loading ...")
        tinda_results = ud.load(hmm_tinda_path)

        hmm_inf_fo_density, hmm_inf_tinda_stats, hmm_inf_best_sequence, \
        hmm_inf_asym_matrix = tinda_results["inference"].values()
        
        hmm_sam_fo_density, hmm_sam_tinda_stats, hmm_sam_best_sequence, \
        hmm_sam_asym_matrix = tinda_results["sample"].values()

    else:
        hmm_inf_fo_density, hmm_inf_tinda_stats, hmm_inf_best_sequence, \
        hmm_inf_asym_matrix = ua.run_tinda(hmm_inf_stc)

        hmm_sam_fo_density, hmm_sam_tinda_stats, hmm_sam_best_sequence, \
        hmm_sam_asym_matrix = ua.run_tinda(hmm_sam_stc)

        # Save results
        tinda_results = dict()
        tinda_results["inference"] = {
            "fo_density": hmm_inf_fo_density,
            "tinda_stats": hmm_inf_tinda_stats,
            "best_sequence": hmm_inf_best_sequence,
            "asymmetry_matrix": hmm_inf_asym_matrix,
        }
        tinda_results["sample"] = {
            "fo_density": hmm_sam_fo_density,
            "tinda_stats": hmm_sam_tinda_stats,
            "best_sequence": hmm_sam_best_sequence,
            "asymmetry_matrix": hmm_sam_asym_matrix,
        }
        ud.save(tinda_results, hmm_tinda_path)

    # Find strongest edges
    print("(Step 2-3) Finding strongest edges ...")

    dyneste_inf_edges, dyneste_inf_edge_idx = ua.find_strongest_edges(
        dyneste_inf_fo_density, dyneste_inf_asym_matrix,
        state_sequence=dyneste_inf_best_sequence,
    )
    dyneste_sam_edges, dyneste_sam_edge_idx = ua.find_strongest_edges(
        dyneste_sam_fo_density, dyneste_sam_asym_matrix,
        state_sequence=dyneste_sam_best_sequence,
    )

    hmm_inf_edges, hmm_inf_edge_idx = ua.find_strongest_edges(
        hmm_inf_fo_density, hmm_inf_asym_matrix,
        state_sequence=hmm_inf_best_sequence,
    )
    hmm_sam_edges, hmm_sam_edge_idx = ua.find_strongest_edges(
        hmm_sam_fo_density, hmm_sam_asym_matrix,
        state_sequence=hmm_sam_best_sequence,
    )

    # Plot FO asymmetry matrix
    print("(Step 2-4) Plotting FO asymmetry matrix ...")
    
    up.plot_asymmetry_matrix(
        dyneste_inf_asym_matrix,
        dyneste_inf_best_sequence,
        edge_idx=dyneste_inf_edge_idx,
        vlims=[-0.022, 0.019],
        filename=os.path.join(FIG_DIR, "dyneste_inf_asym_matrix.png"),
    )
    up.plot_asymmetry_matrix(
        dyneste_sam_asym_matrix,
        dyneste_sam_best_sequence,
        edge_idx=dyneste_sam_edge_idx,
        vlims=[-0.022, 0.019],
        filename=os.path.join(FIG_DIR, "dyneste_sam_asym_matrix.png"),
    )

    up.plot_asymmetry_matrix(
        hmm_inf_asym_matrix,
        hmm_inf_best_sequence,
        edge_idx=hmm_inf_edge_idx,
        vlims=[-0.022, 0.019],
        filename=os.path.join(FIG_DIR, "hmm_inf_asym_matrix.png"),
    )
    up.plot_asymmetry_matrix(
        hmm_sam_asym_matrix,
        hmm_sam_best_sequence,
        edge_idx=hmm_sam_edge_idx,
        vlims=[-0.022, 0.019],
        filename=os.path.join(FIG_DIR, "hmm_sam_asym_matrix.png"),
    )

    # Plot TINDA cycles
    print("(Step 2-5) Plotting TINDA cycles ...")

    # Get strongest edges without reordering states by their best sequences
    dyneste_inf_edges, _ = ua.find_strongest_edges(
        dyneste_inf_fo_density, dyneste_inf_asym_matrix,
    )
    dyneste_sam_edges, _ = ua.find_strongest_edges(
        dyneste_sam_fo_density, dyneste_sam_asym_matrix,
    )
    hmm_inf_edges, _ = ua.find_strongest_edges(
        hmm_inf_fo_density, hmm_inf_asym_matrix,
    )
    hmm_sam_edges, _ = ua.find_strongest_edges(
        hmm_sam_fo_density, hmm_sam_asym_matrix,
    )

    up.plot_tinda_cycle(
        dyneste_inf_fo_density,
        dyneste_inf_best_sequence,
        dyneste_inf_edges,
        filename=os.path.join(FIG_DIR, "dyneste_inf_cycle.png"),
    )
    up.plot_tinda_cycle(
        dyneste_sam_fo_density,
        dyneste_sam_best_sequence,
        dyneste_sam_edges,
        filename=os.path.join(FIG_DIR, "dyneste_sam_cycle.png"),
    )

    up.plot_tinda_cycle(
        hmm_inf_fo_density,
        hmm_inf_best_sequence,
        hmm_inf_edges,
        filename=os.path.join(FIG_DIR, "hmm_inf_cycle.png"),
    )
    up.plot_tinda_cycle(
        hmm_sam_fo_density,
        hmm_sam_best_sequence,
        hmm_sam_edges,
        filename=os.path.join(FIG_DIR, "hmm_sam_cycle.png"),
    )

    # Compute cycle strengths
    dyneste_inf_cycle_strengths = ua.get_cycle_strengths(
        dyneste_inf_asym_matrix, dyneste_inf_best_sequence,
    )
    dyneste_sam_cycle_strengths = ua.get_cycle_strengths(
        dyneste_sam_asym_matrix, dyneste_sam_best_sequence,
    )
    # shape: (n_subjects,)

    hmm_inf_cycle_strengths = ua.get_cycle_strengths(
        hmm_inf_asym_matrix, hmm_inf_best_sequence,
    )
    hmm_sam_cycle_strengths = ua.get_cycle_strengths(
        hmm_sam_asym_matrix, hmm_sam_best_sequence,
    )
    # shape: (n_subjects,)

    # Perform statistical tests on cycle strengths
    _, inf_pval, _ = stat_ind_two_samples(
        dyneste_inf_cycle_strengths,
        hmm_inf_cycle_strengths,
        alpha=0.05,
    )

    _, sam_pval, _ = stat_ind_two_samples(
        dyneste_sam_cycle_strengths,
        hmm_sam_cycle_strengths,
        alpha=0.05,
    )

    # Plot cycle strengths
    print("(Step 2-6) Plotting cycle strengths ...")

    data_labels = ["Cycle Strengths", "Models", "Data Type"]
    df_cycle_strengths = pd.concat([
        ud.build_dataframe(
            dyneste_inf_cycle_strengths, "DyNeStE", "Inferred", labels=data_labels
        ),
        ud.build_dataframe(
            dyneste_sam_cycle_strengths, "DyNeStE", "Sampled", labels=data_labels
        ),
        ud.build_dataframe(
            hmm_inf_cycle_strengths, "HMM", "Inferred", labels=data_labels
        ),
        ud.build_dataframe(
            hmm_sam_cycle_strengths, "HMM", "Sampled", labels=data_labels
        ),
    ], ignore_index=True)

    up.plot_cycle_strengths(
        df_cycle_strengths,
        palette={"DyNeStE": "#6EB5C0", "HMM": "#B6443F"},
        p_vals=[inf_pval, sam_pval],
        fontsize=12,
        filename=os.path.join(FIG_DIR, "cycle_strengths.png"),
    )

    # -------------- [3] TINDA Quintile Analysis -------------- #
    print("Step 3: Performing TINDA quintile analysis ...")

    # NEED EDIT

    print("Analysis complete.")
