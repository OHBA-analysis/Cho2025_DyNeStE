"""Compare model run-to-run variability."""

# Import packages
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MaxNLocator
from sys import argv
from utils import plotting as up


if __name__ == "__main__":
    # -------------- [1] Settings -------------- #
    print("Step 1: Setting up ...")

    # Set user defined arguments
    if len(argv) != 3:
        raise ValueError(
            "Need to pass two arguments: run IDs and data type" +
            f" (e.g., python {argv[0]} 0-9 full)"
        )
    run_ids = list(map(int, argv[1].split("-")))  # range of runs to compare
    data_type = argv[2]  # data type
    print(f"[INFO] Run: run{run_ids[0]} - run{run_ids[1]} | Data Type: {data_type}")

    n_runs = int(np.diff(run_ids)) + 1

    # Validate user inputs
    if data_type not in ["full", "split1", "split2"]:
        raise ValueError("Data type must be one of ['full', 'split1', 'split2'].")

    # Set data directories and file paths
    BASE_DIR = "/well/woolrich/users/olt015/Cho2025_DyNeStE/nott_meguk"
    DYNESTE_DIR = os.path.join(BASE_DIR, f"results/{data_type}/dyneste")
    HMM_DIR = os.path.join(BASE_DIR, f"results/{data_type}/hmm")
    FIG_DIR = os.path.join(BASE_DIR, f"figures")

    # -------------- [2] Model Comparison -------------- #
    print("Step 2: Comparing model run-to-run variability ...")

    # Define file name to load
    file_name = f"final_loss_{run_ids[0]}-{run_ids[1]}.npy"
    
    # Load final losses
    dyneste_loss = np.load(os.path.join(DYNESTE_DIR, file_name))
    hmm_loss = np.load(os.path.join(HMM_DIR, file_name))
    # shape (*_loss): (n_runs,)

    print("*** DyNeStE Final Loss ***")
    print("\tMean: ", np.mean(dyneste_loss))
    print("\tStd: ", np.std(dyneste_loss))

    print("*** HMM Final Loss ***")
    print("\tMean: ", np.mean(hmm_loss))
    print("\tStd: ", np.std(hmm_loss))

    # -------------- [3] Visualization -------------- #
    print("Step 3: Visualizing model run-to-run variability ...")

    # Set visualization parameters
    box_args = {"saturation": 0.4, "linewidth": 1.5, "width": 0.6, "zorder": 2}
    strip_args = {"size": 7, "linewidth": 1.5, "alpha": 0.8, "jitter": 0.15}

    # Plot box plots
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))
    sns.boxplot(y=dyneste_loss, **box_args, color="#6EB5C0", ax=ax[0])
    sns.boxplot(y=hmm_loss, **box_args, color="#B6443F", ax=ax[1])

    # Add individual data points
    sns.stripplot(
        y=dyneste_loss, **strip_args, color="#6EB5C0", ax=ax[0]
    )
    sns.stripplot(
        y=hmm_loss, **strip_args, color="#B6443F", ax=ax[1]
    )

    # Adjust axis settings
    ax[0].set_xlabel(f"DyNeStE (n={n_runs})", fontsize=12)
    ax[1].set_xlabel(f"HMM (n={n_runs})", fontsize=12)
    ax[0].set_ylabel("Final Training Loss", fontsize=12)
    for axis in ax:
        axis.spines[["left", "bottom"]].set_linewidth(1.5)
        axis.spines[["right", "top"]].set_visible(False)
        axis.tick_params(labelsize=12)
    ax[1].yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax[1].yaxis.set_major_formatter(ScalarFormatter())
    ax[1].ticklabel_format(axis="y", style="plain", useOffset=False)

    # Save the figure
    plt.tight_layout()
    up.save(fig, f"{FIG_DIR}/model_variability.png")

    print("Comparison complete.")
