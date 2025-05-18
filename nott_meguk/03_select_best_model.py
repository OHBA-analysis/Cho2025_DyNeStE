"""Select the best model run."""

# Import packages
import os
import numpy as np
from sys import argv
from utils.data import load


if __name__ == "__main__":
    # -------------- [1] Settings -------------- #
    print("Step 1: Setting up ...")

    # Set user defined arguments
    if len(argv) != 4:
        raise ValueError(
            "Need to pass three arguments: model type, run IDs, and data type" +
            f" (e.g., python {argv[0]} dyneste 0-9 full)"
        )
    model_type = argv[1]  # model type
    run_ids = list(map(int, argv[2].split("-")))  # range of runs to compare
    data_type = argv[3]  # data type
    print(f"[INFO] Model Type: {model_type} | Run: run{run_ids[0]} - run{run_ids[1]}" 
          + f" | Data Type: {data_type}")

    # Validate user inputs
    if data_type not in ["full", "split1", "split2"]:
        raise ValueError("Data type must be one of ['full', 'split1', 'split2'].")

    # Set data directories and file paths
    BASE_DIR = "/well/woolrich/users/olt015/Cho2025_DyNeStE/nott_meguk"
    DATA_DIR = os.path.join(BASE_DIR, f"results/{data_type}/{model_type}/run{{0}}")
    history_path = os.path.join(DATA_DIR, "model/history.pkl")
    inference_path = os.path.join(DATA_DIR, "inference/inf_params.pkl")

    # -------------- [2] Best Model Selection -------------- #
    print("Step 2: Selecting best model run ...")

    # If the range of runs is larger than 10, group 10 model runs as one set
    if np.diff(run_ids) + 1 > 10:
        intervals = [
            [i, min(i + 9, run_ids[1])]
            for i in range(run_ids[0], run_ids[1] + 1, 10)
        ]
    else: intervals = [run_ids]

    # Get the best model run
    best_runs, best_fes = [], []
    for i, (start, end) in enumerate(intervals):
        print(f"Loading free energy (run{start}-run{end}) ...")
        free_energy, loss = [], []
        run_id_list = np.arange(start, end + 1)
        for id in run_id_list:
            history = load(history_path.replace("{0}", str(id)))
            inf_params = load(inference_path.replace("{0}", str(id)))
            free_energy.append(inf_params["free_energy"])
            loss.append(history["loss"][-1])
        best_fes.append(np.min(free_energy))
        best_runs.append(run_id_list[free_energy.index(best_fes[i])])
        print(f"\tFree energy (n={len(run_id_list)}): {free_energy}")
        print(f"\tFinal loss (n={len(run_id_list)}): {loss}")
        print(f"\tBest run: run{best_runs[i]}")
        print(f"\tBest free energy: {best_fes[i]}")

    # Identify the optimal run from all the best runs
    opt_fe = np.min(best_fes)
    opt_run = best_runs[np.argmin(best_fes)]
    print(f"The lowest free energy is {opt_fe} from run{opt_run}.")

    print("Selection complete.")
