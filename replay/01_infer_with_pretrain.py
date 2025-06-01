"""Run inference on the replay dataset with a pre-trained model."""

# Import packages
import os
import glob
import numpy as np
from sys import argv
from osl_dynamics.data import Data
from osl_dynamics.models import load
from utils.data import save


if __name__ == "__main__":
    # -------------- [1] Settings -------------- #
    print("Step 1: Setting up ...")

    # Set user arguments
    if len(argv) != 4:
        raise ValueError(
            "Need to pass three arguments: model type, run ID, and data type "
            + f"(e.g., python {argv[0]} hmm 0 study1)"
        )
    model_type = argv[1]  # model_type
    run_id = int(argv[2])  # run ID
    data_type = argv[3]  # data type

    # Validate user inputs
    if model_type not in ["hmm", "dyneste"]:
        raise ValueError(
            f"Invalid model type: {model_type}. Must be one of ['hmm', 'dyneste']."
        )
    if data_type not in ["study1", "study2"]:
        raise ValueError(
            f"Invalid data type: {data_type}. Must be one of ['study1', 'study2']."
        )

    # Define pre-trained model runs
    if model_type == "hmm":
        pretrain_run_id = 2
    if model_type == "dyneste":
        pretrain_run_id = 2

    # Set directories to load model and data
    BASE_DIR = "/well/woolrich/users/olt015/Cho2025_DyNeStE/replay"
    PRETRAIN_DIR = os.path.join(os.path.dirname(BASE_DIR), "nott_meguk")
    MODEL_DIR = os.path.join(
        PRETRAIN_DIR,
        f"results/full/{model_type}/run{pretrain_run_id}/model/trained_model",
    )  # pre-trained model to use
    DATA_DIR = os.path.join(BASE_DIR, f"data/{data_type}")

    # Set directories to save results
    save_dir = os.path.join(
        BASE_DIR, f"results/{data_type}/{model_type}/run{run_id}"
    )
    os.makedirs(save_dir, exist_ok=True)

    # -------------- [2] Inference with Pre-Trained Model -------------- #
    print("Step 2: Applying the pre-trained model ...")

    # Load pre-trained model
    model = load(MODEL_DIR)
    model.summary()

    # Load data
    file_names = sorted(glob.glob(os.path.join(DATA_DIR, "array*.npy")))
    training_data = Data(
        file_names,
        store_dir=f"tmp_{data_type}_{model_type}_{run_id}",
    )

    # Get principal components used for the pre-training
    pca_components = np.load(
        os.path.join(PRETRAIN_DIR, "data/full/pca_components.npy")
    )

    # Prepare data
    prepare_config = {
        "tde_pca": {"n_embeddings": 15, "pca_components": pca_components},
        "standardize": {},
    }
    training_data.prepare(prepare_config)

    # Perform inference
    alpha = model.get_alpha(training_data)
    free_energy = model.free_energy(training_data)
    print(f"Free energy: {free_energy}")

    # Save inferred parameters
    outputs = {
        "alpha": alpha,
        "free_energy": free_energy,
    }
    save(outputs, os.path.join(save_dir, "inf_params.pkl"))
    
    # Save trimmed time series
    trimmed_ts = training_data.trim_time_series(
        n_embeddings=15,
        sequence_length=model.config.sequence_length,
        prepared=False,
    )
    # NOTE: Here, the length of time series accounts for the number of embeddings
    #       and the sequence length of the model.
    save(trimmed_ts, os.path.join(save_dir, "trimmed_ts.pkl"))

    # Delete temporary data directory
    training_data.delete_dir()

    print("[Replay Study] Model inference complete.")
