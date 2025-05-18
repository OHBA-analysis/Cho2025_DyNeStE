"""Run DyNeStE on the Nottingham MEGUK dataset."""

# Import packages
import os
from glob import glob
from sys import argv
from osl_dynamics.data import Data
from osl_dynamics.models.dyneste import Config, Model
from utils.data import save


if __name__ == "__main__":
    # -------------- [1] Settings -------------- #
    print("Step 1: Setting up ...")

    # Set user defined arguments
    if len(argv) != 3:
        raise ValueError(
            f"Need to pass two arguments: run ID and data type (e.g., python {argv[0]} 0 full)"
        )
    run_id = int(argv[1])  # run ID
    data_type = argv[2]  # data type

    # Validate user inputs
    if data_type not in ["full", "split1", "split2"]:
        raise ValueError(
            "Data type must be one of ['full', 'split1', 'split2']."
        )

    # Set output directories
    BASE_DIR = "/well/woolrich/users/olt015/Cho2025_DyNeStE/nott_meguk"
    SAVE_DIR = os.path.join(BASE_DIR, f"results/{data_type}/dyneste/run{run_id}")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Set output subdirectories
    model_dir = os.path.join(SAVE_DIR, "model")
    inference_dir = os.path.join(SAVE_DIR, "inference")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(inference_dir, exist_ok=True)

    # -------------- [2] Training Configurations -------------- #
    print("Step 2: Defining training configurations ...")

    # Define model hyperparameters
    config = Config(
        n_states=12,
        n_channels=80,
        sequence_length=200,
        inference_n_units=128,
        inference_normalization="layer",
        model_n_units=128,
        model_normalization="layer",
        learn_means=False,
        learn_covariances=True,
        gradient_clip=10.0,
        do_kl_annealing=True,
        kl_annealing_curve="tanh",
        kl_annealing_sharpness=10,
        n_kl_annealing_epochs=50,
        do_gs_annealing=True,
        gs_annealing_curve="exp",
        initial_gs_temperature=1.0,
        final_gs_temperature=0.06,
        gs_annealing_slope=0.014,
        n_gs_annealing_epochs=350,
        batch_size=128,
        learning_rate=1e-3,
        lr_decay=5e-3,
        n_epochs=350,
    )

    # -------------- [3] Model Training -------------- #
    print("Step 3: Training the model ...")

    # Load training dataset
    file_names = sorted(glob(
        os.path.join(BASE_DIR, f"data/{data_type}/array*.npy")
    ))
    training_data = Data(
        file_names,
        store_dir=f"tmp_{data_type}_dyneste_{run_id}",
    )

    # Build model
    model = Model(config)
    model.summary()

    # Add regularization for the observation model
    model.set_regularizers(training_data)

    # Train model
    init_history = model.random_subset_initialization(
        training_data,
        n_init=10,
        n_epochs=2,
        take=1,
        do_gs_annealing=False,
    )  # initialization
    history = model.fit(training_data)  # full training

    # Save trained model
    model.save(os.path.join(model_dir, "trained_model"))

    # Save training history
    save(init_history, os.path.join(model_dir, "init_history.pkl"))
    save(history, os.path.join(model_dir, "history.pkl"))

    # ----------- [5] Save Results -------------- #
    print("Step 5: Saving results ...")

    # Get free energy
    free_energy = model.free_energy(training_data)
    print("Variational Free Energy: ", free_energy)

    # Get inferred state probabilities for each subject
    alphas = model.get_alpha(training_data)

    # Get inferred state covariances
    covs = model.get_covariances()

    # Save results
    outputs = {
        "free_energy": free_energy,
        "alpha": alphas,
        "covariance": covs,
    }
    save(outputs, os.path.join(inference_dir, "inf_params.pkl"))

    # Delete temporary data directory
    training_data.delete_dir()

    print("[DyNeStE] Model training complete.")
