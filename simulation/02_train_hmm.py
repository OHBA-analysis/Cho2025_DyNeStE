"""Script for training an HMM model on a simulated data."""

# Import packages
import os
from sys import argv
from osl_dynamics import data, simulation
from osl_dynamics.inference import modes, metrics
from osl_dynamics.models.hmm import Config, Model
from osl_dynamics.utils import plotting, set_random_seed
from utils.data import save


if __name__ == "__main__":
    # -------------- [1] Settings -------------- #
    print("Step 1: Setting up ...")

    # Set run ID
    if len(argv) != 2:
        raise ValueError(
            f"Need to pass one argument: run ID (e.g., python {argv[0]} 0)"
        )
    run_id = int(argv[1])  # run ID

    # Set random seed
    set_random_seed(run_id, op_determinism=True)

    # Set directories to store outputs
    BASE_DIR = "/well/woolrich/users/olt015/Cho2025_DyNeStE/simulation"
    SAVE_DIR = os.path.join(BASE_DIR, f"results/results_hmm_{run_id}")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # -------------- [2] Training Configurations -------------- #
    print("Step 2: Defining training configurations ...")

    # Define model hyperparameters
    config = Config(
        n_states=3,
        n_channels=11,
        sequence_length=200,
        learn_means=False,
        learn_covariances=True,
        batch_size=16,
        learning_rate=0.01,
        n_epochs=20,
    )

    # -------------- [3] Data Simulation -------------- #
    print("Step 3: Simulating data ...")

    # Simulate data
    sim = simulation.HSMM_MVN(
        n_samples=25600,
        n_states=config.n_states,
        n_channels=config.n_channels,
        means="zero",
        covariances="random",
        observation_error=0.0,
        gamma_shape=10,
        gamma_scale=5,
    )
    sim.standardize()  # prepare data

    # Plot the ground truth transition probability matrix
    plotting.plot_matrices(
        sim.off_diagonal_trans_prob,
        filename=os.path.join(SAVE_DIR, "sim_trans_prob.png"),
    )  # describes state switching in the HSMM

    # Get Data object for training
    training_data = data.Data(
        sim.time_series,
        store_dir=f"tmp_hmm_{run_id}",
    )

    # Create tensorflow datasets for training and model evaluation
    training_dataset = training_data.dataset(
        config.sequence_length, config.batch_size, shuffle=True
    )
    prediction_dataset = training_data.dataset(
        config.sequence_length, config.batch_size, shuffle=False
    )

    # -------------- [4] Model Training -------------- #
    print("Step 4: Training the model ...")

    # Build model
    model = Model(config)
    model.summary()

    # Add regularization for the observation model
    model.set_regularizers(training_dataset)

    # Train model
    init_history = model.random_state_time_course_initialization(
        training_dataset,
        n_init=5,
        n_epochs=2,
        take=1,
    )  # initialization
    history = model.fit(training_dataset)  # full training

    # Save trained model
    model.save(os.path.join(SAVE_DIR, "trained_model"))

    # Save training history
    save(init_history, os.path.join(SAVE_DIR, "init_history.pkl"))
    save(history, os.path.join(SAVE_DIR, "history.pkl"))

    # ----------- [5] Summarize inferences -------------- #
    print("Step 5: Summarizing inferences ...")

    # Calculate the free energy
    free_energy = model.free_energy(prediction_dataset)
    print("Variational Free Energy: ", free_energy)

    # Infer alpha and state time courses
    inf_alp = model.get_alpha(prediction_dataset)
    inf_stc = modes.argmax_time_courses(inf_alp)
    sim_stc = sim.mode_time_course

    # Infer state-specific covariances
    sim_cov = sim.covariances
    inf_cov = model.get_covariances()

    # Match state orders between simulation and inference
    order = modes.match_modes(sim_stc, inf_stc, return_order=True)[1]
    inf_alp = inf_alp[:, order]
    inf_stc = inf_stc[:, order]
    inf_cov = inf_cov[order]

    # Calculate the Dice-Sørensen coefficient
    dice = metrics.dice_coefficient(sim_stc, inf_stc)
    print("Dice-Sørensen coefficient: ", dice)

    # Sample a state time course with the inferred transition probability matrix
    trans_prob = model.get_trans_prob()
    sam_stc = model.sample_state_time_course(25600)

    # Save outputs
    outputs = {
        # Performance metrics
        "free_energy": free_energy,
        "dice_coefficient": dice,
        # Inferred parameters (temporal)
        "sim_stc": sim_stc,
        "inf_alp": inf_alp,
        "inf_stc": inf_stc,
        "sam_stc": sam_stc,
        "transition_probability": trans_prob,
        # Inferred parameters (spatial)
        "sim_cov": sim_cov,
        "inf_cov": inf_cov,
    }

    save(outputs, os.path.join(SAVE_DIR, "inf_params.pkl"))

    # Delete temporary data directory
    training_data.delete_dir()

    print("[HMM] Model training complete.")
