"""Functions for model loading and handling."""

import os
import pickle
import tensorflow as tf
from osl_dynamics.config_api.pipeline import load_config
from osl_dynamics.models.hmm import Config as HMMConfig, Model as HMMModel
from osl_dynamics.models.dyneste import Config as DyNeStEConfig, Model as DyNeStEModel


MODEL_CLASSES = {
    "HMM": (HMMConfig, HMMModel),
    "DyNeStE": (DyNeStEConfig, DyNeStEModel),
}


def get_config(config_path):
    """Gets a model config object from a dictionary.

    Parameters
    ----------
    config_path : str
        Path to the configuration file.

    Returns
    -------
    Config : osl_dynamics.models.<MODEL_NAME>.Config
        Config object.
    """

    config = load_config(config_path)
    model_name = config["model_name"]
    if model_name not in MODEL_CLASSES.keys():
        raise ValueError(f"model ({model_name}) not implemented.")
    
    Config = MODEL_CLASSES[model_name][0]
    return Config(**config)


def create_model(config_path, verbose=True):
    """Creates a model based on the configuration.

    Parameters
    ----------
    config : str
        Path to a configuration file.

    Returns
    -------
    model : osl.dynamics.models.<MODEL_NAME>.Model
        Model object.
    verbose : bool, optional
        Whether to print model summary.
    """

    # Load config
    config = get_config(config_path)
    model_name = config.model_name

    # Load adequate Model object
    if model_name not in MODEL_CLASSES.keys():
        raise ValueError(f"model {model_name} not implemented.")
    Model = MODEL_CLASSES[model_name][1]
    
    # Build model
    model = Model(config)
    if verbose:
        model.summary()
    return model


def load_model(model_dir, from_checkpoint=False):
    """Load a saved model from a directory.

    Parameters
    ----------
    model_dir : str
        Directory containing the saved model.
    from_checkpoint : bool, optional
        Whether to load the model from a checkpoint.

    Returns
    -------
    model : osl.dynamics.models.<MODEL_NAME>.Model
        Model object.
    """
    model = create_model(os.path.join(model_dir, "config.yml"))
    if from_checkpoint:
        checkpoint = tf.train.Checkpoint(
            model=model.model, optimizer=model.model.optimizer
        )
        checkpoint.restore(
            tf.train.latest_checkpoint(f"{model_dir}/checkpoints")
        ).expect_partial()
    else:
        model.load_weights(f"{model_dir}/model.weights.h5")

    try:
        with open(f"{model_dir}/history.pkl", "rb") as f:
            model.history = pickle.load(f)
    except FileNotFoundError:
        pass
    return model
