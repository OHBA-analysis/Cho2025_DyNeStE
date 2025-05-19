"""Functions for data manipulation."""

import os
import pickle


def save(data, save_path):
    """Saves data to a specified path using pickle.

    This is a wrapper function for the pickle module
    to save data in a binary format.

    Parameters
    ----------
    data : object
        Data to be saved. Can be any Python object.
    save_path : str
        Path where the data will be saved. The file extension should be .pkl.
    """
    if not save_path.endswith(".pkl"):
        raise ValueError("The file extension should be .pkl.")

    with open(save_path, "wb") as output_handle:
        pickle.dump(data, output_handle)
    output_handle.close()


def load(save_path):
    """Loads data from a specified path using pickle.

    Parameters
    ----------
    save_path : str
        Path from where the data will be loaded.
        The file extension should be .pkl.

    Returns
    -------
    data : object
        Data loaded from the file. Can be any Python object.
    """
    if not save_path.endswith(".pkl"):
        raise ValueError("The file extension should be .pkl.")

    with open(save_path, "rb") as input_handle:
        data = pickle.load(input_handle)
    input_handle.close()

    return data


def load_inf_params(dir_path, model_type, run_id):
    """Loads inferred parameters from a given directory.

    Parameters
    ----------
    dir_path : str
        Directory path where the parameters are saved.
    model_type : str
        Type of the model used to infer the parameters.
    run_id : int
        Run ID of the model.

    Returns
    -------
    inf_params : dict
        Dictionary containing the inferred parameters.
    """
    dir_path = os.path.join(dir_path, "{0}/run{1}")
    dir_name = dir_path.format(model_type, run_id)
    inf_params = load(os.path.join(dir_name, "inf_params.pkl"))
    return inf_params
