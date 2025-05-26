"""Functions for data manipulation."""

import pickle
import numpy as np
from osl_dynamics.inference import modes


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


def match_order(ref_info, cmp_info, method="covariances"):
    """Get the state/mode order for a specified model run 
       compared to the reference model run.

    Parameters
    ----------
    ref_info : tuple
        Data type, model type, and run ID of the reference model.
    cmp_info : tuple
        Data type, model type, and run ID of the model to compare with the reference.
    method : str
        Method to use for matching the state/mode order.

    Returns
    -------
    order : list or None
        State/mode order of the specified model run.
        Returns None if the specified model run is the reference.
    """

    if ref_info == cmp_info:
        return None
    
    ref_data_type, ref_model_type, ref_id = ref_info
    cmp_data_type, cmp_model_type, cmp_id = cmp_info

    # Load inferred parameters
    BASE_DIR = "/well/woolrich/users/olt015/Cho2025_DyNeStE/nott_meguk"
    data_path = BASE_DIR + "/results/{}/{}/run{}/inference/inf_params.pkl"
    ref_params = load(data_path.format(ref_data_type, ref_model_type, ref_id))
    cmp_params = load(data_path.format(cmp_data_type, cmp_model_type, cmp_id))

    # Get alphas
    ref_alphas = ref_params["alpha"]
    cmp_alphas = cmp_params["alpha"]

    # Get state/mode order
    if method == "covariances":
        ref_gfo = np.mean(
            modes.fractional_occupancies(ref_alphas), axis=0
        )  # shape: (n_states,)
        cmp_gfo = np.mean(
            modes.fractional_occupancies(cmp_alphas), axis=0
        )  # shape: (n_states,)
        demean = lambda x, w: x - np.average(x, weights=w, axis=0, keepdims=True)
        ref_cov = demean(ref_params["covariance"], ref_gfo)
        cmp_cov = demean(cmp_params["covariance"], cmp_gfo)
        # shape: (n_states, n_channels, n_channels)
        order = modes.match_covariances(ref_cov, cmp_cov, return_order=True)[1]
    
    if method == "modes":
        # Validate the number of subjects
        if len(ref_alphas) != len(cmp_alphas):
            raise ValueError("the number of subjects in the reference and comparison models do not match.")
        n_subjects = len(ref_alphas)

        # Adjust the length of subject-wise alphas
        for n in range(n_subjects):
            n_keep = min(ref_alphas[n].shape[0], cmp_alphas[n].shape[0])
            ref_alphas[n] = ref_alphas[n][:n_keep]
            cmp_alphas[n] = cmp_alphas[n][:n_keep]
        # NOTE: If different number of time-delay embeddings or sliding window was used during data preparation, 
        #       this adjustment is not sufficient, and the matched order is likely to be flawed. This only adjusts
        #       for the use of different sequence lengths.

        # Get the network order
        order = modes.match_modes(
            np.concatenate(ref_alphas, axis=0),
            np.concatenate(cmp_alphas, axis=0),
            return_order=True,
        )[1]
        # alpha time courses are concatenated subject-wise

    return order
