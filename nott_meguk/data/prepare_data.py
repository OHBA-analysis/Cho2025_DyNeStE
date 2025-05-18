"""Script for preparing and saving the data."""

# Import packages
import os
import pickle
import numpy as np
from glob import glob
from osl_dynamics.data import Data


# Define a utility function
def prepare_and_save_data(output_dir=".", split_half=False):
    # Get data paths
    DATA_DIR = "/well/woolrich/projects/toolbox_paper/ctf_rest/training_data/networks"
    data_paths = sorted(glob(f"{DATA_DIR}/*.npy"))

    # Split data if needed
    if split_half:
        n_data = len(data_paths)
        data_paths = [
            data_paths[: n_data // 2],
            data_paths[n_data // 2 :],
        ]
        output_dir = [f"{output_dir}/split1", f"{output_dir}/split2"]
    else:
        data_paths = [data_paths]
        output_dir = [f"{output_dir}/full"]

    for i, dp in enumerate(data_paths):
        # Create save directory if it does not exist
        save_dir = output_dir[i]
        os.makedirs(save_dir, exist_ok=True)

        # Build data object
        data = Data(
            dp,
            sampling_frequency=250,
            use_tfrecord=True,
            buffer_size=2000,
            n_jobs=12,
            mask_file="MNI152_T1_8mm_brain.nii.gz",
            parcellation_file="fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz",
            store_dir=f"tmp_meguk",
        )

        # Save unprepared data
        trimmed_ts = data.trim_time_series(
            n_embeddings=15, sequence_length=200, prepared=False
        )
        with open(f"{save_dir}/trimmed_ts.pkl", "wb") as f:
            pickle.dump(trimmed_ts, f)

        # Save prepared data
        data.prepare({
            "tde_pca": {"n_embeddings": 15, "n_pca_components": 80},
            "standardize": {},
        })
        data.save(save_dir)

        # Save principal components
        np.save(f"{save_dir}/pca_components.npy", data.pca_components)

        # Clean up temporary data directory
        data.delete_dir()


if __name__ == "__main__":
    # Prepare and save data
    prepare_and_save_data()  # full data
    prepare_and_save_data(split_half=True)  # split-half data

    print("Data preparation complete.")
