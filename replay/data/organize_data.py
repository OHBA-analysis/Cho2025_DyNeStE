"""Organize the raw data into a numpy format for its use in the BMRC server.

This post-processes the .mat data files stored in the hbaws server.
"""

# Import packages
import os
import re
import pickle
import numpy as np
from glob import glob
from scipy.io import loadmat


if __name__ == "__main__":
    # -------------- [1] Settings -------------- #
    print("Step 1: Setting up ...")

    # Set up directories
    BASE_DIR = "/well/woolrich/users/olt015/Cho2025_DyNeStE/replay/data"
    DATA_DIR = os.path.join(BASE_DIR, "raw")

    # Get data subdirectories
    subdirectories = os.listdir(DATA_DIR)
    subdirectories.sort()

    # -------------- [2] Data Organization -------------- #
    print("Step 2: Organizing data ...")

    for dir_name in subdirectories:
        print(f"[INFO] Organizing data in {dir_name} ...")

        # Create a directory to save the organized data
        SAVE_DIR = os.path.join(BASE_DIR, dir_name)
        os.makedirs(SAVE_DIR, exist_ok=True)

        # Get data files
        session_regex = re.compile(r"session(\d+)")
        data_files = sorted(
            glob(os.path.join(DATA_DIR, dir_name, "parceldata*.mat")),
            key=lambda path: int(session_regex.search(path).group(1)) if session_regex.search(path) else float('inf'),
        )
        n_sessions = len(data_files)

        # Parse data
        parse_indices = []
        for i, file in enumerate(data_files):
            print(f"\tProcessing file: '{file}'")

            # Load data
            data = loadmat(file)["data"]

            # Find an index where samples become all zero across channels
            all_zero_mask = np.all(data == 0, axis=0) # shape: (n_samples,)
            zero_start_idx = np.where(~all_zero_mask)[0][-1] + 1
            parse_indices.append(zero_start_idx)

            # Remove the zero samples and transpose the data
            data = data[:, :zero_start_idx].T # shape: (n_samples, n_channels)

            # Save data as a numpy array
            save_name = f"array{i:0{len(str(n_sessions))}d}.npy"
            np.save(os.path.join(SAVE_DIR, save_name), data)

        # Get replay scores
        replay_file = os.path.join(DATA_DIR, dir_name, "replay_scores_index.mat")
        replay_scores_index = np.squeeze(loadmat(replay_file)["replay_scores_index"])

        if len(replay_scores_index) != n_sessions:
            raise ValueError("Number of replay scores and sessions do not match.")
        
        # Parse replay scores
        replay_indices = []
        for i, score in enumerate(replay_scores_index):
            score = np.squeeze(score)
            score -= 1 # convert to 0-based indexing (MATLAB -> Python)
            parsed_score = score[score <= parse_indices[i] - 1] # remove scores that exceed the data length
            replay_indices.append(parsed_score)
            
        # Save replay score indices as a list
        with open(os.path.join(SAVE_DIR, "replay_indices.pkl"), "wb") as f:
            pickle.dump(replay_indices, f)

    print("Data organization complete.")
