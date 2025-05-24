"""Script for loading and visualizing the subject demographics."""

# Import packages
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from glob import glob


if __name__ == "__main__":
    # Define data directory
    FIG_DIR = "/well/woolrich/users/olt015/Cho2025_DyNeStE/nott_meguk/data"
    RAW_DIR = "/well/woolrich/projects/mrc_meguk/public/"
    DATA_DIR = "/well/woolrich/projects/toolbox_paper/ctf_rest"

    raw_data_path = os.path.join(RAW_DIR, "Demographics/nottingham_demographics_anonymised.csv")
    meta_data_path = os.path.join(DATA_DIR, "training_data/demographics.csv")
    if not os.path.exists(meta_data_path):
        raise FileNotFoundError(f"Demographics file not found at {meta_data_path}.")
    
    # Read demographics data
    raw_data = pd.read_csv(raw_data_path)  # raw demographics data
    demographics = pd.read_csv(meta_data_path)  # processed demographics data

    # Get demographic features
    ids = demographics["name"].to_list()
    genders = demographics["gender"].to_list()
    handedness = demographics["handedness"].to_list()

    subjects = []
    for file in sorted(glob(os.path.join(DATA_DIR, "src/*/sflip_parc-raw.fif"))):
        subjects.append(file.split("/")[-2][4:])

    ages = []
    for s in subjects:
        age_range = raw_data.loc[raw_data["id"] == s]["age_range"].values[0]
        ages.append(age_range)
    # NOTE: We take the ages from the raw meta data, because demogrpahics.csv stores
    #       the mean values of age ranges.

    # Validate demographics data
    if np.any(handedness == 0):
        warnings.warn(
            "Some handedness values are 0. Assuming right-handed for these subjects.",
            Warning,
        )
    
    # Create dataframes
    genders = ["Male" if g == 1 else "Female" for g in genders]
    handedness = ["Left" if h < 0 else "Right" for h in handedness]

    df_age = pd.DataFrame({"Age": ages})
    df_age["Age"] = pd.Categorical(
        df_age["Age"],
        categories=["18-24", "24-30", "30-36", "36-42", "42-48", "48-54", "54-60", "60+"],
        ordered=True,
    )

    df_gender = pd.DataFrame({"Sex": genders})
    df_gender["Sex"] = pd.Categorical(
        df_gender["Sex"],
        categories=["Female", "Male"],
        ordered=True,
    )

    df_handedness = pd.DataFrame({"Handedness": handedness})
    df_handedness["Handedness"] = pd.Categorical(
        df_handedness["Handedness"],
        categories=["Left", "Right"],
        ordered=True,
    )

    # Set visualization hyperparameters
    fontsize=13
    facecolor="#114B5F"
    edgecolor="none"

    # Visualize demographics
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3))

    sns.countplot(
        data=df_age, x="Age",
        color=facecolor, edgecolor=edgecolor,
        width=0.7, alpha=0.75, ax=ax[0],
    )
    ax[0].tick_params(axis="x", labelrotation=45)
    ax[0].set_xlabel("Age (years)", fontsize=fontsize)

    sns.countplot(
        data=df_gender, x="Sex",
        color=facecolor, edgecolor=edgecolor,
        width=0.5, alpha=0.75, ax=ax[1],
    )
    ax[1].set_xlabel("Sex", fontsize=fontsize)

    sns.countplot(
        data=df_handedness, x="Handedness",
        color=facecolor, edgecolor=edgecolor,
        width=0.5, alpha=0.75, ax=ax[2],
    )
    ax[2].set_xlabel("Handedness", fontsize=fontsize)

    for axis in ax:
        axis.set_ylabel("Count", fontsize=fontsize)
        axis.tick_params(
            axis="both", which="major", width=1.5, labelsize=fontsize
        )
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["bottom", "left"]].set_linewidth(1.5)

    # fig.suptitle(
    #     "Nottingham MEGUK Dataset Subject Demographics",
    #     fontsize=fontsize + 2, fontweight="bold",
    # )  # set suptitle

    plt.tight_layout()
    fig.savefig(
        os.path.join(FIG_DIR, "demographics.png"),
        dpi=300, bbox_inches="tight", transparent=False,
    )
    plt.close(fig)

    print("Analysis complete.")
