# Replay

### 🔎 Overview

This directory contains scripts for model inference and data analysis on the Replay dataset. For details about the dataset, please refer to Liu et al. ([2019](https://doi.org/10.1016/j.cell.2019.06.012)) and Higgins et al. ([2021](https://doi.org/10.1016/j.neuron.2020.12.007)).

They are organised into three categories:

1. Data Organisation
   * For organising the dataset
2. Model Inference
   * For performing inference using pre-trained DyNeStE and TDE-HMM models
3. Data Analysis
   * For analysing and visualising model inferences

For detailed descriptions of each script, please refer to the sections below.

## 🗄️ Data Organisation

The `data` folder includes a script called `organize_data.py`. This script processes and saves resting-state MEG data and the replay event indices extracted from it for subsequent model training and post hoc analysis.

## ⚙️ Model Inference

Instead of training new models from scratch, we use DyNeStE and TDE-HMM models pre-trained on the Nottingham MEGUK dataset to infer network states from the Replay dataset. Because this step performs inference only, we skip multiple training runs and concomitant model selection.

| Scripts                      | Description                                                       |
| :--------------------------- | :---------------------------------------------------------------- |
| `01_infer_with_pretrain.py`  | Loads a pre-trained model and performs inference on the dataset.  |

## 🧐 Data Analysis

For the data analysis, we have three main scripts:

| Scripts                               | Description                                                                     | Figures |
| :------------------------------------ | :------------------------------------------------------------------------------ | :------ |
| `02_analyze_network_descriptions.py`  | Computes dynamic resting state networks and their network profiles.             | A7, A8  |
| `03_analyze_replay_network.py`        | Analyses replay-evoked network dynamics and computes relevant summary metrics.  | -       |
| `04_visualize_replay_network.py`.     | Visualises replay-evoked network dynamics and post hoc analysis results.        | 10      |

**NOTE:** The figures in the Appendix are indicated by the prefix "A".

### 🙋‍♂️ FAQ: What about the `utils` subdirectory?
The `utils` subdirectory contains essential functions required to run the scripts summarised above. Each script in `utils` includes multiple functions. These functions are self-explanatory and include detailed annotations, so their descriptions are not repeated here.
