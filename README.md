# Cho2025_DyNeStE

This repository contains the scripts and data for reproducing results in the "*Discrete Representation of Long-Range Brain Network Dynamics via Generative Modelling*" paper.

💡 Please email SungJun Cho at sungjun.cho@ndcn.ox.ac.uk or simply raise GitHub Issues if you have any questions or concerns.

## ⚡️ Getting Started

This repository contains all the scripts necessary to reproduce the analyses and figures presented in the manuscript. It is divided into three main directories.

| Directory             | Description                                                                                              |
| :-------------------- | :------------------------------------------------------------------------------------------------------- |
| `simulation`          | Scripts for training DyNeStE and TDE-HMM models on simulated data and subsequent analysis.               |
| `nott_meguk`          | Scripts for training DyNeStE and TDE-HMM models on the Nottingham MEGUK dataset and subsequent analysis. |
| `replay`              | Scripts for inference and analysis on the Replay dataset using pre-trained DyNeStE and TDE-HMM models.   |

For detailed descriptions of the scripts in each directory, please consult the README file located within each respective folder.

**NOTE:** All the codes within this repository were executed on the Oxford Biomedical Research Computing (BMRC) servers. While individual threads were allocated varying CPUs and GPUs, general information about the BRMC resources can be found at [_Using the BMRC Cluster with Slurm_](https://www.medsci.ox.ac.uk/for-staff/resources/bmrc/using-the-bmrc-cluster-with-slurm) and [_GPU Resources_](https://www.medsci.ox.ac.uk/for-staff/resources/bmrc/gpu-resources).

## 🎯 Requirements
To start, you first need to install the [`osl-dynamics`](https://github.com/OHBA-analysis/osl-dynamics) software package and set up its environment by following the installation guide [here](https://osl.readthedocs.io/en/latest/install.html). The scripts for this paper had following dependencies:

```
python==3.12.9
osl-dynamics==2.1.5
```

Once these steps are complete, you may clone or download this repository to your preferred directory, and you're ready to begin!

## 🪪 License
Copyright (c) 2025 [SungJun Cho](https://github.com/scho97) and [OHBA Analysis Group](https://github.com/OHBA-analysis). `Cho2025_DyNeStE` is a free and open-source software licensed under the [MIT License](https://github.com/OHBA-analysis/Cho2025_DyNeStE/blob/main/LICENSE).
