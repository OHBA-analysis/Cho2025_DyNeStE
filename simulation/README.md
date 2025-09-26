# Simulation

### 🔎 Script Descriptions

This directory contains scripts for simulating data, training models, and analysing results.

| Scripts                      | Description                                                             | Figures       |
| :--------------------------- | :---------------------------------------------------------------------- | :------------ |
| `01_train_dyneste.py`        | Simulates data, trains DyNeStE, and saves the inferred parameters.      | -      |
| `02_train_hmm.py`            | Simulates data, trains TDE-HMM, and saves the inferred parameters.      | -      |
| `03_analyze_simulations.py`  | Analyses inferred models and visualises the results.                    | 2A-C   |
| `04_compare_simulations.py`  | Compares DyNeStE and TDE-HMM across multiple simulation runs.           | 2D     |

**NOTE:** Data were simulated using Hidden Semi-Markov Models. DyNeStE and the HMM used the same random seed for each simulation run.

### 🙋‍♂️ FAQ: What about the `utils` subdirectory?
The `utils` subdirectory contains essential functions required to run the scripts summarised above. Each script in `utils` includes multiple 
functions. These functions are self-explanatory and include detailed annotations, so their descriptions are not repeated here.
