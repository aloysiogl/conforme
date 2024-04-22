# Usage

First, make sure you followed the dataset [download instructions](download_data.md), otherwise you might get strange errors.

You'll first need to run the experiments with:

```bash
poetry run python scripts/run_experiments_tables.py
poetry run python scripts/run_experiments_betas.py
```

The results fore each experiment is stored in a JSON file containing everything that was run. They are stored under the `results` directory. The naming follows this convention: `<DATASET>[_profile]_horizon<HORIZON>`. Here `<DATASET>` is the dataset name, `_profile` is an optional string that indicates that execution times and memory usage are measured, and `<HORIZON>` is the horizon of the forecast. For example, the results for the EEG dataset with a horizon of 40 are stored in the file `results/eeg_profile_horizon40.json`.

Then, to make the results more readable, you can run the `summary_table.py` script. There are two options for this script `--results-path <PATH>` or `-r <PATH>` and `--output-type <TYPE>` or `-o <TYPE>`. Type can be either `areas` or `times`, but it is set by default to `areas`. Here is an example:

```bash
poetry run python scripts/summary_table.py -r results/eeg_profile_horizon10.json -o times
```

The tables in the paper were used by applying the script above. They are all generated in the standard output.

To generate different experiments, one could get inspiration from the `run_experiments_tables.py` and `run_experiments_betas.py`.

To generate the figures one script is dedicated to interval sizes per horizon and another to the plot in figure 1 (a) that shows mean interval size per beta. Run these scripts after the experiments and from the root. You'll find below the commands to plot all figures in the paper:  

```bash
poetry run python scripts/plot_prediction_horizon.py -s eeg_all
poetry run python scripts/plot_prediction_horizon.py -s eeg_bin
poetry run python scripts/plot_prediction_horizon.py -s argoverse_all
poetry run python scripts/plot_betas.py
```

The script `plot_prediction_horizon.py` has the option `-s` or `--setting` to specify which figure to generate. `eeg_all`, `eeg_bin`, and `argoverse_all` correspond, respectively, to the figures 2(a), 1(b) and 2(b) in the paper. The script places the generated figures in the results folder. Finally `plot_betas.py` generates the figure 1(a) in the paper and has no arguments.
