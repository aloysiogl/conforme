# Repository for the paper "ConForME: Multi-horizon conditional conformal time series forecasting"

Link to the paper: TODO

Table of contents:

- [Abstract](doc/abstract.md)
- [Installation](doc/install.md)
- [Download data](doc/download_data.md)
- [Usage](doc/usage.md)
- [License](LICENSE.txt)

## TLDR

You'll need `poetry` and `git-lfs` to run and download the data. Then, run the following commands from the root of the repo to install, pull the data, and run the experiments:

```bash
poetry install
git lfs install && git lfs pull
poetry run python scripts/run_experiments_tables.py
poetry run python scripts/run_experiments_betas.py
```

JSON files with the results will be saved in the `results` directory. Further instructions detailing how to make these results more readable can be found in [here](doc/usage.md).

## Citation

TODO
