# Repository for the paper "ConForME: Multi-horizon conformal time series forecasting"

Paper: [ConForME: Multi-horizon conformal time series forecasting](doc/conforme-aloysio_eric_laurent_sylvie.pdf)

Table of contents:

- [Abstract](doc/abstract.md)
- [Installation](doc/install.md)
- [Download data](doc/download_data.md)
- [Usage](doc/usage.md)
- [Code structure](doc/structure.md)
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

## How to cite

```bibtex
@inproceedings{conforme2024,
  author       = {Aloysio Galvao Lopes and
                  Eric Goubault and
                  Laurent Pautet and
                  Sylvie Putot},
  editor       = {Simone Vantini and
                  Matteo Fontana and
                  Aldo Solari and
                  Henrik Bostr{\"{o}}m and
                  Lars Carlsson},
  title        = {ConForME: Multi-horizon conformal time series forecasting},
  booktitle    = {Conformal and Probabilistic Prediction with Applications, 9-11 September
                  2024, Milan, Italy},
  series       = {Proceedings of Machine Learning Research},
  volume       = {230},
  publisher    = {{PMLR}},
  year         = {2024},
}
```
