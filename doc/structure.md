# Code structure

The package is structured as follows:

- `conformal` contains the main classes to implement ConForME. It also provides a some generic structures to implement any multi-horizon conformal predictor. You can also define new conformal scores and prediction zones based on those scores.
- `data_processing` contains the scrips to transform the data to a usable format.
- `experiments` contains the base code to run the experiments as generically as possible and to profile memory and runtime.
- `model` contains rnn which is used throughout most experiments.
- `results` contains the code for wrapping and synthesizing the results as well as to evaluate the performance.
