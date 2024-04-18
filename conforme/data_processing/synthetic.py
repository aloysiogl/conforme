# Adapted from Kamilė Stankevičiūtė https://github.com/kamilest/conformal-rnn
# Licensed under the BSD 3-clause license

import os.path
import sys
from typing import Optional, List, Tuple

if sys.version_info < (3, 8, 3):
    import pickle5 as pickle
else:
    import pickle

import numpy as np
import numpy.typing as npt
import torch

from .types import (
    NumpyFloatArray,
    StaticCustomParameters,
    TrainSplit,
    TestSplit,
    TrainTestSplit,
)


# Settings controlling the independent variables of experiments depending on
# the experiment mode:
#   periodic: Controls periodicity.
#   time_dependent: Controls increasing noise amplitude within a single
#     time-series.
#   static: Controls noise amplitudes across the collection of time-series.
# See paper for details.

EXPERIMENT_MODES = {
    "periodic": [2, 10],
    "time_dependent": range(1, 6),
    "static": range(1, 6),
    "sample_complexity": [10, 100, 1000, 10000],
}

DEFAULT_PARAMETERS = {
    "length": 15,
    "horizon": 5,
    "mean": 1,
    "variance": 2,
    "memory_factor": 0.9,
    "amplitude": 5,
    "harmonics": 1,
    "periodicity": None,
}


def autoregressive(X_gen: NumpyFloatArray, w: npt.NDArray[np.float64]):
    """Generates the autoregressive component of a single time series
    example."""
    return np.array(
        [
            np.sum(X_gen[0 : k + 1] * np.flip(w[0 : k + 1]).reshape(-1, 1))
            for k in range(len(X_gen))
        ]
    )


# https://www.statsmodels.org/devel/examples/notebooks/generated/statespace_seasonal.html
def seasonal(
    duration: int,
    periodicity: int,
    amplitude: float = 1.0,
    harmonics: int = 1,
    random_state: Optional[np.random.RandomState] = None,
    asynchronous: bool = True,
):
    if random_state is None:
        random_state = np.random.RandomState(0)

    harmonics = harmonics if harmonics else int(np.floor(periodicity / 2))

    lambda_p = 2 * np.pi / float(periodicity)

    gamma_jt = amplitude * random_state.randn(harmonics)
    gamma_star_jt = amplitude * random_state.randn(harmonics)

    # Pad for burn in
    if asynchronous:
        # Will make series start at random phase when burn-in is discarded
        total_timesteps = duration + random_state.randint(duration)  # type: ignore
    else:
        total_timesteps = 2 * duration
    series = np.zeros(total_timesteps)
    for t in range(total_timesteps):
        gamma_jtp1 = np.zeros_like(gamma_jt)
        gamma_star_jtp1 = np.zeros_like(gamma_star_jt)
        for j in range(1, harmonics + 1):
            cos_j = np.cos(lambda_p * j)
            sin_j = np.sin(lambda_p * j)
            gamma_jtp1[j - 1] = (
                gamma_jt[j - 1] * cos_j
                + gamma_star_jt[j - 1] * sin_j
                + amplitude * random_state.randn()
            )
            gamma_star_jtp1[j - 1] = (
                -gamma_jt[j - 1] * sin_j
                + gamma_star_jt[j - 1] * cos_j
                + amplitude * random_state.randn()
            )
        series[t] = np.sum(gamma_jtp1)
        gamma_jt = gamma_jtp1
        gamma_star_jt = gamma_star_jtp1

    return series[-duration:].reshape(-1, 1)  # Discard burn in


class AutoregressiveForecastDataset(torch.utils.data.Dataset):  # type: ignore
    """Synthetic autoregressive forecast dataset."""

    def __init__(
        self, X: torch.Tensor, Y: torch.Tensor, sequence_lengths: npt.NDArray[np.int64]
    ):
        super(AutoregressiveForecastDataset, self).__init__()  # type: ignore
        self.X = X
        self.Y = Y
        self.sequence_lengths = sequence_lengths

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx], self.sequence_lengths[idx]


AutoregressiveForecastDatasetSplits = Tuple[
    AutoregressiveForecastDataset,
    AutoregressiveForecastDataset,
    AutoregressiveForecastDataset,
]


def split_train_sequence(X_full: List[torch.Tensor], horizon: int):
    # Splitting time series into training sequence X and target sequence Y;
    # Y stores the time series predicted targets `horizon` steps away
    X: List[torch.Tensor] = []
    Y: List[torch.Tensor] = []
    for seq in X_full:
        seq_len = len(seq)
        if seq_len >= 2 * horizon:
            X.append(seq[:-horizon])
            Y.append(seq[-horizon:])
        elif seq_len > horizon:
            X.append(seq[: seq_len - horizon])
            Y.append(seq[(seq_len - horizon) :])

    return X, Y


def generate_autoregressive_forecast_dataset(
    n_samples: int,
    experiment: str,
    setting: int,
    n_features: int = 1,
    dynamic_sequence_lengths: bool = False,
    horizon: Optional[int] = None,
    custom_parameters: StaticCustomParameters = None,
    random_state: Optional[np.random.RandomState] = None,
):
    assert experiment in EXPERIMENT_MODES.keys()

    if random_state is None:
        random_state = np.random.RandomState(0)

    params = DEFAULT_PARAMETERS.copy()
    if custom_parameters is not None:
        for key in custom_parameters.keys():
            params[key] = custom_parameters[key]

    if params["length"] is None:
        raise ValueError("Length of time series must be specified.")

    if horizon is not None:
        params["horizon"] = horizon
        dynamic_sequence_lengths = False

    if type(params["horizon"]) is not int:
        raise ValueError("Horizon must be an integer.")

    if experiment == "sample_complexity":
        experiment = "time_dependent"
        n_samples = setting
        setting = 1

    # Setting static or dynamic sequence lengths
    if dynamic_sequence_lengths:
        sequence_lengths: npt.NDArray[np.float64] = (  # type: ignore
            params["horizon"]
            + params["length"] // 2
            + random_state.geometric(p=2 / params["length"], size=n_samples)  # type: ignore
        )
    else:
        sequence_lengths: npt.NDArray[np.float64] = np.array([params["length"] + params["horizon"]] * n_samples)  # type: ignore
    sequence_lengths: npt.NDArray[np.int64]

    # Noise profile-dependent settings
    if experiment == "static":
        noise_vars = [[0.1 * setting] * sl for sl in sequence_lengths]

    elif experiment == "time_dependent":
        noise_vars = [[0.1 * setting * k for k in range(sl)] for sl in sequence_lengths]
    else:
        # No additional noise beyond the variance of X_gen
        noise_vars = [[0] * sl for sl in sequence_lengths]

    if experiment == "periodic":
        params["periodicity"] = setting

    # Create the input features of the generating process
    X_gen: List[npt.NDArray[np.float64]] = [
        random_state.normal(params["mean"], params["variance"], (sl, n_features))  # type: ignore
        for sl in sequence_lengths
    ]

    if params["memory_factor"] is None:
        raise ValueError("Memory factor must be specified.")

    w: npt.NDArray[np.float64] = np.array(
        [params["memory_factor"] ** k for k in range(np.max(sequence_lengths))]
    )

    # X_full stores the time series values generated from features X_gen.
    ar = [autoregressive(x, w).reshape(-1, n_features) for x in X_gen]
    noise = [random_state.normal(0.0, nv).reshape(-1, n_features) for nv in noise_vars]

    if params["periodicity"] is not None:
        if type(params["periodicity"]) is not int:
            raise ValueError("Periodicity must be an integer.")
        if type(params["amplitude"]) is not int:
            raise ValueError("Amplitude must be an integer.")
        if type(params["harmonics"]) is not int:
            raise ValueError("Harmonics must be an integer.")
        periodic = [
            seasonal(
                sl,
                params["periodicity"],
                params["amplitude"],
                params["harmonics"],
                random_state=random_state,
                asynchronous=dynamic_sequence_lengths,
            )
            for sl in sequence_lengths
        ]
    else:
        periodic = np.array([np.zeros(sl) for sl in sequence_lengths])

    X_full = [
        torch.tensor(i + j + k.reshape(-1, 1)) for i, j, k in zip(ar, noise, periodic)
    ]

    # Splitting time series into training sequence X and target sequence Y;
    # Y stores the time series predicted targets `horizon` steps away
    X, Y = split_train_sequence(X_full, params["horizon"])
    train_sequence_lengths: npt.NDArray[np.int64] = sequence_lengths - params["horizon"]

    return X, Y, train_sequence_lengths


def get_raw_sequences(
    experiment: str,
    n_train: Optional[int] = None,
    n_test: Optional[int] = None,
    dynamic_sequence_lengths: bool = False,
    horizon: Optional[int] = None,
    recompute_dataset: bool = False,
    seed: int = 0,
) -> List[TrainTestSplit]:
    assert experiment in EXPERIMENT_MODES.keys()

    if n_train is None:
        n_train = 2000
    if n_test is None:
        n_test = 500

    raw_sequences: List[TrainTestSplit] = []
    random_state = np.random.RandomState(seed)

    for setting in EXPERIMENT_MODES[experiment]:
        if experiment == "sample_complexity":
            n_train = setting
        dataset_file = "processed_data/synthetic-{}-{}-{}-{}{}{}.pkl".format(
            experiment,
            setting,
            seed,
            n_train,
            ("-dynamic" if dynamic_sequence_lengths else ""),
            (
                "-horizon{}".format(horizon)
                if horizon is not None and horizon != DEFAULT_PARAMETERS["horizon"]
                else ""
            ),
        )

        if os.path.isfile(dataset_file) and not recompute_dataset:
            with open(dataset_file, "rb") as f:
                raw_train_sequences, raw_test_sequences = pickle.load(f)
            raw_sequences.append((raw_train_sequences, raw_test_sequences))
        else:
            X_train, Y_train, sequence_lengths_train = (
                generate_autoregressive_forecast_dataset(
                    n_samples=n_train,
                    experiment=experiment,
                    setting=setting,
                    dynamic_sequence_lengths=dynamic_sequence_lengths,
                    horizon=horizon,
                    random_state=random_state,
                )
            )

            X_train: List[torch.Tensor]
            Y_train: List[torch.Tensor]
            sequence_lengths_train: npt.NDArray[np.int64]

            X_test, Y_test, sequence_lengths_test = (
                generate_autoregressive_forecast_dataset(
                    n_samples=n_test,
                    experiment=experiment,
                    setting=setting,
                    dynamic_sequence_lengths=dynamic_sequence_lengths,
                    horizon=horizon,
                    random_state=random_state,
                )
            )

            X_test: List[torch.Tensor]
            Y_test: List[torch.Tensor]
            sequence_lengths_test: npt.NDArray[np.int64]

            with open(dataset_file, "wb") as f:
                pickle.dump(
                    (
                        (X_train, Y_train, sequence_lengths_train),
                        (X_test, Y_test, sequence_lengths_test),
                    ),
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

            raw_sequences.append(
                (
                    (X_train, Y_train, sequence_lengths_train),
                    (X_test, Y_test, sequence_lengths_test),
                )
            )

    return raw_sequences


def get_synthetic_dataset(
    raw_sequences: TrainTestSplit,
    p_calibration: float = 0.5,
    seed: int = 0,
) -> AutoregressiveForecastDatasetSplits:
    (X_train, Y_train, sequence_lengths_train), (
        X_test,
        Y_test,
        sequence_lengths_test,
    ) = raw_sequences

    (
        (X_train, Y_train, sequence_lengths_train),
        (X_calibration, Y_calibration, sequence_lengths_calibration),
    ) = split_train_dataset(
        X_train, Y_train, sequence_lengths_train, p_calibration, seed=seed
    )

    # X: [n_samples, max_seq_len, n_features]
    X_train_tensor = torch.nn.utils.rnn.pad_sequence(
        X_train, batch_first=True
    ).float()

    # Y: [n_samples, horizon, n_features]
    Y_train_tensor = torch.nn.utils.rnn.pad_sequence(
        Y_train, batch_first=True
    ).float()

    train_dataset = AutoregressiveForecastDataset(
        X_train_tensor, Y_train_tensor, sequence_lengths_train
    )

    X_calibration_tensor = torch.nn.utils.rnn.pad_sequence(
        X_calibration, batch_first=True
    ).float()

    # Y: [n_samples, horizon, n_features]
    Y_calibration_tensor = torch.nn.utils.rnn.pad_sequence(
        Y_calibration, batch_first=True
    ).float()

    calibration_dataset = AutoregressiveForecastDataset(
        X_calibration_tensor, Y_calibration_tensor, sequence_lengths_calibration
    )

    # X: [n_samples, max_seq_len, n_features]
    X_test_tensor = torch.nn.utils.rnn.pad_sequence(
        X_test, batch_first=True
    ).float()

    # Y: [n_samples, horizon, n_features]
    Y_test_tensor = torch.nn.utils.rnn.pad_sequence(
        Y_test, batch_first=True
    ).float()

    test_dataset = AutoregressiveForecastDataset(
        X_test_tensor, Y_test_tensor, sequence_lengths_test
    )

    synthetic_dataset = (train_dataset, calibration_dataset, test_dataset)

    return synthetic_dataset


def split_train_dataset(
    X_train: List[torch.Tensor],
    Y_train: List[torch.Tensor],
    sequence_lengths_train: np.ndarray[int, np.dtype[np.int64,]],
    n_calibration: float,
    seed: int = 0,
) -> Tuple[TrainSplit, TestSplit]:
    """Splits the train dataset into training and calibration sets."""
    n_train = len(X_train)
    idx_perm = np.random.RandomState(seed).permutation(n_train)
    idx_calibration: np.ndarray[int, np.dtype[np.int64,]] = idx_perm[
        : int(n_train * n_calibration)
    ]
    idx_train = idx_perm[int(n_train * n_calibration) :]

    X_calibration: List[torch.Tensor] = [X_train[i] for i in idx_calibration]
    Y_calibration: List[torch.Tensor] = [Y_train[i] for i in idx_calibration]
    sequence_lengths_calibration = np.array(
        [sequence_lengths_train[i] for i in idx_calibration]
    )

    X_train = [X_train[i] for i in idx_train]
    Y_train = [Y_train[i] for i in idx_train]
    sequence_lengths_train = np.array([sequence_lengths_train[i] for i in idx_train])

    return (X_train, Y_train, sequence_lengths_train), (
        X_calibration,
        Y_calibration,
        sequence_lengths_calibration,
    )
