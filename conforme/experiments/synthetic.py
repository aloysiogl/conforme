# Copyright (c) 2021, Kamilė Stankevičiūtė
# Licensed under the BSD 3-clause license

import copy
import gc
from typing import Any, Optional

import torch

from ..conformal.predictions import Targets1D
from ..data_processing.synthetic import (
    EXPERIMENT_MODES,
    get_raw_sequences,
    get_synthetic_dataset,
)
from ..model.rnn import RNN
from .parameters import SyntheticParameters


def get_model_path(
    experiment: str,
    rnn_mode: str,
    mode: int,
    seed: int,
    dynamic_sequence_lengths: bool,
    horizon: Optional[int],
):
    return "saved_models/{}-{}-{}-{}-{}{}{}.pt".format(
        experiment,
        "rnn",
        rnn_mode,
        mode,
        seed,
        ("-dynamic" if dynamic_sequence_lengths else ""),
        ("-horizon{}".format(horizon)),
    )


def get_results_path(
    experiment: str,
    baseline: str,
    seed: int,
    dynamic_sequence_lengths: bool,
    horizon: int,
):
    return "saved_results/{}-{}-{}{}{}.pkl".format(
        experiment,
        baseline,
        seed,
        ("-dynamic" if dynamic_sequence_lengths else ""),
        ("-horizon{}".format(horizon)),
    )


def train_and_get_calibration_test_synthetic(
    experiment: str,
    experiment_mode: int,
    params: SyntheticParameters,
    seed: int,
    horizon: int,
    save_model: bool,
    retrain_model: bool,
    recompute_dataset: bool = False,
    dynamic_sequence_lengths: bool = False,
):
    """
    Runs an experiment for a synthetic dataset.

    Args:
        experiment: type of experiment ('time_dependent', 'static', 'periodic', 'sample_complexity')
        experiment_mode: setting controlling independet variables
        params: training parameters
        seed: random seed
        horizon: forecasting horizon
        save_model: whether to save the model in the `./saved_models/` directory
        retrain_model: whether to retrain the predictor model
        recompute_dataset: whether to generate the dataset from scratch
        dynamic_sequence_lengths: whether to use datasets where sequences have different randomly sampled lengths

    Returns:
        calibration and test predictions and ground truths
    """

    assert experiment in EXPERIMENT_MODES.keys(), "Invalid experiment"

    mode_idx = EXPERIMENT_MODES[experiment].index(experiment_mode)

    torch.manual_seed(seed)  # type: ignore

    params = copy.deepcopy(params)
    params.horizon = horizon

    raw_sequence_datasets = get_raw_sequences(
        experiment=experiment,
        n_train=params.n_train,
        dynamic_sequence_lengths=dynamic_sequence_lengths,
        horizon=horizon,
        seed=seed,
        recompute_dataset=recompute_dataset,
    )

    print("Training synthetic {}".format(experiment))

    raw_sequence_dataset = raw_sequence_datasets[mode_idx]

    print("Training dataset {}".format(mode_idx))

    if not retrain_model:
        forecaster_path = get_model_path(
            experiment,
            params.rnn_mode,
            EXPERIMENT_MODES[experiment][mode_idx],
            seed,
            dynamic_sequence_lengths,
            horizon,
        )
        print("Forecaster path: {}".format(forecaster_path))
    else:
        forecaster_path = None

    params.output_size = horizon if horizon else params.horizon

    train_dataset, calibration_dataset, test_dataset = get_synthetic_dataset(
        raw_sequence_dataset, seed=seed
    )
    model = RNN(
        embedding_size=params.embedding_size,
        horizon=params.horizon,
        rnn_mode=params.rnn_mode,
        path=forecaster_path,
    )

    model.fit(
        train_dataset,
        epochs=params.epochs,
        lr=params.lr,
        batch_size=params.batch_size,
    )

    calibration_data_loader: Any = torch.utils.data.DataLoader(  # type: ignore
        calibration_dataset, batch_size=32
    )
    cal_gts = Targets1D(torch.cat([batch[1] for batch in calibration_data_loader]))
    cal_preds = Targets1D(
        model.get_point_predictions_and_errors(calibration_dataset)[0]
    )

    test_dataloader: Any = torch.utils.data.DataLoader(test_dataset, batch_size=32)  # type: ignore
    test_gts = Targets1D(torch.cat([batch[1] for batch in test_dataloader]))

    test_preds = Targets1D(model.get_point_predictions_and_errors(test_dataset)[0])

    if save_model:
        torch.save(  # type: ignore
            model,
            get_model_path(
                experiment,
                model.rnn_mode,
                EXPERIMENT_MODES[experiment][mode_idx],
                seed,
                dynamic_sequence_lengths,
                horizon,
            ),
        )

    del model
    gc.collect()

    return cal_preds, cal_gts, test_preds, test_gts
