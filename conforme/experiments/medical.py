# Copyright (c) 2021, Kamilė Stankevičiūtė
# Adapted from Ahmed M. Alaa github.com/ahmedmalaa/rnn-blockwise-jackknife
# Licensed under the BSD 3-clause license
# TODO change those headers

from typing import Any, Tuple

import torch

from ..conformal.predictions import Targets1D
from ..data_processing.dataset import ensure_1d_dataset_split
from ..model.rnn import RNN
from .parameters import MedicalParameters, get_specific_parameters_for_medical_dataset


def get_model_path(dataset: str, rnn_mode: str, seed: int, horizon: int):
    return "saved_models/{}-{}-{}{}.pt".format(
        dataset,
        rnn_mode,
        seed,
        ("-horizon{}".format(horizon)),
    )


def train_and_get_calibration_test_medical(
    dataset: str,
    params: MedicalParameters,
    seed: int,
    horizon: int,
    save_model: bool = False,
    retrain_model: bool = True,
) -> Tuple[Targets1D, Targets1D, Targets1D, Targets1D]:
    additional_params = get_specific_parameters_for_medical_dataset(dataset)

    additional_params.horizon_length = horizon

    split_fn = additional_params.get_split

    print("Training RNN for {}".format(dataset))

    if not retrain_model:
        forecaster_path = get_model_path(
            dataset, params.rnn_mode, seed, additional_params.horizon_length
        )
        print("Forecaster path: {}".format(forecaster_path))
    else:
        forecaster_path = None

    train_dataset, calibration_dataset, test_dataset = ensure_1d_dataset_split(
        split_fn(conformal=True, horizon=additional_params.horizon_length, seed=seed)
    )

    model = RNN(
        embedding_size=params.embedding_size,
        horizon=additional_params.horizon_length,
        rnn_mode=params.rnn_mode,
        path=forecaster_path,
    )

    model.fit(  # type: ignore
        train_dataset,
        epochs=additional_params.epochs,
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

    test_preds = Targets1D(
        model.get_point_predictions_and_errors(test_dataset)[0]
    )

    if save_model:
        torch.save(model, "saved_models/{}-{}.pt".format(dataset, seed))  # type: ignore

    return cal_preds, cal_gts, test_preds, test_gts
