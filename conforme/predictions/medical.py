# Copyright (c) 2021, Kamilė Stankevičiūtė
# Adapted from Ahmed M. Alaa github.com/ahmedmalaa/rnn-blockwise-jackknife
# Licensed under the BSD 3-clause license
# TODO change those headers

import pickle
from typing import Callable, Optional

import torch
from torch.profiler import ProfilerActivity, profile

from ..conformal.predictor import ConformalPredictor
from ..conformal.zones import L1IntervalZones

from ..conformal.predictions import Targets1D
from ..data_processing.dataset import ensure_1d_dataset_split
from ..model.rnn import RNN
from ..result.evaluation import evaluate_cfrnn_performance, evaluate_performance
from .parameters import MedicalParameters, get_specific_parameters_for_medical_dataset

BASELINES = {"CFRNN", "CFCRNN"}


def get_model_path(
    dataset: str, rnn_mode: str, seed: int, horizon: int, baseline: Optional[str] = None
):
    # TODO put the name of the forecaster RNN correctly
    return "saved_models/{}-{}-{}-{}{}.pt".format(
        dataset,
        "rnn",
        rnn_mode,
        seed,
        ("-horizon{}".format(horizon)),
    )


def run_medical_experiments(
    dataset: str,
    baseline: str,
    params: MedicalParameters,
    make_conformal_predictor: Callable[[], ConformalPredictor[Targets1D]],
    save_model: bool = False,
    retrain_model: bool = True,
    should_profile: bool = False,
    n_threads: Optional[int] = None,
    seed: int = 0,
    horizon: Optional[int] = None,
):
    assert baseline in BASELINES, "Invalid baselines"
    additional_params = get_specific_parameters_for_medical_dataset(dataset, baseline)

    if horizon is not None:
        additional_params.horizon_length = horizon

    split_fn = additional_params.get_split

    if n_threads is not None:
        torch.set_num_threads(n_threads)
        torch.set_num_interop_threads(n_threads)

    print("Training {}".format(baseline))

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

    calibration_data_loader = torch.utils.data.DataLoader(
        calibration_dataset, batch_size=32
    )
    calibration_data = Targets1D(
        torch.cat([batch[1] for batch in calibration_data_loader])
    )
    calibration_predictions = Targets1D(
        model.get_point_predictions_and_errors(calibration_dataset)[0]
    )

    predictor = make_conformal_predictor()

    if should_profile:
        with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
            predictor.calibrate(calibration_predictions, calibration_data)

        total_average_prof = prof.key_averages().total_average()
        profile_info = {
            "self_cpu_time_total_cal": total_average_prof.self_cpu_time_total,
            "self_cpu_memory_usage_cal": total_average_prof.self_cpu_memory_usage,
        }
    else:
        predictor.calibrate(calibration_predictions, calibration_data)

    point_predictions = Targets1D(
        model.get_point_predictions_and_errors(test_dataset, corrected=True)[0]
    )

    if should_profile:
        with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
            targets = Targets1D(torch.cat([batch[1] for batch in test_dataloader]))
            results = evaluate_performance(point_predictions, targets, predictor, L1IntervalZones)

        total_average_prof = prof.key_averages().total_average()
        profile_info = {
            **profile_info,
            "self_cpu_time_total_test": total_average_prof.self_cpu_time_total,
            "self_cpu_memory_usage_test": total_average_prof.self_cpu_memory_usage,
        }
    else:
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
        targets = Targets1D(torch.cat([batch[1] for batch in test_dataloader]))
        results = evaluate_performance(point_predictions, targets, predictor, L1IntervalZones)

    if save_model:
        torch.save(model, "saved_models/{}-{}-{}.pt".format(dataset, baseline, seed))  # type: ignore
    if not should_profile:
        profile_info = None

    results.set_performance_metrics(profile_info)
    results.n_threads = n_threads

    return results, predictor


def load_medical_results(dataset: str, baseline: str, seed: int):
    with open("saved_results/{}-{}-{}.pkl".format(dataset, baseline, seed), "rb") as f:
        results = pickle.load(f)
    return results
