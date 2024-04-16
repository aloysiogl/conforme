# Copyright (c) 2021, Kamilė Stankevičiūtė
# Licensed under the BSD 3-clause license

import copy
import gc
import pickle
from typing import Callable, List, Optional

import torch
from torch.profiler import ProfilerActivity, profile

from conformal_time_series.conformal.predictions import Targets1D
from conformal_time_series.conformal.score import l1_conformal_score
from conformal_time_series.model.rnn import RNN
from conformal_time_series.result.containers import Results
from ..conformal.predictor import CFRNN, ConForMEBin, ConformalPredictor

from ..data_processing.synthetic import (
    EXPERIMENT_MODES,
    get_raw_sequences,
    get_synthetic_dataset,
)
from ..model.rnn import RNN
from ..result.evaluation import evaluate_cfrnn_performance
from .parameters import SyntheticParameters

BASELINES = {"CFCRNN", "CFRNN"}


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


def run_synthetic_experiments(
    experiment: str,
    baseline: str,
    params: SyntheticParameters,
    make_conformal_predictor: Callable[[], ConformalPredictor[Targets1D]],
    retrain_auxiliary: bool = False,
    recompute_dataset: bool = False,
    dynamic_sequence_lengths: bool = False,
    should_profile: bool = False,
    n_threads: Optional[int] = None,
    n_train: Optional[int] = None,
    horizon: Optional[int] = None,
    save_model: bool = False,
    save_results: bool = True,
    seed: int = 0,
):
    """
    Runs an experiment for a synthetic dataset.

    Args:
        experiment: type of experiment ('time_dependent', 'static', 'periodic', 'sample_complexity')
        baseline: the model to be trained ('BJRNN', 'DPRNN', 'QRNN', 'CFRNN', 'AdaptiveCFRNN')
        retrain_auxiliary: whether to retrain the AuxiliaryForecaster of the CFRNN models
        recompute_dataset: whether to generate the dataset from scratch
        params: dictionary of training parameters
        dynamic_sequence_lengths: whether to use datasets where sequences have different randomly sampled lengths
        n_train: number of training examples
        horizon: forecasting horizon
        beta: (in AdaptiveCFRNN) hyperparameter to dampen the importance of the correction factor
        correct_conformal: whether to use Bonferroni-corrected calibration scores
        save_model: whether to save the model in the `./saved_models/` directory
        save_results: whether to save the results in `./saved_results/`
        rnn_mode: (in CFRNN) the type of RNN of the underlying forecaster (RNN/LSTM/GRU)
        seed: random seed

    Returns:
        a dictionary of result metrics
    """

    assert baseline in BASELINES, "Invalid baseline"
    assert experiment in EXPERIMENT_MODES.keys(), "Invalid experiment"

    results: List[Results] = []
    predictors: List[ConformalPredictor[Targets1D]] = []

    torch.manual_seed(seed)  # type: ignore

    params = copy.deepcopy(params)
    
    if horizon is not None:
        params.horizon = horizon

    raw_sequence_datasets = get_raw_sequences(
        experiment=experiment,
        n_train=n_train,
        dynamic_sequence_lengths=dynamic_sequence_lengths,
        horizon=horizon,
        seed=seed,
        recompute_dataset=recompute_dataset,
    )

    print("Training {}".format(baseline))

    for i, raw_sequence_dataset in enumerate(raw_sequence_datasets):
        print("Training dataset {}".format(i))

        if not retrain_auxiliary:
            forecaster_path = get_model_path(
                experiment,
                params.rnn_mode,
                EXPERIMENT_MODES[experiment][i],
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
                result = evaluate_cfrnn_performance(point_predictions, test_dataset, predictor)
            
            total_average_prof = prof.key_averages().total_average()
            profile_info = {
                **profile_info,
                "self_cpu_time_total_test": total_average_prof.self_cpu_time_total,
                "self_cpu_memory_usage_test": total_average_prof.self_cpu_memory_usage,
            }
        else:
            result = evaluate_cfrnn_performance(point_predictions, test_dataset, predictor)

        if not should_profile:
            profile_info = None

        result.set_performance_metrics(profile_info)
        result.n_threads = n_threads

        results.append(result)
        predictors.append(predictor)

        if save_model:
            torch.save(
                model,
                get_model_path(
                    experiment,
                    model.rnn_mode,
                    EXPERIMENT_MODES[experiment][i],
                    seed,
                    dynamic_sequence_lengths,
                    horizon,
                ),
            )

        del model
        gc.collect()

    if save_results:
        with open(
            get_results_path(
                experiment, baseline, seed, dynamic_sequence_lengths, horizon
            ),
            "wb",
        ) as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    return results, predictors
