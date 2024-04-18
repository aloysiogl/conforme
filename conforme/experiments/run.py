#
# This file is part of https://github.com/aloysiogl/conforme.
# Copyright (c) 2024 Aloysio Galvao Lopes.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

from functools import partial
from typing import Callable, Dict, Tuple, Type, TypeVar

from torch.profiler import ProfilerActivity, profile

from conforme.conformal.predictions import Targets2D
from conforme.data_processing.argoverse import (
    get_calibration_test as get_calibration_test_argoverse,
)
from conforme.experiments.medical import train_and_get_calibration_test_medical
from conforme.experiments.parameters import MedicalParameters, SyntheticParameters
from conforme.experiments.synthetic import train_and_get_calibration_test_synthetic

from ..conformal.predictions import Targets, Targets1D
from ..conformal.predictor import ConformalPredictor, ConformalPredictorParams
from ..conformal.zones import DistanceZones, L1IntervalZones, Zones
from ..result.evaluation import evaluate_performance

T = TypeVar("T", bound=Targets)
R = TypeVar("R")


def profile_call(
    callable: Callable[[], R], suffix: str, should_profile: bool
) -> Tuple[Dict[str, int], R]:
    if should_profile:
        with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
            result = callable()
        total_average_prof = prof.key_averages().total_average()  # type: ignore
        profile_info = {  # type: ignore
            f"self_cpu_time_total_{suffix}": total_average_prof.self_cpu_time_total,  # type: ignore
            f"self_cpu_memory_usage_{suffix}": total_average_prof.self_cpu_memory_usage,  # type: ignore
        }
    else:
        result = callable()
        profile_info = {}
    return profile_info, result  # type: ignore


def run_experiment(
    get_calibration_test: Callable[[], Tuple[T, T, T, T]],
    make_conformal_predictor: Callable[[], ConformalPredictor[T]],
    zones_constructor: Type[Zones[T]],
    should_profile: bool,
):
    # Get predictions and ground truths for calibration and test datasets
    cal_preds, cal_gts, test_preds, test_gts = get_calibration_test()

    predictor = make_conformal_predictor()
    eval_perf = partial(evaluate_performance, zones_constructor)
    profile_info: Dict[str, int] = {}

    # Calibration pass
    profile_info = {
        **profile_info,
        **profile_call(
            lambda: predictor.calibrate(cal_preds, cal_gts), "cal", should_profile
        )[0],
    }

    # Test pass
    profile_results, result = profile_call(
        lambda: eval_perf(test_preds, test_gts, predictor), "test", should_profile
    )
    profile_info = {**profile_info, **profile_results}

    result.set_performance_metrics(profile_info)
    return result, predictor


"""
Runners for each dataset
"""


def get_argoverse_runner(
    make_cp: Callable[[], ConformalPredictor[Targets2D]],
    should_profile: bool,
):
    def run(seed: int):
        return run_experiment(
            get_calibration_test_argoverse,
            make_cp,
            DistanceZones,
            should_profile,
        )

    return run


def get_synthetic_runner(
    make_cp: Callable[[], ConformalPredictor[Targets2D]],
    should_profile: bool,
):
    def run(seed: int):
        return run_experiment(
            get_calibration_test_argoverse,
            make_cp,
            DistanceZones,
            should_profile,
        )

    return run


def prepare_synthetic_runner(
    experiment: str,
    experiment_mode: int,
    save_model: bool,
    retrain_model: bool,
    recompute_dataset: bool,
    dynamic_sequence_lengths: bool,
    params: ConformalPredictorParams[Targets1D],
):
    def get_runner(
        make_cp: Callable[[], ConformalPredictor[Targets1D]], should_profile: bool
    ):
        synthetic_params = SyntheticParameters()

        def run(seed: int):
            def get_calibration_test():
                return train_and_get_calibration_test_synthetic(
                    experiment,
                    experiment_mode,
                    synthetic_params,
                    seed,
                    params.horizon,
                    save_model,
                    retrain_model,
                    recompute_dataset,
                    dynamic_sequence_lengths,
                )

            return run_experiment(
                get_calibration_test,
                make_cp,
                L1IntervalZones,
                should_profile,
            )

        return run

    return get_runner


def prepare_medical_runner(
    dataset: str,
    save_model: bool,
    retrain_model: bool,
    params: ConformalPredictorParams[Targets1D],
):
    horizon = params.horizon

    def get_runner(
        make_cp: Callable[[], ConformalPredictor[Targets1D]], should_profile: bool
    ):
        medical_params = MedicalParameters()

        def run(seed: int):
            def get_calibration_test():
                return train_and_get_calibration_test_medical(
                    dataset, medical_params, seed, horizon, save_model, retrain_model
                )

            return run_experiment(
                get_calibration_test,
                make_cp,
                L1IntervalZones,
                should_profile,
            )

        return run

    return get_runner
