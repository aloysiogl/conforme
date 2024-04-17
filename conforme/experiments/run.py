from functools import partial
from typing import Callable, Dict, Tuple, Type, TypeVar, Union

from torch.profiler import ProfilerActivity, profile

from conforme.conformal.predictions import Targets2D
from conforme.data_processing.argoverse import get_calibration_test as get_calibration_test_argoverse
from conforme.experiments.medical import train_and_get_calibration_test_medical
from conforme.experiments.parameters import MedicalParameters



from ..result.evaluation import evaluate_performance

from ..conformal.zones import DistanceZones, L1IntervalZones, Zones
from ..conformal.predictor import ConformalPredictor

from ..conformal.predictions import Targets, Targets1D

T = TypeVar("T", bound=Targets)
R = TypeVar("R")


def profile_call(callable: Callable[[], R], suffix: str, should_profile: bool) -> Tuple[Dict[str, int], R]:
    if should_profile:
        with profile(
            activities=[ProfilerActivity.CPU], profile_memory=True
        ) as prof:
            result = callable()
        total_average_prof = prof.key_averages().total_average()  # type: ignore
        profile_info = {  # type: ignore
            # type: ignore
            f"self_cpu_time_total_{suffix}": total_average_prof.self_cpu_time_total,
            # type: ignore
            f"self_cpu_memory_usage_{suffix}": total_average_prof.self_cpu_memory_usage,
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
        **profile_call(lambda: predictor.calibrate(cal_preds, cal_gts), "cal", should_profile)[0]
    }

    # Test pass
    profile_results, result = profile_call(lambda: eval_perf(
        test_preds, test_gts, predictor), "test", should_profile)
    profile_info = {
        **profile_info,
        **profile_results
    }

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


def get_medical_runner(
        make_cp: Callable[[], ConformalPredictor[Targets1D]],
        should_profile: bool,
        dataset: str,
        save_model: bool,
        retrain_model: bool,
        horizon: int,
):

    medical_params = MedicalParameters()

    def run(seed: int):
        def get_calibration_test():
            return train_and_get_calibration_test_medical(
                dataset,
                medical_params,
                seed,
                horizon,
                save_model,
                retrain_model
            )

        return run_experiment(
            get_calibration_test,
            make_cp,
            L1IntervalZones,
            should_profile,
        )
    return run
