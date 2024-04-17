from functools import partial
from typing import Any, Callable, Dict, List, Tuple, Type, TypeVar, Union

import numpy as np
from torch.profiler import ProfilerActivity, profile

from conforme.data_processing.argoverse import get_calibration_test as get_calibration_test_argoverse

from ..result.results_database import ResultsDatabase

from ..result.containers import Results, ResultsWrapper

from ..result.evaluation import evaluate_performance

from ..conformal.zones import DistanceZones, Zones
from ..conformal.predictor import ConformalPredictor

from ..conformal.predictions import Targets

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


T_1 = TypeVar("T_1", bound=Targets)


def evaluate_conformal_method(
    run_experiment: Callable[[], Tuple[Results, ConformalPredictor[T_1]]],
    results_database: ResultsDatabase,
    params: Dict[str, Union[str, Any]],
    seeds: List[int] = [0, 1, 2, 3, 4],
    skip_existing: bool = False,
    save_results: bool = True,
):
    if skip_existing and results_database.lookup(params):
        return

    results_wrapper = ResultsWrapper()
    tunnable_params = None
    for seed in seeds:
        np.random.seed(seed)
        results, predictor = run_experiment()
        results_wrapper.add_results([results])
        tunnable_params = predictor.get_tunnable_params()

    result = {
        "outputs": results_wrapper.get_dict(),
        "tunnable_params": tunnable_params,
    }
    results_database.modify_result(params, result)

    if save_results:
        results_database.save()
