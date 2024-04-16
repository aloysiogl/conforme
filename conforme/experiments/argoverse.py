from typing import Callable, Optional

import torch
from torch.profiler import ProfilerActivity, profile

from conforme.conformal.zones import DistanceZones

from ..result.evaluation import evaluate_performance

from ..conformal.predictions import Targets2D
from ..conformal.predictor import ConformalPredictor
from ..data_processing.argoverse import get_calibration_test

def eval_perf(test_preds, test_gts, predictor):
    return evaluate_performance(test_preds, test_gts, predictor, DistanceZones)

def run_argoverse_experiments(
    make_conformal_predictor: Callable[[], ConformalPredictor[Targets2D]],
    should_profile: bool = False,
    n_threads: Optional[int] = None,
    seed: int = 0,
    horizon: Optional[int] = None,
):
    torch.manual_seed(seed)
    if horizon != 30:
        raise ValueError("Argoverse only supports horizon=30")
    cal_preds, cal_gts, test_preds, test_gts = get_calibration_test()

    predictor = make_conformal_predictor()
    if should_profile:
        with profile(
            activities=[ProfilerActivity.CPU], profile_memory=True
        ) as prof:
            predictor.calibrate(cal_preds, cal_gts)

        total_average_prof = prof.key_averages().total_average()
        profile_info = {
            "self_cpu_time_total_cal": total_average_prof.self_cpu_time_total,
            "self_cpu_memory_usage_cal": total_average_prof.self_cpu_memory_usage,
        }
    else:
        predictor.calibrate(cal_preds, cal_gts)

    if should_profile:
        with profile(
            activities=[ProfilerActivity.CPU], profile_memory=True
        ) as prof:
            results = eval_perf(test_preds, test_gts, predictor)
        total_average_prof = prof.key_averages().total_average()
        profile_info = {
            **profile_info,
            "self_cpu_time_total_test": total_average_prof.self_cpu_time_total,
            "self_cpu_memory_usage_test": total_average_prof.self_cpu_memory_usage,
        }
    else:
        results = eval_perf(test_preds, test_gts, predictor)
    if not should_profile:
        profile_info = None

    results.set_performance_metrics(profile_info)
    results.n_threads = n_threads
    
    return results, predictor