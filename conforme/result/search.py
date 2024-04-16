from typing import TypeVar
from conforme.result.evaluation import evaluate_performance
from ..result.containers import ResultsWrapper
from ..conformal.predictor import ConForMEBin
from ..conformal.predictions import Targets

T = TypeVar("T", bound=Targets)


def binary_search_beta(
    predictor: ConForMEBin[T], cal_preds: T, cal_gts: T
):
    l_beta = 0
    r_beta = 1
    eps = 0.01
    best_beta = 0.0

    def compute_mean_interval_withds_for_beta(beta):
        predictor.set_beta(beta)
        predictor.calibrate(cal_preds, cal_gts)
        result = evaluate_performance(cal_preds, cal_gts, predictor)
        wrapped_results = ResultsWrapper()
        wrapped_results.add_results([result])
        return wrapped_results.get_mean_area()[0].item()

    while r_beta - l_beta > eps:
        beta = (l_beta + r_beta) / 2
        if compute_mean_interval_withds_for_beta(beta) < compute_mean_interval_withds_for_beta(l_beta): 
            l_beta = beta
        else:
            r_beta = beta
        best_beta = beta

    return best_beta
