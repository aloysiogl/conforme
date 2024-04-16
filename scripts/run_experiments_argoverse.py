from typing import Callable, List

import numpy as np
from tqdm import tqdm

from conforme.conformal.predictions import Targets2D
from conforme.conformal.predictor import (
    CFRNN,
    ConformalPredictor,
    ConForME,
    ConForMEBin,
)
from conforme.conformal.score import distance_2d_conformal_score
from conforme.data_processing.argoverse import get_calibration_test
from conforme.experiments.argoverse import run_argoverse_experiments
from conforme.result.containers import ResultsWrapper
from conforme.result.evaluation import evaluate_performance
from conforme.result.results_database import ResultsDatabase


def evaluate_conformal_method(
    horizon: int,
    make_conformal_predictor: Callable[[float, int], ConformalPredictor[Targets2D]],
    seeds: List[int] = [0, 1, 2, 3, 4],
    alpha: float = 0.1,
    should_profile: bool = False,
    n_threads: int = 1,
    skip_existing: bool = False,
    save_results: bool = True,
):
    def make_pred():
        return make_conformal_predictor(alpha, horizon)

    database_params = {
        "dataset": dataset,
        "nb_seeds": len(seeds),
        "horizon": horizon,
        "alpha": alpha,
        "conformal_predictor_params": make_pred().get_params(),
        "method": make_pred().get_name(),
    }

    if skip_existing and results_database.lookup(database_params):
        return

    results_wrapper = ResultsWrapper()
    tunnable_params = None
    for seed in seeds:
        np.random.seed(seed)
        results, predictor = run_argoverse_experiments(
            make_conformal_predictor=make_pred,
            should_profile=should_profile,
            n_threads=n_threads,
            seed=seed,
            horizon=horizon,
        )
        results_wrapper.add_results([results])
        tunnable_params = predictor.get_tunnable_params()
    result = {
        "outputs": results_wrapper.get_dict(),
        "tunnable_params": tunnable_params,
    }
    results_database.modify_result(database_params, result)
    if save_results:
        results_database.save()


dataset = "argoverse"
horizon = 30
alpha = 0.1
should_profile = False
n_threads = None
lr = 0.00000001
max_epochs = 100
name_string = f"{dataset}_betas{'_profile' if should_profile else ''}_horizon{horizon}"
results_database = ResultsDatabase("./results", name_string)


def make_cfrnn(alpha: float, horizon: int):
    return CFRNN(
        score_fn=distance_2d_conformal_score,
        alpha=alpha,
        horizon=horizon,
    )


def partial_maker(approximate_partition_size: int, epochs: int):
    def make_partial(alpha: float, horizon: int):
        return ConForME(
            score_fn=distance_2d_conformal_score,
            alpha=alpha,
            horizon=horizon,
            approximate_partition_size=approximate_partition_size,
            epochs=epochs,
            lr=lr,
        )

    return make_partial


partial_makes = [
    partial_maker(i, epochs) for i in [1, 2, 3, 10, 30] for epochs in [1, max_epochs]
]


def compute(make_conformal):
    evaluate_conformal_method(
        horizon,
        make_conformal_predictor=make_conformal,
        n_threads=n_threads,
        alpha=alpha,
        should_profile=should_profile,
        skip_existing=True,
        save_results=True,
    )


def cfcrnn_maker(optimize: bool):
    def make_cfcrnn(alpha: float, horizon: int):
        return ConForMEBin(
            score_fn=distance_2d_conformal_score,
            alpha=alpha,
            horizon=horizon,
            beta=0.5,
            optimize=optimize,
        )

    return make_cfcrnn

def beta_maker(beta: float):
    def make_cfcrnn(alpha: float, horizon: int):
        return ConForMEBin(
            score_fn=distance_2d_conformal_score,
            alpha=alpha,
            horizon=horizon,
            beta=beta,
            optimize=False,
        )

    return make_cfcrnn

cfcrnn_makes = [cfcrnn_maker(optimize) for optimize in [True, False]]
beta_makes = [beta_maker(beta) for beta in np.linspace(0.01, 0.99, 100)]

# makes = [make_cfrnn]
# makes = [partial_maker(40, 1)]
# makes = partial_makes + cfcrnn_makes + [make_cfrnn]
makes = beta_makes

# uses list because map is lazy
for make in tqdm(makes):
    compute(make)
