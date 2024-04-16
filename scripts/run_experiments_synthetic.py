from typing import Callable, List
import numpy as np
import pickle

from tqdm import tqdm

from conforme.conformal.predictions import Targets1D
from conforme.conformal.predictor import ConForMEBin, CFRNN, ConForME, ConformalPredictor
from conforme.conformal.score import l1_conformal_score
from conforme.result.containers import ResultsWrapper

from conforme.result.results_database import ResultsDatabase
from conforme.train.synthetic import (
    run_synthetic_experiments,
)
from conforme.train.parameters import SyntheticParameters

def evaluate_conformal_method(
    experiment_type: str,
    horizon: int,
    make_conformal_predictor: Callable[[float, int], ConformalPredictor[Targets1D]],
    seeds: List[int] = [0, 1, 2, 3, 4],
    alpha: float = 0.1,
    profile: bool = False,
    n_threads: int = 1,
    skip_existing: bool = False,
    save_results: bool = True,
):
    # TODO remove
    baseline = "CFCRNN"

    def make_pred():
        return make_conformal_predictor(alpha, horizon)

    results_wrappers: List[ResultsWrapper] = []
    for seed in seeds:
        np.random.seed(seed)
        params = SyntheticParameters()
        params.coverage = 1 - alpha
        results, predictors = run_synthetic_experiments(
            experiment=experiment_type,
            baseline=baseline,
            n_train=2000,
            horizon=horizon,
            retrain_auxiliary=False,
            recompute_dataset=False,
            save_model=False,
            save_results=True,
            make_conformal_predictor=make_pred,
            should_profile=profile,
            n_threads=n_threads,
            params=params,
            seed=seed,
        )
        if len(results_wrappers) == 0:
            results_wrappers = [ResultsWrapper() for _ in range(len(results))]
        for i in range(len(results)):
            results_wrappers[i].add_results([results[i]])

    for i in range(len(results_wrappers)):
        database_params = {
            "nb_seeds": len(seeds),
            "horizon": horizon,
            "alpha": alpha,
            "conformal_predictor_params": make_pred().get_params(),
            "method": make_pred().get_name(),
            "experiment_index": i,
            "experiment_type": experiment_type,
        }

        results_wrapper = results_wrappers[i]
        tunnable_params = predictors[i].get_tunnable_params()

        result = {
            "outputs": results_wrapper.get_dict(),
            "tunnable_params": tunnable_params,
        }
        results_database.modify_result(database_params, result)

        if save_results:
            results_database.save()

experiment = "static"
horizon = 10
alpha = 0.1
lr = 0.0000001
n_threads = None
should_profile = False
results_database = ResultsDatabase("./results", "static_betas_horizon10")
max_epochs = 50

def cfcrnn_maker(optimize: bool):
    def make_cfcrnn(alpha: float, horizon: int):
        return ConForMEBin(
            score_fn=l1_conformal_score,
            alpha=alpha,
            horizon=horizon,
            beta=0.5,
            optimize=optimize,
        )

    return make_cfcrnn

def beta_maker(beta: float):
    def make_cfcrnn(alpha: float, horizon: int):
        return ConForMEBin(
            score_fn=l1_conformal_score,
            alpha=alpha,
            horizon=horizon,
            beta=beta,
            optimize=False,
        )

    return make_cfcrnn


cfcrnn_makes = [cfcrnn_maker(optimize) for optimize in [True, False]]
beta_makes = [beta_maker(beta) for beta in np.linspace(0.01, 0.99, 100)]

def partial_maker(approximate_partition_size: int, epochs: int):
    def make_partial(alpha: float, horizon: int):
        return ConForME(
            score_fn=l1_conformal_score,
            alpha=alpha,
            horizon=horizon,
            approximate_partition_size=approximate_partition_size,
            epochs=epochs,
            lr=lr,
        )

    return make_partial

partial_makes = [partial_maker(i, epochs) for i in [1, 2, 3, 5, 10, 20, 40] for epochs in [1, max_epochs]]

def make_cfrnn(alpha: float, horizon: int):
    return CFRNN(
        score_fn=l1_conformal_score,
        alpha=alpha,
        horizon=horizon,
    )


def compute(make_conformal):
    evaluate_conformal_method(
        experiment,
        horizon,
        make_conformal_predictor=make_conformal,
        n_threads=n_threads,
        alpha=alpha,
        profile=should_profile,
        skip_existing=True,
        save_results=True,
    )

# makes = [partial_maker(40, 1)]
# makes = [make_cfrnn] + partial_makes + cfcrnn_makes
makes = beta_makes

# uses list because map is lazy
for make in tqdm(makes):
    compute(make)