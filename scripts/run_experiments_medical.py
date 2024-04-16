from tqdm import tqdm
from typing import Callable, List

import numpy as np

from conforme.conformal.predictions import Targets1D
from conforme.conformal.predictor import (
    ConForMEBin,
    CFRNN,
    CFCEric,
    CFCRNNFull,
    ConForME,
    ConformalPredictor,
)
from conforme.conformal.score import l1_conformal_score
from conforme.result.results_database import ResultsDatabase
from conforme.train.medical import (
    run_medical_experiments,
)
from conforme.train.parameters import MedicalParameters
from conforme.result.containers import ResultsWrapper


def evaluate_conformal_method(
    dataset: str,
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
    params = MedicalParameters()
    params.coverage = 1 - alpha
    tunnable_params = None
    for seed in seeds:
        np.random.seed(seed)
        results, predictor = run_medical_experiments(
            dataset=dataset,
            baseline=baseline,
            retrain_model=False,
            save_model=False,
            save_results=True,
            horizon=horizon,
            n_threads=n_threads,
            params=params,
            make_conformal_predictor=make_pred,
            should_profile=profile,
            seed=seed,
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


# Genenal parameters
dataset = "eeg"
horizon = 10
alpha = 0.1
n_threads = None
should_profile = False
lr = 0.00000001
max_epochs = 100
name_string = f"{dataset}_betas{'_profile' if should_profile else ''}_horizon{horizon}"
results_database = ResultsDatabase("./results", name_string)


def eric_maker(epochs: int):
    def make_eric(alpha: float, horizon: int):
        return CFCEric(
            score_fn=l1_conformal_score,
            alpha=alpha,
            horizon=horizon,
            epochs=epochs,
            lr=lr,
        )

    return make_eric


eric_makes = [eric_maker(epochs) for epochs in [1, max_epochs]]


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


partial_makes = [
    partial_maker(i, epochs) for i in [1, 2, 5, 10, 25, 50] for epochs in [1]
]


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


def cfcrnn_full_maker(epochs: int):
    def make_cfcrnn_full(alpha: float, horizon: int):
        return CFCRNNFull(
            score_fn=l1_conformal_score,
            alpha=alpha,
            horizon=horizon,
            epochs=epochs,
            lr=lr,
        )

    return make_cfcrnn_full


cfcrnn_full_makes = [cfcrnn_full_maker(epochs) for epochs in [1, max_epochs]]


def make_cfrnn(alpha: float, horizon: int):
    return CFRNN(
        score_fn=l1_conformal_score,
        alpha=alpha,
        horizon=horizon,
    )


def compute(make_conformal):
    evaluate_conformal_method(
        dataset,
        horizon,
        make_conformal_predictor=make_conformal,
        n_threads=n_threads,
        alpha=alpha,
        profile=should_profile,
        skip_existing=False,
        save_results=True,
    )


# makes = eric_makes + partial_makes + cfcrnn_makes + cfcrnn_full_makes + [make_cfrnn]
# makes = partial_makes + cfcrnn_makes + [make_cfrnn]
makes = beta_makes
# makes = [make_cfrnn]
# makes = [partial_maker(40, 1)]

# uses list because map is lazy
for make in tqdm(makes):
    compute(make)
