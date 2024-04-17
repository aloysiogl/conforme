from typing import Callable, List, Tuple, TypeVar

from tqdm import tqdm
from conforme.conformal.predictions import Targets, Targets2D
from conforme.conformal.predictor import ConForMEParams, ConformalPredictor, ConformalPredictorParams, get_cfrnn_maker, get_conforme_maker
from conforme.conformal.score import distance_2d_conformal_score
from conforme.conformal.zones import DistanceZones
from conforme.data_processing.argoverse import get_calibration_test as get_calibration_test_argoverse
from conforme.experiments.run import evaluate_conformal_method, run_experiment
from conforme.result.containers import Results
from conforme.result.results_database import ResultsDatabase

T = TypeVar("T", bound=Targets)


def evaluate_dataset(
    dataset: str,
    results_database: ResultsDatabase,
    get_runner: Callable[[Callable[[], ConformalPredictor[T]], bool], Callable[[], Tuple[Results, ConformalPredictor[T]]]],
    make_cp: Callable[[], ConformalPredictor[T]],
    should_profile: bool,
    seeds: List[int],
    skip_existing: bool,
    save_results: bool,
):
    params = {
        "dataset": dataset,
        "nb_seeds": len(seeds),
        "conformal_predictor_params": make_cp().get_params(),
        "method": make_cp().get_name(),
    }

    evaluate_conformal_method(
        get_runner(make_cp, should_profile),
        results_database,
        params,
        seeds,
        skip_existing,
        save_results
    )


def get_argoverse_runner(
        make_cp: Callable[[], ConformalPredictor[Targets2D]],
        should_profile: bool,
):
    def run():
        return run_experiment(
            get_calibration_test_argoverse,
            make_cp,
            DistanceZones,
            should_profile,
        )
    return run


argoverse_general_params = ConformalPredictorParams(
    alpha=0.1,
    horizon=30,
    score_fn=distance_2d_conformal_score,
)


argoverse_cp_makers = [
    get_conforme_maker(
        ConForMEParams(
            general_params=argoverse_general_params,
            approximate_partition_size=s,
            epochs=e,
            lr=0.00000001)) for s in [1, 2, 3, 10, 30] for e in [1, 100]
] + [
    get_cfrnn_maker(argoverse_general_params)
]


dataset = "argoverse"
should_profile = True
name_string = f"{dataset}{'_pftest' if should_profile else ''}_horizon{argoverse_general_params.horizon}"
results_database = ResultsDatabase("./results", name_string)

for make_cp in tqdm(argoverse_cp_makers):
    evaluate_dataset(
        dataset,
        results_database,
        get_argoverse_runner,
        make_cp,
        should_profile,
        seeds=[0, 1, 2, 3, 4],
        skip_existing=False,
        save_results=True
    )
