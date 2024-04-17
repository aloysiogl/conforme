from typing import Callable, List, TypeVar

from tqdm import tqdm
from conforme.conformal.predictions import Targets, Targets1D
from conforme.conformal.predictor import ConForMEParams, ConformalPredictor, ConformalPredictorParams, get_cfrnn_maker, get_conforme_maker
from conforme.conformal.score import l1_conformal_score, distance_2d_conformal_score
from conforme.experiments.run import get_argoverse_runner, get_medical_runner
from conforme.result.evaluation import evaluate_dataset
from conforme.result.results_database import ResultsDatabase

T = TypeVar("T", bound=Targets)


def run_experiments_for_dataset(dataset: str,
                                should_profile: bool,
                                general_params: ConformalPredictorParams[T],
                                cp_makers: List[Callable[[], Callable[[], ConformalPredictor[T]]]],
                                experience_suffix: str = "",
                                seeds: List[int] = [0, 1, 2, 3, 4]):
    name_string = f"{dataset}{'_profile' if should_profile else ''}_horizon{eeg10_general_params.horizon} \
                    {'_' if experience_suffix else ''}{experience_suffix}"
    results_database = ResultsDatabase("./results", name_string)

    for make_cp in tqdm(cp_makers):
        def get_runner(make_cp: Callable[[], ConformalPredictor[Targets1D]], should_profile: bool):
            return get_medical_runner(
                make_cp,
                should_profile,
                dataset,
                save_model=True,
                retrain_model=False,
                horizon=general_params.horizon
            )
        evaluate_dataset(
            dataset,
            results_database,
            get_runner,
            make_cp,
            should_profile,
            seeds=seeds,
            skip_existing=True,
            save_results=True
        )


"""Parameter definitions for the conoformal predictors in each experiment """

synthetic_general_params = ConformalPredictorParams(
    alpha=0.1,
    horizon=10,
    score_fn=l1_conformal_score,
)

synthetic_cp_makers = [
    get_conforme_maker(
        ConForMEParams(
            general_params=synthetic_general_params,
            approximate_partition_size=s,
            epochs=e,
            lr=0.00000001)) for s in [1, 2, 3, 10, 30] for e in [1, 100]
] + [
    get_cfrnn_maker(synthetic_general_params)
]

eeg10_general_params = ConformalPredictorParams(
    alpha=0.1,
    horizon=10,
    score_fn=l1_conformal_score,
)

eeg10_cp_makers = [
    get_conforme_maker(
        ConForMEParams(
            general_params=eeg10_general_params,
            approximate_partition_size=s,
            epochs=e,
            lr=0.00000001)) for s in [1, 2, 3, 5, 10] for e in [1, 100]
] + [
    get_cfrnn_maker(eeg10_general_params)
]

eeg40_general_params = ConformalPredictorParams(
    alpha=0.1,
    horizon=40,
    score_fn=l1_conformal_score,
)

eeg40_cp_makers = [
    get_conforme_maker(
        ConForMEParams(
            general_params=eeg10_general_params,
            approximate_partition_size=s,
            epochs=e,
            lr=0.00000001)) for s in [1, 2, 3, 5, 10] for e in [1, 100]
] + [
    get_cfrnn_maker(eeg10_general_params)
]

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


"""Running the experiments"""

# run_experiments_for_dataset("argoverse", True, argoverse_general_params, argoverse_cp_makers)

# run_experiments_for_dataset("eeg", True, eeg10_general_params, eeg10_cp_makers)

# run_experiments_for_dataset("eeg", True, eeg40_general_params, eeg10_cp_makers)
