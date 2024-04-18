from typing import Any, Callable, Dict, List, Tuple, Type, Union

import numpy as np
import torch
from tqdm import tqdm

from conforme.result.results_database import ResultsDatabase

from ..conformal.predictions import Targets
from ..conformal.predictor import ConformalPredictor, ConformalPredictorParams
from ..conformal.zones import Zones
from ..result.containers import Results, ResultsWrapper


def evaluate_performance[T: Targets](
    zone_constructor: Type[Zones[T]],
    predictions: T,
    targets: T,
    conformal_predictor: ConformalPredictor[T],
):
    errors = predictions.values - targets.values
    zones = zone_constructor(
        predictions, conformal_predictor.limit_scores(predictions))
    independent_coverages, joint_coverages = (
        zones.compute_coverage(targets)
    )

    mean_independent_coverage = torch.mean(
        independent_coverages.float(), dim=0).squeeze()
    mean_joint_coverage = torch.mean(joint_coverages.float(), dim=0).item()
    areas = zones.compute_zone_areas().squeeze()

    return Results(
        point_predictions=predictions,
        errors=errors,
        independent_coverage_indicators=independent_coverages.squeeze(),
        joint_coverage_indicators=joint_coverages.squeeze(),
        mean_independent_coverage=mean_independent_coverage,
        mean_joint_coverage=mean_joint_coverage,
        confidence_zone_area=areas,
        mean_confidence_zone_area_per_horizon=areas.mean(dim=0),
        mean_confidence_zone_area=areas.mean().item(),
        min_confidence_zone_area=areas.min().item(),  # type: ignore
        max_confidence_zone_area=areas.max().item(),  # type: ignore
    )


def evaluate_conformal_method[T: Targets](
    run_experiment: Callable[[int], Tuple[Results, ConformalPredictor[T]]],
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
        results, predictor = run_experiment(seed)
        results_wrapper.add_results([results])
        tunnable_params = predictor.get_tunnable_params()

    result = {
        "outputs": results_wrapper.get_dict(),
        "tunnable_params": tunnable_params,
    }
    results_database.modify_result(params, result)

    if save_results:
        results_database.save()


def evaluate_dataset[T: Targets](
    dataset: str,
    results_database: ResultsDatabase,
    get_runner: Callable[[Callable[[], ConformalPredictor[T]], bool], Callable[[int], Tuple[Results, ConformalPredictor[T]]]],
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


def evaluate_experiments_for_dataset[T: Targets](dataset: str,
                                     should_profile: bool,
                                     general_params: ConformalPredictorParams[T],
                                     cp_makers: list[Callable[[], ConformalPredictor[T]]],
                                     get_runner: Callable[[Callable[[], ConformalPredictor[T]], bool], Callable[[int], Tuple[Results, ConformalPredictor[T]]]],
                                     experience_suffix: str = "",
                                     seeds: List[int] = [0, 1, 2, 3, 4]):
    name_string = f"{dataset}{'_profile' if should_profile else ''}_horizon{general_params.horizon}" \
        f"{'_' if experience_suffix else ''}{experience_suffix}"
    results_database = ResultsDatabase("./results", name_string)

    for make_cp in tqdm(cp_makers):
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
