from typing import Type, TypeVar

import torch

from ..conformal.predictions import Targets
from ..conformal.predictor import ConformalPredictor
from ..conformal.zones import Zones
from ..result.containers import Results

T = TypeVar("T", bound=Targets)

def evaluate_performance(
    zone_constructor: Type[Zones[T]], 
    predictions: T,
    targets: T,
    conformal_predictor: ConformalPredictor[T],
):
    errors = predictions.values - targets.values
    zones = zone_constructor(predictions, conformal_predictor.limit_scores(predictions))
    independent_coverages, joint_coverages = (
        zones.compute_coverage(targets)
    )

    mean_independent_coverage = torch.mean(independent_coverages.float(), dim=0).squeeze()
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
        min_confidence_zone_area=areas.min().item(), # type: ignore
        max_confidence_zone_area=areas.max().item(), # type: ignore
    )
