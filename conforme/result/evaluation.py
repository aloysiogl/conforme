from typing import Type, TypeVar
import torch
from ..data_processing.dataset import Dataset1D
from ..conformal.predictions import Targets, Targets1D, Targets2D
from ..result.containers import Results
from ..conformal.zones import L1IntervalZones, DistanceZones, Zones
from ..conformal.predictor import ConformalPredictor

T = TypeVar("T", bound=Targets)

def evaluate_cfrnn_performance(
    predictions: Targets1D,
    test_dataset: Dataset1D,
    conformal_predictor: ConformalPredictor[Targets1D],
):
    limit_scores = conformal_predictor.limit_scores(predictions)
    l1_interval_zones = L1IntervalZones(predictions, limit_scores)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
    targets = Targets1D(torch.cat([batch[1] for batch in test_dataloader]))

    independent_coverages, joint_coverages, lower, upper = (
        l1_interval_zones.compute_coverage(targets)
    )

    mean_independent_coverage = torch.mean(independent_coverages.float(), dim=0)
    mean_joint_coverage = torch.mean(joint_coverages.float(), dim=0).item()
    interval_widths = (upper - lower).squeeze()
    errors = predictions.values - targets.values

    # TODO: one of the next steps is to only rely on the point prediction and do conformal part outside
    evaluate_performance(predictions, targets, conformal_predictor, L1IntervalZones)

    return Results(
        point_predictions=predictions,
        errors=errors,
        independent_coverage_indicators=independent_coverages.squeeze(),
        joint_coverage_indicators=joint_coverages.squeeze(),
        mean_independent_coverage=mean_independent_coverage,
        mean_joint_coverage=mean_joint_coverage,
        confidence_zone_area=interval_widths,
        mean_confidence_zone_area_per_horizon=interval_widths.mean(dim=0),
        mean_confidence_zone_area=interval_widths.mean().item(),
        min_confidence_zone_area=interval_widths.min(),
        max_confidence_zone_area=interval_widths.max(),
    )



def evaluate_performance(
    predictions: T,
    targets: T,
    conformal_predictor: ConformalPredictor[T],
    zone_constructor: Type[Zones[T]], 
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
