import torch

from .predictions import Targets1D, Targets2D
from .score import ConformalScores, distance_2d_conformal_score


class Zones:
    def __init__(self, limiting_scores: ConformalScores):
        self._limiting_scores = limiting_scores


class L1IntervalZones(Zones):
    def __init__(self, predictions: Targets1D, limiting_scores: ConformalScores):
        assert predictions.values.shape == limiting_scores.values.shape
        super().__init__(limiting_scores)
        self._interval_centers = predictions

    @staticmethod
    def from_intervals(intervals: torch.Tensor):
        """
        intervals: [n_samples, 2, horizon, 1]
        """
        assert intervals.shape[1] == 2
        assert intervals.shape[3] == 1
        assert intervals.ndim == 4

        lower, upper = intervals[:, 0], intervals[:, 1]

        center = (lower + upper) / 2
        scores = upper - center

        return L1IntervalZones(Targets1D(center), ConformalScores(scores))

    def compute_coverage(self, targets: Targets1D):
        target_values = targets.values
        center = self._interval_centers.values
        lower = center - self._limiting_scores.values
        upper = center + self._limiting_scores.values

        horizon_coverages = torch.logical_and(
            target_values >= lower, target_values <= upper
        )

        # [batch, horizon, n_outputs], [batch, n_outputs], lower, upper
        return horizon_coverages, torch.all(horizon_coverages, dim=1), lower, upper


class DistanceZones(Zones):
    def __init__(self, predictions: Targets2D, limiting_scores: ConformalScores):
        super().__init__(limiting_scores)
        self._zone_centers = predictions
    
    def compute_zone_areas(self):
        scores = self._limiting_scores.values
        areas = torch.pi*scores**2

        return areas

    def compute_coverage(self, targets: Targets2D):
        comparison_scores = distance_2d_conformal_score(targets, self._zone_centers)

        assert comparison_scores.values.shape == self._limiting_scores.values.shape
        horizon_coverages = comparison_scores.values <= self._limiting_scores.values

        # [batch, horizon, n_outputs], [batch, n_outputs], lower, upper
        return horizon_coverages, torch.all(horizon_coverages, dim=1)
