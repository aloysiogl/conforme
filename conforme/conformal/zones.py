from abc import abstractmethod
from typing import Generic, Tuple, TypeVar


import torch

from .predictions import Targets, Targets1D, Targets2D

from .score import ConformalScores, distance_2d_conformal_score


T = TypeVar("T", bound=Targets)


class Zones(Generic[T]):
    def __init__(self, predictions: T, limiting_scores: ConformalScores):
        self._limiting_scores = limiting_scores
        self._predictions = predictions

    @abstractmethod
    def compute_coverage(self, targets: T) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def compute_zone_areas(self) -> torch.Tensor:
        pass


class L1IntervalZones(Zones[Targets1D]):
    def __init__(self, predictions: Targets1D, limiting_scores: ConformalScores):
        assert predictions.values.shape == limiting_scores.values.shape
        super().__init__(predictions, limiting_scores)

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

    def get_upper_lower(self):
        center = self._predictions.values
        lower = center - self._limiting_scores.values
        upper = center + self._limiting_scores.values

        return lower, upper

    def compute_coverage(self, targets: Targets1D):
        target_values = targets.values
        lower, upper = self.get_upper_lower()

        horizon_coverages = torch.logical_and(
            target_values >= lower, target_values <= upper
        )

        # [batch, horizon, n_outputs], [batch, n_outputs], lower, upper
        return horizon_coverages, torch.all(horizon_coverages, dim=1)

    def compute_zone_areas(self):
        lower, upper = self.get_upper_lower()
        interval_widths = (upper - lower).squeeze()

        return interval_widths


class DistanceZones(Zones[Targets2D]):
    def __init__(self, predictions: Targets2D, limiting_scores: ConformalScores):
        super().__init__(predictions, limiting_scores)

    def compute_zone_areas(self):
        scores = self._limiting_scores.values
        areas = torch.pi * scores**2

        return areas

    def compute_coverage(self, targets: Targets2D):
        comparison_scores = distance_2d_conformal_score(targets, self._predictions)

        assert comparison_scores.values.shape == self._limiting_scores.values.shape
        horizon_coverages = comparison_scores.values <= self._limiting_scores.values

        # [batch, horizon, n_outputs], [batch, n_outputs], lower, upper
        return horizon_coverages, torch.all(horizon_coverages, dim=1)
