import torch

from .predictions import Targets1D, Targets2D


class ConformalScores:
    def __init__(self, scores: torch.Tensor):
        """
        scores: [n_samples, horizon, 1]
        """
        assert scores.ndim == 3
        assert scores.shape[2] == 1
        self._scores = scores

    @property
    def values(self) -> torch.Tensor:
        return torch.clone(self._scores)

    def __len__(self) -> int:
        return self._scores.shape[0]


def l1_conformal_score(targets: Targets1D, predictions: Targets1D) -> ConformalScores:
    scores = torch.nn.functional.l1_loss(
        predictions.values, targets.values, reduction="none"
    )

    return ConformalScores(scores)

def distance_2d_conformal_score(targets: Targets2D, predictions: Targets2D) -> ConformalScores:
    x_cordinate_difference = targets.values[:, :, 0]-predictions.values[:, :, 0]
    y_cordinate_difference = targets.values[:, :, 1]-predictions.values[:, :, 1]
    scores = torch.sqrt(x_cordinate_difference**2 + y_cordinate_difference**2)
    scores = scores.view(*scores.shape, 1)
    return ConformalScores(scores)
