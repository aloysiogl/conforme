#
# This file is part of https://github.com/aloysiogl/conforme.
# Copyright (c) 2024 Aloysio Galvao Lopes.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

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


def distance_2d_conformal_score(
    targets: Targets2D, predictions: Targets2D
) -> ConformalScores:
    x_cordinate_difference = targets.values[:, :, 0] - predictions.values[:, :, 0]
    y_cordinate_difference = targets.values[:, :, 1] - predictions.values[:, :, 1]
    scores = torch.sqrt(x_cordinate_difference**2 + y_cordinate_difference**2)
    scores = scores.view(*scores.shape, 1)
    return ConformalScores(scores)
