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

from abc import abstractmethod

import torch


class Targets:
    def __init__(self, predictions: torch.Tensor) -> None:
        self._predictions = predictions

    @abstractmethod
    def __len__(self) -> int:
        pass

    @property
    def values(self) -> torch.Tensor:
        return torch.clone(self._predictions)


class Targets1D(Targets):
    def __init__(self, predictions: torch.Tensor) -> None:
        """
        predictions: [n_samples, horizon, 1]
        """
        assert predictions.ndim == 3
        assert predictions.shape[2] == 1
        super().__init__(predictions)

    def __len__(self) -> int:
        return self._predictions.shape[0]


class Targets2D(Targets):
    def __init__(self, predictions: torch.Tensor) -> None:
        """
        predictions: [n_samples, horizon, 2]
        """
        assert predictions.ndim == 3
        assert predictions.shape[2] == 2
        super().__init__(predictions)

    def __len__(self) -> int:
        return self._predictions.shape[0]
