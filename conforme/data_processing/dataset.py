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

from typing import Any, Tuple, Union

import torch


class Dataset1D(torch.utils.data.Dataset):  # type: ignore
    def __init__(
        self, X: torch.Tensor, Y: torch.Tensor, sequence_lengths: torch.Tensor
    ):
        super(Dataset1D, self).__init__()  # type: ignore
        assert X.ndim == 3, f"X.ndim = {X.ndim}, X.shape = {X.shape}"
        assert X.shape[2] == 1, f"X.shape[2] = {X.shape[2]}"
        assert Y.ndim == 3, f"Y.ndim = {Y.ndim}, Y.shape = {Y.shape}"
        assert Y.shape[2] == 1, f"Y.shape[2] = {Y.shape[2]}"
        assert (
            sequence_lengths.ndim == 1
        ), f"sequence_lengths.ndim = {sequence_lengths.ndim}, sequence_lengths.shape = {sequence_lengths.shape}"
        assert (
            sequence_lengths.shape[0] == X.shape[0]
        ), f"sequence_lengths.shape[0] = {sequence_lengths.shape[0]}, X.shape[0] = {X.shape[0]}"
        assert (
            sequence_lengths.shape[0] == Y.shape[0]
        ), f"sequence_lengths.shape[0] = {sequence_lengths.shape[0]}, Y.shape[0] = {Y.shape[0]}"
        self.X = X
        self.Y = Y
        self.sequence_lengths = sequence_lengths

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx], self.sequence_lengths[idx]


def ensure_1d_dataset_split(
    split: Tuple[Union[Any, Dataset1D], Union[Any, Dataset1D], Union[Any, Dataset1D]],
) -> Tuple[Dataset1D, Dataset1D, Dataset1D]:
    if any([not isinstance(split[i], Dataset1D) for i in range(3)]):
        raise ValueError("All splits must be of type Dataset1D")
    return split
