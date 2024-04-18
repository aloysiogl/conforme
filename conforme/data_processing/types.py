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

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch

NumpyFloatArray = npt.NDArray[np.float64]
SplitNumpy = Tuple[NumpyFloatArray, NumpyFloatArray]
StaticCustomParameters = Optional[Dict[str, Union[int, float, None]]]

# tensor for features, labels/values and sequence lengths
TrainSplit = Tuple[List[torch.Tensor], List[torch.Tensor], npt.NDArray[np.int64]]
TestSplit = Tuple[List[torch.Tensor], List[torch.Tensor], npt.NDArray[np.int64]]
TrainTestSplit = Tuple[TrainSplit, TestSplit]
