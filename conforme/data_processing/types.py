from typing import Optional, Dict, Union, Tuple, List

import torch
import numpy.typing as npt
import numpy as np

NumpyFloatArray = npt.NDArray[np.float64]
SplitNumpy = Tuple[NumpyFloatArray, NumpyFloatArray]
StaticCustomParameters = Optional[Dict[str, Union[int, float, None]]]

# tensor for features, labels/values and sequence lengths
TrainSplit = Tuple[List[torch.Tensor], List[torch.Tensor], npt.NDArray[np.int64]]
TestSplit = Tuple[List[torch.Tensor], List[torch.Tensor], npt.NDArray[np.int64]]
TrainTestSplit = Tuple[TrainSplit, TestSplit]
