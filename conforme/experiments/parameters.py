from dataclasses import dataclass
from typing import Optional, Tuple, Union

from pydantic import BaseModel
from typing_extensions import Protocol

from ..data_processing.covid import get_covid_splits
from ..data_processing.dataset import Dataset1D
from ..data_processing.eeg import get_eeg_splits
from ..data_processing.types import SplitNumpy


class MedicalParameters(BaseModel):
    beta: float = 2 / 3
    batch_size: int = 150
    embedding_size: int = 20
    coverage: float = 0.9
    lr: float = 0.01
    n_steps: int = 1000
    input_size: int = 1
    rnn_mode: str = "LSTM"
    max_steps: Optional[int] = None
    output_size: Optional[int] = None


class SplitFunctionEEG(Protocol):
    def __call__(
        self,
        length: int = 40,
        horizon: int = 10,
        calibrate: float = 0.2,
        conformal: bool = True,
        cached: bool = True,
        seed: Optional[int] = None,
    ) -> Tuple[
        Union[Dataset1D, SplitNumpy],
        Union[Dataset1D, SplitNumpy, None],
        Union[Dataset1D, SplitNumpy],
    ]: ...


class SplitFunctionCOVID(Protocol):
    def __call__(
        self,
        length: int = 100,
        horizon: int = 50,
        conformal: bool = True,
        n_train: int = 200,
        n_calibration: int = 100,
        n_test: int = 80,
        cached: bool = True,
        seed: Optional[int] = None,
    ) -> Tuple[
        Union[Dataset1D, SplitNumpy],
        Union[Dataset1D, SplitNumpy, None],
        Union[Dataset1D, SplitNumpy],
    ]: ...


@dataclass
class MedicalAditionalParams:
    get_split: Union[SplitFunctionEEG, SplitFunctionCOVID]
    epochs: int
    horizon_length: int
    timeseries_length: int


def get_specific_parameters_for_medical_dataset(
    dataset: str, model_name: str
) -> MedicalAditionalParams:
    model_name_to_epochs = {
        "CFRNN": {"mimic": 1000, "eeg": 100, "covid": 1000},
        "CFCRNN": {"mimic": 1000, "eeg": 100, "covid": 1000},
    }

    if model_name not in model_name_to_epochs:
        raise ValueError(f"Invalid model_name: {model_name}")
    if dataset == "eeg":
        return MedicalAditionalParams(
            get_split=get_eeg_splits,
            epochs=model_name_to_epochs[model_name]["eeg"],
            horizon_length=10,
            timeseries_length=40,
        )
    elif dataset == "covid":
        return MedicalAditionalParams(
            get_split=get_covid_splits,
            epochs=model_name_to_epochs[model_name]["covid"],
            horizon_length=50,
            timeseries_length=100,
        )

    raise ValueError(f"Invalid dataset: {dataset}")


class SyntheticParameters(BaseModel):
    epochs: int = 1000
    normaliser_epochs: int = 1000
    n_steps: int = 500
    batch_size: int = 100
    embedding_size: int = 20
    max_steps: int = 10
    horizon: int = 5
    coverage: float = 0.9
    lr: float = 0.01
    beta: float = 2 / 3
    rnn_mode: str = "LSTM"
    output_size: Optional[int] = None
