from dataclasses import dataclass
from typing import Optional, Tuple, Union

from pydantic import BaseModel
from typing_extensions import Protocol

from ..data_processing.covid import get_covid_splits
from ..data_processing.dataset import Dataset1D
from ..data_processing.eeg import get_eeg_splits
from ..data_processing.types import SplitNumpy


class MedicalParameters(BaseModel):
    batch_size: int = 150
    embedding_size: int = 20
    lr: float = 0.01
    rnn_mode: str = "LSTM"


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
    dataset: str
) -> MedicalAditionalParams:

    epochs_per_dataset = {"eeg": 100, "covid": 1000}

    if dataset == "eeg":
        return MedicalAditionalParams(
            get_split=get_eeg_splits,
            epochs=epochs_per_dataset["eeg"],
            horizon_length=10,
            timeseries_length=40,
        )
    elif dataset == "covid":
        return MedicalAditionalParams(
            get_split=get_covid_splits,
            epochs=epochs_per_dataset["covid"],
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
    lr: float = 0.01
    beta: float = 2 / 3
    rnn_mode: str = "LSTM"
    output_size: Optional[int] = None
    n_train: Optional[int] = None
