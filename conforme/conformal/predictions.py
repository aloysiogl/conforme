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
