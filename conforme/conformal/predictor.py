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
from dataclasses import dataclass
from math import ceil
from typing import Any, Callable, List, Optional

import torch
import torch.nn as nn

from .predictions import Targets
from .score import ConformalScores


@dataclass
class ConformalPredictorParams[P: Targets]:
    alpha: float
    horizon: int
    score_fn: Callable[[P, P], ConformalScores]


class ConformalPredictor[P: Targets]:
    def __init__(
        self, score_fn: Callable[[P, P], ConformalScores], alpha: float, horizon: int
    ) -> None:
        self._score_fn = score_fn
        self._scores: Optional[ConformalScores] = None
        self._alpha = alpha
        self._horizon = horizon
        self._limit_scores = None
        self._n_computations = 0

    @abstractmethod
    def calibrate(self, targets: P, predictions: P):
        pass

    @staticmethod
    def get_q(alpha: float, n_calibration: int) -> Optional[float]:
        if alpha < 1 / (n_calibration + 1):
            return None
        return min((n_calibration + 1.0) * (1 - alpha) / n_calibration, 1)

    @abstractmethod
    def limit_scores(self, targets: P) -> ConformalScores:
        pass

    def get_params(self):
        return {
            "alpha": self._alpha,
            "horizon": self._horizon,
            "method": self.get_name(),
        }

    def get_tunnable_params(self) -> dict[str, Any]:
        return {
            "n_computations": self._n_computations,
        }

    def get_name(self):
        return self.__class__.__name__


class CFRNN[P: Targets](ConformalPredictor[P]):
    def __init__(
        self,
        score_fn: Callable[[P, P], ConformalScores],
        alpha: float,
        horizon: int,
    ) -> None:
        super().__init__(score_fn, alpha, horizon)

    def calibrate(self, targets: P, predictions: P):
        self._scores = self._score_fn(targets, predictions)
        bonferroni_alpha = self._alpha / self._horizon
        q = ConformalPredictor.get_q(
            bonferroni_alpha,
            len(targets),
        )
        if q is None:
            self._limit_scores = torch.ones(self._horizon) * float("inf")
        else:
            values = self._scores.values
            values = values.squeeze(-1)
            quantiles = torch.quantile(values, q, dim=0)
            assert quantiles.ndim == 1
            assert quantiles.shape[0] == self._horizon
            self._limit_scores = quantiles
        self._n_computations += 1

    def limit_scores(self, targets: P) -> ConformalScores:
        target_values = targets.values
        target_shape = target_values.shape
        return ConformalScores(
            self._limit_scores.repeat(target_values.shape[0]).view(
                target_shape[0], target_shape[1], 1
            )
        )


def get_cfrnn_maker[P: Targets](params: ConformalPredictorParams[P]):
    def make():
        return CFRNN(
            alpha=params.alpha,
            horizon=params.horizon,
            score_fn=params.score_fn,
        )

    return make


@dataclass
class ConForMEBinParams[P: Targets]:
    beta: float
    optimize: bool
    general_params: ConformalPredictorParams[P]
    epsilon_binary_search: float = 0.01


class ConForMEBin[P: Targets](ConformalPredictor[P]):
    def __init__(
        self,
        score_fn: Callable[[P, P], ConformalScores],
        alpha: float,
        beta: float,
        horizon: int,
        optimize: bool,
        epsilon_binary_search: float,
    ) -> None:
        super().__init__(score_fn, alpha, horizon)
        self._beta = beta
        self._optimze = optimize
        self._eps = epsilon_binary_search
        self.n_computations = 0

    def compute_ending_block_alpha(
        self, block_alpha: float, starting_block_alpha: float
    ):
        return (block_alpha - starting_block_alpha) / (1 - starting_block_alpha)

    def set_beta(self, beta: float):
        self._beta = beta

    def predict(self, targets: P, predictions: P):
        self._scores = self._score_fn(targets, predictions)
        scores = self._scores.values
        scores = scores.squeeze(-1)

        consider_last_element = self._horizon % 2 == 1
        effective_horizon = ceil(self._horizon / 2)

        pair_alpha = self._alpha / effective_horizon

        # notation inversed from paper as index starts at 0
        even_alpha = self._beta * pair_alpha
        odd_alpha = self.compute_ending_block_alpha(pair_alpha, even_alpha)

        def limit_score_for_sequence(
            sequence: torch.Tensor, alpha: float
        ) -> torch.Tensor:
            q = ConformalPredictor.get_q(alpha, len(sequence))
            if q is None:
                return torch.tensor(float("inf"))
            return torch.quantile(sequence, q)

        limit_scores: List[torch.Tensor] = []

        for i in range(effective_horizon):
            if i == effective_horizon - 1 and consider_last_element:
                last_sequence = scores[:, -1]
                limit_scores.append(limit_score_for_sequence(last_sequence, pair_alpha))
            else:
                idx = 2 * i
                even_sequence = scores[:, idx]
                even_limit_score = limit_score_for_sequence(even_sequence, even_alpha)
                idx = 2 * i + 1
                odd_scores = scores[:, idx]
                odd_sequence = odd_scores[even_sequence <= even_limit_score]
                odd_limit_score = limit_score_for_sequence(odd_sequence, odd_alpha)

                limit_scores.append(even_limit_score)
                limit_scores.append(odd_limit_score)

        stacked_limit_scores = torch.stack(limit_scores)
        assert stacked_limit_scores.ndim == 1
        assert stacked_limit_scores.shape[0] == self._horizon
        self._n_computations += 1
        return stacked_limit_scores

    def calibrate(self, targets: P, predictions: P):
        def performance_metric_for_beta(beta: float):
            self.set_beta(beta)
            scores = self.predict(targets, predictions)
            return torch.mean(scores)

        l_beta = 0
        r_beta = 1
        eps = 0.01
        best_beta = self._beta

        if self._optimze:
            while r_beta - l_beta > eps:
                beta = (l_beta + r_beta) / 2
                if performance_metric_for_beta(beta) < performance_metric_for_beta(
                    l_beta
                ):
                    l_beta = beta
                else:
                    r_beta = beta
                best_beta = beta

        self.set_beta(best_beta)
        self._limit_scores = self.predict(targets, predictions)

    def get_params(self):
        params = super().get_params()
        additional_params = {
            "optimize": self._optimze,
            "epsilon_binary_search": self._eps,
        }
        return {**params, **additional_params}

    def get_name(self):
        if self._optimze:
            return f"{self.__class__.__name__}Optim"
        return self.__class__.__name__

    def get_tunnable_params(self):
        return {
            **super().get_tunnable_params(),
            "beta": self._beta,
        }

    def limit_scores(self, targets: P) -> ConformalScores:
        target_values = targets.values
        target_shape = target_values.shape
        return ConformalScores(
            self._limit_scores.repeat(target_values.shape[0]).view(
                target_shape[0], target_shape[1], 1
            )
        )


def get_conformebin_maker[P: Targets](params: ConForMEBinParams[P]):
    def make():
        return ConForMEBin(
            alpha=params.general_params.alpha,
            horizon=params.general_params.horizon,
            score_fn=params.general_params.score_fn,
            beta=params.beta,
            optimize=params.optimize,
            epsilon_binary_search=params.epsilon_binary_search,
        )

    return make


@dataclass
class ConForMEParams[P: Targets]:
    approximate_partition_size: int
    epochs: int
    lr: float
    general_params: ConformalPredictorParams[P]


class ConForME[P: Targets](nn.Module, ConformalPredictor[P]):
    def __init__(
        self,
        score_fn: Callable[[P, P], ConformalScores],
        alpha: float,
        horizon: int,
        approximate_partition_size: int,
        epochs: int = 200,
        lr: float = 0.000001,
    ) -> None:
        nn.Module.__init__(self)  # type: ignore
        ConformalPredictor.__init__(self, score_fn, alpha, horizon)  # type: ignore
        n_blocks = ConForME.number_of_blocks(horizon, approximate_partition_size)
        self._lr = lr
        self._epochs = epochs
        self._approximate_partition_size = approximate_partition_size
        base_block_size = horizon // n_blocks
        self._block_sizes = [
            base_block_size + 1 if i < horizon % n_blocks else base_block_size
            for i in range(n_blocks)
        ]
        assert sum(self._block_sizes) == horizon
        effective_alpha = alpha / (n_blocks)
        self._alphas = nn.Parameter(torch.ones(n_blocks) * effective_alpha)
        self._alphas_per_block = nn.ParameterList(
            [
                nn.Parameter(torch.ones(self._block_sizes[i]))
                * self._alpha
                / self._horizon
                for i in range(n_blocks)
            ]
        )

    @staticmethod
    def number_of_blocks(horizon: int, approximate_partition_size: int) -> int:
        return ceil(horizon / approximate_partition_size)

    def predict(self, targets: P, predictions: P):
        scores = self._score_fn(targets, predictions).values
        scores = scores.squeeze(-1)

        def limit_score_for_sequence(
            sequence: torch.Tensor, alpha: float
        ) -> torch.Tensor:
            q = ConformalPredictor.get_q(alpha, len(sequence))
            if q is None:
                return torch.tensor(float("inf"))
            return torch.quantile(sequence, q)

        idx = 0
        all_limit_scores: list[torch.Tensor] = []
        for i in range(len(self._alphas_per_block)):
            limit_scores: List[torch.Tensor] = []
            limit_scores.append(torch.tensor(float("inf")))
            cal_points_mask = torch.ones(scores[:, idx].shape, dtype=torch.bool)

            for j in range(len(self._alphas_per_block[i])):
                previous_limits = limit_scores[j]
                previous_sequence = (
                    torch.ones(scores[:, 0].shape) * float("inf")
                    if j == 0
                    else scores[:, idx - 1]
                )
                previous_sequence_in_previous_limits = (
                    previous_sequence <= previous_limits
                )

                cal_points_mask = cal_points_mask & previous_sequence_in_previous_limits
                current_scores = scores[:, idx]
                current_selected_scores = current_scores[cal_points_mask]
                current_limit_scores = limit_score_for_sequence(
                    current_selected_scores, self._alphas_per_block[i][j]
                )

                limit_scores.append(current_limit_scores)
                idx += 1

            limit_scores_tensor = torch.stack(limit_scores)
            limit_scores_tensor = limit_scores_tensor[1:]
            all_limit_scores.append(limit_scores_tensor)

        alphas_list: list[nn.Parameter] = [t for t in self._alphas_per_block]
        limit_scores_tensor = torch.cat(all_limit_scores)
        self._n_computations += 1

        return limit_scores_tensor, alphas_list

    def calibrate(self, targets: P, predictions: P):
        optimizer = torch.optim.SGD(self.parameters(), lr=self._lr)
        criterion = torch.nn.L1Loss()
        if self._epochs <= 1:
            self._limit_scores, _ = self.predict(targets, predictions)
            return

        self.train()
        best_loss = float("inf")
        best_scores = None
        for epoch in range(self._epochs):
            optimizer.zero_grad()

            limit_scores, alphas_list = self.predict(targets, predictions)
            loss = limit_scores

            loss1 = criterion(limit_scores, torch.zeros_like(limit_scores))

            def compute_effective_one_minus_alpha(alphas_list: list[nn.Parameter]):
                prods = [torch.prod(1 - alphas) for alphas in alphas_list]
                block_alphas = [1 - prod for prod in prods]
                tot_alpha = torch.sum(torch.stack(block_alphas))
                return 1 - tot_alpha

            effective_one_minus_alpha = compute_effective_one_minus_alpha(alphas_list)

            alpha_diff = torch.abs(
                effective_one_minus_alpha - (1 - torch.tensor(self._alpha))
            )
            loss2 = torch.tan(alpha_diff * (torch.pi / 2) / 0.01)
            one_minus_alphas_sum = 1 - effective_one_minus_alpha.detach().item()
            current_loss_1 = loss1.detach().item()
            loss = loss1 + loss2
            current_loss = loss.detach().item()
            loss.backward()
            optimizer.step()
            if current_loss < best_loss:
                best_loss = current_loss
                best_scores = limit_scores.detach().clone()

            print(
                f"Epoch: {epoch}, current loss: {current_loss}, current loss 1: {
                    current_loss_1}, alphas_sum: {one_minus_alphas_sum}"
            )

        assert best_scores is not None, "Best scores should not be none at this stage"

        self._limit_scores = best_scores
        print(
            self._limit_scores,
            self._limit_scores.detach().mean(),
        )

    def get_params(self):
        params = super().get_params()
        additional_params = {
            "approximate_partition_size": self._approximate_partition_size,
            "mean_block_size": sum(self._block_sizes) / len(self._block_sizes),
            "epochs": self._epochs,
            "lr": self._lr,
        }
        return {**params, **additional_params}

    def get_name(self):
        if self._epochs == 1:
            return f"{self.__class__.__name__}{len(self._block_sizes)}"
        return f"{self.__class__.__name__}{len(self._block_sizes)}Optim"

    def get_tunnable_params(self):
        return {
            **super().get_tunnable_params(),
            "alphas_per_block": [t.tolist() for t in self._alphas_per_block],
            "alphas_per_block_prods": [
                1 - torch.prod(1 - t).item() for t in self._alphas_per_block
            ],
        }

    def limit_scores(self, targets: P) -> ConformalScores:
        target_values = targets.values
        target_shape = target_values.shape
        return ConformalScores(
            self._limit_scores.repeat(target_values.shape[0]).view(
                target_shape[0], target_shape[1], 1
            )
        )


def get_conforme_maker[P: Targets](params: ConForMEParams[P]):
    def make():
        return ConForME(
            score_fn=params.general_params.score_fn,
            alpha=params.general_params.alpha,
            horizon=params.general_params.horizon,
            approximate_partition_size=params.approximate_partition_size,
            epochs=params.epochs,
            lr=params.lr,
        )

    return make


class CFCOverlapBinary[P: Targets](nn.Module, ConformalPredictor[P]):
    def __init__(
        self,
        score_fn: Callable[[P, P], ConformalScores],
        alpha: float,
        horizon: int,
        epochs: int = 200,
        lr: float = 0.000001,
    ) -> None:
        nn.Module.__init__(self)  # type: ignore
        ConformalPredictor.__init__(self, score_fn, alpha, horizon)  # type: ignore
        self._epochs = epochs
        self._lr = lr
        no_offset_block_n_blocks = ceil(horizon / 2)
        offset_n_blocks = 1 + ceil((horizon - 1) / 2)
        base_no_offset_block_size = horizon // no_offset_block_n_blocks
        no_offset_block_sizes = [
            base_no_offset_block_size + 1
            if i < horizon % no_offset_block_n_blocks
            else base_no_offset_block_size
            for i in range(no_offset_block_n_blocks)
        ]
        base_offset_block_size = (horizon - 1) // (offset_n_blocks - 1)
        offset_block_sizes = [1] + [
            base_offset_block_size + 1
            if i < (horizon - 1) % (offset_n_blocks - 1)
            else base_offset_block_size
            for i in range(offset_n_blocks - 1)
        ]
        assert sum(no_offset_block_sizes) == horizon
        assert sum(offset_block_sizes) == horizon

        base_alpha_no_offset = alpha / 2
        base_alpha_offset = alpha / 2

        effective_alpha_no_offset = base_alpha_no_offset / (no_offset_block_n_blocks)
        effective_alpha_offset = base_alpha_offset / (offset_n_blocks)
        alphas_no_offset = nn.Parameter(
            torch.ones(no_offset_block_n_blocks) * effective_alpha_no_offset
        )
        alphas_offset = nn.Parameter(
            torch.ones(offset_n_blocks) * effective_alpha_offset
        )

        self._alphas_per_block_no_offset = nn.ParameterList(
            [
                nn.Parameter(torch.ones(no_offset_block_sizes[i]))
                * alphas_no_offset[i]
                / no_offset_block_sizes[i]
                for i in range(no_offset_block_n_blocks)
            ]
        )

        self._alphas_per_block_offset = nn.ParameterList(
            [
                nn.Parameter(torch.ones(offset_block_sizes[i]))
                * alphas_offset[i]
                / offset_block_sizes[i]
                for i in range(offset_n_blocks)
            ]
        )

    def predict(self, targets: P, predictions: P):
        scores = self._score_fn(targets, predictions).values
        scores = scores.squeeze(-1)

        def limit_score_for_sequence(
            sequence: torch.Tensor, alpha: float
        ) -> torch.Tensor:
            q = ConformalPredictor.get_q(alpha, len(sequence))
            if q is None:
                return torch.tensor(float("inf"))
            return torch.quantile(sequence, q)

        def limit_scores(
            alphas_per_block: nn.ParameterList,
        ):
            idx = 0
            all_limit_scores: list[torch.Tensor] = []
            for i in range(len(alphas_per_block)):
                limit_scores: List[torch.Tensor] = []
                limit_scores.append(torch.tensor(float("inf")))
                cal_points_mask = torch.ones(scores[:, idx].shape, dtype=torch.bool)

                for j in range(len(alphas_per_block[i])):
                    previous_limits = limit_scores[j]
                    previous_sequence = (
                        torch.ones(scores[:, 0].shape) * float("inf")
                        if j == 0
                        else scores[:, idx - 1]
                    )
                    previous_sequence_in_previous_limits = (
                        previous_sequence <= previous_limits
                    )

                    cal_points_mask = (
                        cal_points_mask & previous_sequence_in_previous_limits
                    )
                    current_scores = scores[:, idx]
                    current_selected_scores = current_scores[cal_points_mask]
                    current_limit_scores = limit_score_for_sequence(
                        current_selected_scores, alphas_per_block[i][j].item()
                    )

                    limit_scores.append(current_limit_scores)
                    idx += 1

                limit_scores_tensor = torch.stack(limit_scores)
                limit_scores_tensor = limit_scores_tensor[1:]
                all_limit_scores.append(limit_scores_tensor)
            return torch.cat(all_limit_scores)

        limit_scores_no_offset = limit_scores(self._alphas_per_block_no_offset)
        limit_scores_offset = limit_scores(self._alphas_per_block_offset)
        limit_scores_tensor = (
            torch.stack([limit_scores_no_offset, limit_scores_offset]).min(dim=0).values
        )
        self._n_computations += 2

        return (
            limit_scores_tensor,
            torch.cat([t for t in self._alphas_per_block_no_offset]),
            torch.cat([t for t in self._alphas_per_block_offset]),
        )

    def calibrate(self, targets: P, predictions: P):
        optimizer = torch.optim.SGD(self.parameters(), lr=self._lr)
        criterion = torch.nn.L1Loss()

        self.train()
        best_loss = float("inf")
        best_scores = None
        alphas_objective = self._alpha / 2

        def compute_alpha_loss(alphas: torch.Tensor, alpha_objective: float):
            prodd = torch.prod(1 - alphas)
            alpha_diff = torch.abs(prodd - (1 - torch.tensor(alpha_objective)))
            return torch.tan(alpha_diff * (torch.pi / 2) / 0.01)

        for epoch in range(self._epochs):
            optimizer.zero_grad()

            limit_scores, alphas_no_offset, alphas_offset = self.predict(
                targets, predictions
            )
            loss = limit_scores
            loss1 = criterion(limit_scores, torch.zeros_like(limit_scores))
            loss2 = compute_alpha_loss(alphas_no_offset, alphas_objective)
            loss3 = compute_alpha_loss(alphas_offset, alphas_objective)
            # regularization_loss = 20 * torch.abs(
            #     1 - torch.exp(torch.sum(alphas) - self._alpha)
            # )
            one_minus_alphas_sum = 1 - alphas_no_offset.sum().detach().item()
            one_mins_alphas = 1 - alphas_no_offset.detach()
            prod = torch.prod(one_mins_alphas)
            current_loss_1 = loss1.detach().item()
            # loss = loss1 + regularization_loss
            loss = loss1 + loss2 + loss3
            # loss = loss1
            current_loss = loss.detach().item()
            loss.backward()
            optimizer.step()
            if current_loss < best_loss:
                best_loss = current_loss
                best_scores = limit_scores.detach().clone()
            print(
                f"Epoch: {epoch}, current loss: {current_loss}, current loss 1: {
                    current_loss_1}, alphas_sum: {one_minus_alphas_sum}, prod: {prod.item()}"
            )
            # print(alphas)

        assert best_scores is not None, "Best scores should not be none at this stage"

        self._limit_scores = best_scores
        # self._alphas = torch.nn.Parameter(best_alphas)
        print(
            self._limit_scores,
            self._limit_scores.detach().mean(),
        )

    def get_params(self):
        params = super().get_params()
        additional_params = {"epochs": self._epochs, "lr": self._lr}
        return {**params, **additional_params}

    def get_name(self):
        if self._epochs == 1:
            return self.__class__.__name__
        else:
            return f"{self.__class__.__name__}Optim"

    def get_tunnable_params(self):
        return {
            **super().get_tunnable_params(),
            "alphas_no_offset": [t.tolist() for t in self._alphas_per_block_no_offset],
            "alphas_offset": [t.tolist() for t in self._alphas_per_block_offset],
        }

    def limit_scores(self, targets: P) -> ConformalScores:
        target_values = targets.values
        target_shape = target_values.shape
        assert (
            self._limit_scores is not None
        ), "Need to call calibrate before call limit_scores"
        return ConformalScores(
            self._limit_scores.repeat(target_values.shape[0]).view(
                target_shape[0], target_shape[1], 1
            )
        )
