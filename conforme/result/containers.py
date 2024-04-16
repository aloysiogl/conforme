from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

import torch


@dataclass
class Results:
    point_predictions: Any
    independent_coverage_indicators: Any
    joint_coverage_indicators: Any
    errors: Any
    mean_independent_coverage: torch.Tensor
    mean_joint_coverage: float
    confidence_zone_area: Any
    mean_confidence_zone_area_per_horizon: Any
    mean_confidence_zone_area: float
    min_confidence_zone_area: float
    max_confidence_zone_area: float
    self_cpu_time_total_cal: Optional[float] = None
    self_cpu_memory_usage_cal: Optional[float] = None
    self_cpu_time_total_test: Optional[float] = None
    self_cpu_memory_usage_test: Optional[float] = None
    n_threads: Optional[int] = None

    def set_performance_metrics(self, dict: Optional[Dict[str, int]]):
        if dict is None:
            return

        self.self_cpu_time_total_cal = dict["self_cpu_time_total_cal"]
        self.self_cpu_memory_usage_cal = dict["self_cpu_memory_usage_cal"]
        self.self_cpu_time_total_test = dict["self_cpu_time_total_test"]
        self.self_cpu_memory_usage_test = dict["self_cpu_memory_usage_test"]


class ResultsWrapper:
    def __init__(self):
        self._results: List[Results] = []

    def add_result(self, result: Results):
        self._results.append(result)

    def add_results(self, results: List[Results]):
        self._results += results

    def get_joint_coverages(self):
        coverages = [res.mean_joint_coverage for res in self._results]
        mean = np.mean(coverages)
        std = np.std(coverages)
        return mean.item(), std.item()

    def get_mean_area_per_horizon(self):
        mean_area = torch.stack(
            [res.mean_confidence_zone_area_per_horizon for res in self._results]
        )
        means = mean_area.mean(dim=0)
        stds = mean_area.std(dim=0)

        return means.tolist(), stds.tolist()

    def get_mean_area(self):
        mean_area = torch.tensor(
            [res.mean_confidence_zone_area for res in self._results]
        )
        mean = mean_area.mean()
        std = mean_area.std()

        return mean.item(), std.item()

    def get_min_area(self):
        min_area = torch.tensor([res.min_confidence_zone_area for res in self._results])
        min = min_area.min()
        stds = min_area.std()

        return min.item(), stds.item()

    def get_max_area(self):
        max_area = torch.tensor([res.max_confidence_zone_area for res in self._results])
        max = max_area.max()
        stds = max_area.std()

        return max.item(), stds.item()

    def get_mean_metric(self, get_metric):
        def filter_none(res):
            return res is not None

        metric_list = [get_metric(res) for res in self._results]

        if all([metric is None for metric in metric_list]):
            return None, None

        metric = torch.tensor(
            list(filter(filter_none, [get_metric(res) for res in self._results])),
            dtype=torch.float,
        )
        mean = metric.mean()
        std = metric.std()

        return mean.item()
    
    def get_std_metric(self, get_metric):
        def filter_none(res):
            return res is not None

        metric_list = [get_metric(res) for res in self._results]

        if all([metric is None for metric in metric_list]):
            return None, None

        metric = torch.tensor(
            list(filter(filter_none, [get_metric(res) for res in self._results])),
            dtype=torch.float,
        )
        std = metric.std()

        return std.item()

    def get_min_metric(self, get_metric):
        def filter_none(res):
            return res is not None

        metric_list = [get_metric(res) for res in self._results]

        if all([metric is None for metric in metric_list]):
            return None, None

        metric = torch.tensor(
            list(filter(filter_none, [get_metric(res) for res in self._results])),
            dtype=torch.float,
        )

        min = metric.min()

        return min.item()

    def get_max_metric(self, get_metric):
        def filter_none(res):
            return res is not None

        metric_list = [get_metric(res) for res in self._results]

        if all([metric is None for metric in metric_list]):
            return None, None

        metric = torch.tensor(
            list(filter(filter_none, [get_metric(res) for res in self._results])),
            dtype=torch.float,
        )

        max = metric.max()

        return max.item()
    
    def get_n_threads(self):
        n_threads = self._results[0].n_threads
        if not all([res.n_threads == n_threads for res in self._results]):
            raise ValueError("Not all results have the same n_threads")
        return n_threads

    # TODO technical debt very ugly and bad
    def get_dict(self):
        """Memory in bytes and time in us."""
        return {
            "coverage": self.get_joint_coverages()[0],
            "coverage_std": self.get_joint_coverages()[1],
            "mean_area": self.get_mean_area()[0],
            "mean_area_std": self.get_mean_area()[1],
            "min_area": self.get_min_area()[0],
            "min_area_std": self.get_min_area()[1],
            "max_area": self.get_max_area()[0],
            "max_area_std": self.get_max_area()[1],
            "mean_area_per_horizon": self.get_mean_area_per_horizon()[0],
            "mean_area_per_horizon_std": self.get_mean_area_per_horizon()[1],
            "mean_cpu_time_cal": self.get_mean_metric(
                lambda res: res.self_cpu_time_total_cal
            ),
            "std_cpu_time_cal": self.get_std_metric(
                lambda res: res.self_cpu_time_total_cal
            ),
            "mean_cpu_memory_cal": self.get_mean_metric(
                lambda res: res.self_cpu_memory_usage_cal
            ),
            "std_cpu_memory_cal": self.get_std_metric(
                lambda res: res.self_cpu_memory_usage_cal
            ),
            "mean_cpu_time_test": self.get_mean_metric(
                lambda res: res.self_cpu_time_total_test
            ),
            "std_cpu_time_test": self.get_std_metric(
                lambda res: res.self_cpu_time_total_test
            ),
            "mean_cpu_memory_test": self.get_mean_metric(
                lambda res: res.self_cpu_memory_usage_test
            ),
            "std_cpu_memory_test": self.get_std_metric(
                lambda res: res.self_cpu_memory_usage_test
            ),
            "max_cpu_time_cal": self.get_max_metric(
                lambda res: res.self_cpu_time_total_cal
            ),
            "max_cpu_memory_cal": self.get_max_metric(
                lambda res: res.self_cpu_memory_usage_cal
            ),
            "max_cpu_time_test": self.get_max_metric(
                lambda res: res.self_cpu_time_total_test
            ),
            "max_cpu_memory_test": self.get_max_metric(
                lambda res: res.self_cpu_memory_usage_test
            ),
            "min_cpu_time_cal": self.get_min_metric(
                lambda res: res.self_cpu_time_total_cal
            ),
            "min_cpu_memory_cal": self.get_min_metric(
                lambda res: res.self_cpu_memory_usage_cal
            ),
            "min_cpu_time_test": self.get_min_metric(
                lambda res: res.self_cpu_time_total_test
            ),
            "min_cpu_memory_test": self.get_min_metric(
                lambda res: res.self_cpu_memory_usage_test
            ),
            "n_threads": self._results[0].n_threads,
        }
