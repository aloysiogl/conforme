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

from .containers import ResultsWrapper


def performance_string_per_horizon(results: ResultsWrapper):
    results_horizon = results.get_mean_area_per_horizon()[0]
    results_horizon_std = results.get_mean_area_per_horizon()[1]
    acc = ""
    for i in range(len(results_horizon)):
        acc += f" & {results_horizon[i]:.1f} \\pm {results_horizon_std[i]:.1f}\n"
    return acc


def performance_summary_string(results: ResultsWrapper):
    coverage = results.get_joint_coverages()[0] * 100
    coverage_std = results.get_joint_coverages()[1] * 100
    mean_interval_widths = results.get_mean_area()[0]
    mean_interval_widths_std = results.get_mean_area()[1]
    min_interval_widths = results.get_min_area()[0]
    min_interval_widths_std = results.get_min_area()[1]
    max_interval_widths = results.get_max_area()[0]
    max_interval_widths_std = results.get_max_area()[1]
    return (
        f"Coverage: {coverage:.1f} \\pm {coverage_std:.1f}\n"
        f"Mean Inteval width: {mean_interval_widths:.1f} \\pm {mean_interval_widths_std:.1f}\n"
        f"Min Interval width: {min_interval_widths:.1f} \\pm {min_interval_widths_std:.1f}\n"
        f"Max Interval width: {max_interval_widths:.1f} \\pm {max_interval_widths_std:.1f}\n"
    )
