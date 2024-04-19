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

from conforme.conformal.predictor import (
    ConformalPredictorParams,
    ConForMEBinParams,
    ConForMEParams,
    get_cfrnn_maker,
    get_conforme_maker,
    get_conformebin_maker,
)
from conforme.conformal.score import distance_2d_conformal_score, l1_conformal_score
from conforme.experiments.run import (
    get_argoverse_runner,
    prepare_medical_runner,
    prepare_synthetic_runner,
)
from conforme.result.evaluation import evaluate_experiments_for_dataset

"""Parameter definitions for the conoformal predictors in each experiment """

synthetic_general_params = ConformalPredictorParams(
    alpha=0.1,
    horizon=10,
    score_fn=l1_conformal_score,
)

synthetic_runner = prepare_synthetic_runner(
    "static", 1, True, False, False, False, synthetic_general_params
)

synthetic_cp_makers = (
    [
        get_conforme_maker(
            ConForMEParams(
                general_params=synthetic_general_params,
                approximate_partition_size=s,
                epochs=e,
                lr=0.000001,
            )
        )
        for s in [1, 2, 3, 5, 10]
        for e in [1, 100]
    ]
    + [get_cfrnn_maker(synthetic_general_params)]
    + [
        get_conformebin_maker(
            ConForMEBinParams(
                general_params=synthetic_general_params,
                beta=0.5,
                optimize=True,
            )
        )
    ]
)


eeg10_general_params = ConformalPredictorParams(
    alpha=0.1,
    horizon=10,
    score_fn=l1_conformal_score,
)

eeg10_runner = prepare_medical_runner("eeg", True, False, eeg10_general_params)

eeg10_cp_makers = (
    [
        get_conforme_maker(
            ConForMEParams(
                general_params=eeg10_general_params,
                approximate_partition_size=s,
                epochs=e,
                lr=0.000001,
            )
        )
        for s in [1, 2, 3, 5, 10]
        for e in [1, 100]
    ]
    + [get_cfrnn_maker(eeg10_general_params)]
    + [
        get_conformebin_maker(
            ConForMEBinParams(
                general_params=eeg10_general_params,
                beta=0.5,
                optimize=True,
            )
        )
    ]
)

eeg40_general_params = ConformalPredictorParams(
    alpha=0.1,
    horizon=40,
    score_fn=l1_conformal_score,
)

eeg40_runner = prepare_medical_runner("eeg", True, False, eeg40_general_params)

eeg40_cp_makers = (
    [
        get_conforme_maker(
            ConForMEParams(
                general_params=eeg40_general_params,
                approximate_partition_size=s,
                epochs=e,
                lr=0.00000001,
            )
        )
        for s in [1, 2, 5, 10, 20, 40]
        for e in [1, 100]
    ]
    + [get_cfrnn_maker(eeg40_general_params)]
    + [
        get_conformebin_maker(
            ConForMEBinParams(
                general_params=eeg40_general_params,
                beta=0.5,
                optimize=True,
            )
        )
    ]
)

argoverse_general_params = ConformalPredictorParams(
    alpha=0.1,
    horizon=30,
    score_fn=distance_2d_conformal_score,
)

argoverse_cp_makers = (
    [
        get_conforme_maker(
            ConForMEParams(
                general_params=argoverse_general_params,
                approximate_partition_size=s,
                epochs=e,
                lr=0.00000001,
            )
        )
        for s in [1, 2, 3, 10, 30]
        for e in [1, 100]
    ]
    + [get_cfrnn_maker(argoverse_general_params)]
    + [
        get_conformebin_maker(
            ConForMEBinParams(
                general_params=argoverse_general_params,
                beta=0.5,
                optimize=True,
            )
        )
    ]
)


covid_general_params = ConformalPredictorParams(
    alpha=0.7,
    horizon=50,
    score_fn=l1_conformal_score,
)

covid_runner = prepare_medical_runner("covid", True, False, covid_general_params)

covid_cp_makers = (
    [
        get_conforme_maker(
            ConForMEParams(
                general_params=covid_general_params,
                approximate_partition_size=s,
                epochs=e,
                lr=0.00000001,
            )
        )
        for s in [1, 2, 5, 10, 25, 50]
        for e in [1]
    ]
    + [get_cfrnn_maker(covid_general_params)]
    + [
        get_conformebin_maker(
            ConForMEBinParams(
                general_params=covid_general_params,
                beta=0.5,
                optimize=True,
            )
        )
    ]
)


"""Running the experiments"""
# apparently type system does not check correctly here

profile = True

evaluate_experiments_for_dataset(
    "synthetic", profile, synthetic_general_params, synthetic_cp_makers, synthetic_runner
)


evaluate_experiments_for_dataset(
    "argoverse",
    profile,
    argoverse_general_params,
    argoverse_cp_makers,
    get_argoverse_runner,
)

evaluate_experiments_for_dataset(
    "eeg", profile, eeg10_general_params, eeg10_cp_makers, eeg10_runner
)

evaluate_experiments_for_dataset(
    "eeg", profile, eeg40_general_params, eeg40_cp_makers, eeg40_runner
)

evaluate_experiments_for_dataset(
    "covid", profile, covid_general_params, covid_cp_makers, covid_runner
)
