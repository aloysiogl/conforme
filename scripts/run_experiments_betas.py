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

import numpy as np

from conforme.conformal.predictor import (
    ConformalPredictorParams,
    ConForMEBinParams,
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

synthetic_cp_makers = [
    get_conformebin_maker(
        ConForMEBinParams(
            general_params=synthetic_general_params,
            beta=b,
            optimize=False,
        )
    )
    for b in np.arange(0.01, 0.99, 100)
]


eeg10_general_params = ConformalPredictorParams(
    alpha=0.1,
    horizon=10,
    score_fn=l1_conformal_score,
)

eeg10_runner = prepare_medical_runner("eeg", True, False, eeg10_general_params)

eeg10_cp_makers = [
    get_conformebin_maker(
        ConForMEBinParams(
            general_params=eeg10_general_params,
            beta=b,
            optimize=False,
        )
    )
    for b in np.arange(0.01, 0.99, 100)
]

eeg40_general_params = ConformalPredictorParams(
    alpha=0.1,
    horizon=40,
    score_fn=l1_conformal_score,
)

eeg40_runner = prepare_medical_runner("eeg", True, False, eeg40_general_params)

eeg40_cp_makers = [
    get_conformebin_maker(
        ConForMEBinParams(
            general_params=eeg40_general_params,
            beta=b,
            optimize=False,
        )
    )
    for b in np.arange(0.01, 0.99, 100)
]

argoverse_general_params = ConformalPredictorParams(
    alpha=0.1,
    horizon=30,
    score_fn=distance_2d_conformal_score,
)

argoverse_cp_makers = [
    get_conformebin_maker(
        ConForMEBinParams(
            general_params=argoverse_general_params,
            beta=b,
            optimize=False,
        )
    )
    for b in np.arange(0.01, 0.99, 100)
]

covid_general_params = ConformalPredictorParams(
    alpha=0.7,
    horizon=50,
    score_fn=l1_conformal_score,
)

covid_runner = prepare_medical_runner("covid", True, False, covid_general_params)

covid_cp_makers = [
    get_conformebin_maker(
        ConForMEBinParams(
            general_params=covid_general_params,
            beta=b,
            optimize=False,
        )
    )
    for b in np.arange(0.01, 0.99, 100)
]


"""Running the experiments"""
# apparently type system does not check correctly here

profile = False

evaluate_experiments_for_dataset(
    "synthetic",
    profile,
    synthetic_general_params,
    synthetic_cp_makers,
    synthetic_runner,
    "_betas",
)

evaluate_experiments_for_dataset(
    "argoverse",
    profile,
    argoverse_general_params,
    argoverse_cp_makers,
    get_argoverse_runner,
    "_betas",
)

evaluate_experiments_for_dataset(
    "eeg", profile, eeg10_general_params, eeg10_cp_makers, eeg10_runner, "_betas"
)

evaluate_experiments_for_dataset(
    "eeg", profile, eeg40_general_params, eeg40_cp_makers, eeg40_runner, "_betas"
)

evaluate_experiments_for_dataset(
    "covid", profile, covid_general_params, covid_cp_makers, covid_runner, "_betas"
)
