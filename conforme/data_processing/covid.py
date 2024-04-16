# Copyright (c) 2021, Kamilė Stankevičiūtė
# Licensed under the BSD 3-clause license

import pickle
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from conformal_time_series.data_processing.dataset import Dataset1D

from .types import NumpyFloatArray, SplitNumpy

covid_root = "data/ltla_2021-05-24.csv"
# covid_root = "data/region_2023-12-14.csv"


def get_raw_covid_data(cached: bool = True):
    if cached:
        with open("data/covid.pkl", "rb") as f:
            dataset = pickle.load(f)
    else:
        dataset = []  # type: ignore
        df = pd.read_csv(covid_root)  # type: ignore
        for area_code in df["areaCode"].unique():  # type: ignore
            dataset.append(  # type: ignore
                df.loc[df["areaCode"] == area_code]  # pylint: disable=unsubscriptable-object
                .sort_values("date")["newCasesByPublishDate"]  # type: ignore
                .to_numpy()[-250:-100]  # type: ignore
            )
        dataset: NumpyFloatArray = np.array(dataset)
        with open("data/covid.pkl", "wb") as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    return dataset


def get_covid_splits(
    length: int = 100,
    horizon: int = 50,
    conformal: bool = True,
    n_train: int = 200,
    n_calibration: int = 100,
    n_test: int = 80,
    cached: bool = True,
    seed: Optional[int] = None,
):
    if seed is None:
        seed = 0
    else:
        cached = False

    train_dataset: Union[Dataset1D, SplitNumpy]
    calibration_dataset: Union[Dataset1D, Optional[SplitNumpy]]
    test_dataset: Union[Dataset1D, SplitNumpy]

    if cached:
        if conformal:
            with open("processed_data/covid_conformal.pkl", "rb") as f:
                train_dataset, calibration_dataset, test_dataset = pickle.load(f)
        else:
            with open("processed_data/covid_raw.pkl", "rb") as f:
                train_dataset, calibration_dataset, test_dataset = pickle.load(f)
    else:
        raw_data = get_raw_covid_data(cached=cached)
        X = raw_data[:, :length]
        Y = raw_data[:, length : length + horizon]

        perm = np.random.RandomState(seed=seed).permutation(
            n_train + n_calibration + n_test
        )
        train_idx = perm[:n_train]
        calibration_idx = perm[n_train : n_train + n_calibration]
        train_calibration_idx = perm[: n_train + n_calibration]
        test_idx = perm[n_train + n_calibration :]

        if conformal:
            X_train = X[train_idx]
            X_calibration = X[calibration_idx]
            X_test = X[test_idx]

            scaler = StandardScaler()
            X_train_scaled: NumpyFloatArray = scaler.fit_transform(X_train)  # type: ignore
            X_test_scaled: NumpyFloatArray = scaler.transform(X_test)  # type: ignore
            X_calibration_scaled: NumpyFloatArray = scaler.transform(X_calibration)  # type: ignore

            train_dataset = Dataset1D(
                torch.FloatTensor(X_train_scaled).reshape(-1, length, 1),
                torch.FloatTensor(Y[train_idx]).reshape(-1, horizon, 1),
                torch.ones(len(train_idx), dtype=torch.int) * length,
            )

            calibration_dataset = Dataset1D(
                torch.FloatTensor(X_calibration_scaled).reshape(-1, length, 1),
                torch.FloatTensor(Y[calibration_idx]).reshape(-1, horizon, 1),
                torch.ones(len(calibration_idx)) * length,
            )

            test_dataset = Dataset1D(
                torch.FloatTensor(X_test_scaled).reshape(-1, length, 1),
                torch.FloatTensor(Y[test_idx]).reshape(-1, horizon, 1),
                torch.ones(len(X_test_scaled), dtype=torch.int) * length,
            )

            with open("processed_data/covid_conformal.pkl", "wb") as f:
                pickle.dump(
                    (train_dataset, calibration_dataset, test_dataset),
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

        else:
            X_train = X[train_calibration_idx]
            X_test = X[test_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)  # type: ignore
            X_test_scaled = scaler.transform(X_test)  # type: ignore

            train_dataset = X_train_scaled, Y[train_calibration_idx]
            calibration_dataset = None
            test_dataset = X_test_scaled, Y[test_idx]

            with open("processed_data/covid_raw.pkl", "wb") as f:
                pickle.dump(
                    (train_dataset, calibration_dataset, test_dataset),
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

        with open("processed_data/covid_test_vis.pkl", "wb") as f:
            pickle.dump((X_test, Y[test_idx]), f, protocol=pickle.HIGHEST_PROTOCOL)

    return train_dataset, calibration_dataset, test_dataset
