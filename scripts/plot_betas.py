import json
from functools import partial
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt


def safe_get(dict: dict[str, Any], path_list: list[str]) -> Optional[float]:
    for key in path_list:
        if key not in dict:
            return None
        dict = dict[key]
    if isinstance(dict, float):
        return dict
    raise ValueError(f"Expected number, got {dict}")


def add_betas_to_plot(path: str, method_name: str, ax: Any):
    with open(path, "r") as f:
        data = json.load(f)

    betas: list[float] = []
    widths: list[float] = []
    global idxc

    for result in data:
        get = partial(safe_get, result)
        beta = get(["params", "conformal_predictor_params", "beta"])
        width = get(["result", "outputs", "mean_area"])
        assert beta is not None and width is not None
        betas.append(beta)
        widths.append(width)
    idx = np.argsort(betas)
    betas_array = np.array(betas)[idx]
    widths_arr: npt.NDArray[np.float_] = np.array(widths)[idx]
    widths_arr = widths_arr - np.min(widths_arr)
    widths_arr /= np.max(widths_arr[widths_arr != np.inf]) - np.min(widths_arr)
    widths_arr[widths_arr == np.inf] = 1

    ax.plot(betas_array, widths_arr, label=method_name)


def main():
    print(plt.style.available)
    plt.style.use("seaborn-v0_8-bright")
    _, ax = plt.subplots()  # type: ignore
    add_betas_to_plot("./results/eeg_horizon10_betas.json", "$EEG_{10}$", ax)
    add_betas_to_plot("./results/eeg_horizon40_betas.json", "$EEG_{40}$", ax)
    add_betas_to_plot("./results/covid_horizon50_betas.json", "$COVID$", ax)
    add_betas_to_plot("./results/argoverse_horizon30_betas.json", "$Argoverse$", ax)
    add_betas_to_plot("./results/synthetic_horizon10_betas.json", "$Synthetic$", ax)

    plt.legend(loc="lower left")  # type: ignore
    plt.savefig("results/betas.png")  # type: ignore


if __name__ == "__main__":
    main()
