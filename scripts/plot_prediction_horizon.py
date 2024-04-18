import json
from functools import partial
from typing import Any

import matplotlib.pyplot as plt


def safe_get(dict: dict[str, Any], path_list: list[str]) -> Any:
    for key in path_list:
        if key not in dict:
            return None
        dict = dict[key]
    return dict


def main():
    database_path = "./results/argoverse_profile_horizon30.json"
    plt.style.use("seaborn-bright")

    with open(database_path, "r") as f:
        data = json.load(f)

    # methods = [
    #     "CFRNN",
    #     "ConForME1",
    #     "ConForME2",
    #     # "ConForMEBinOptim",
    #     "ConForME5",
    #     "ConForME10",
    #     "ConForME20",
    #     "ConForME40",
    # ]
    methods = [
        "CFRNN",
        "ConForME1",
        "ConForME2",
        # "ConForMEBinOptim",
        "ConForME3",
        "ConForME10",
        "ConForME30",
    ]

    # name_map = {
    #     "CFRNN": "CFRNN",
    #     "ConForME1": "$ConForME_{40}$",
    #     "ConForME2": "$ConForME_{20}$",
    #     # "ConForMEBinOptim": "ConForME (BinOptim)",
    #     "ConForME5": "$ConForME_{8}$",
    #     "ConForME10": "$ConForME_{4}$",
    #     "ConForME20": "$ConForME_{2}$",
    #     "ConForME40": "$ConForME_{1}$",
    # }
    name_map = {
        "CFRNN": "CFRNN",
        "ConForME1": "$ConForME_{30}$",
        "ConForME2": "$ConForME_{15}$",
        "ConForME3": "$ConForME_{10}$",
        "ConForME10": "$ConForME_{3}$",
        "ConForME30": "$ConForME_{1}$",
        # "ConForMEBinOptim": "ConForME (BinOptim)",
    }

    _, ax = plt.subplots()  # type: ignore

    for result in data:
        get = partial(safe_get, result)
        method = get(["params", "method"])
        if method in methods:
            series: list[float] = get(["result", "outputs", "mean_area_per_horizon"])
            stds: list[float] = get(["result", "outputs", "mean_area_per_horizon_std"])
            indexes = list(range(1, len(series) + 1))
            # plot with standard deviation
            ax.plot(indexes, series, label=name_map[method])
            ax.fill_between(
                indexes,
                [a - b for a, b in zip(series, stds)],
                [a + b for a, b in zip(series, stds)],
                alpha=0.2,
            )

    # ax.spines["right"].set_visible(False)
    # ax.spines["left"].set_visible(False)
    # ax.spines["top"].set_visible(False)

    plt.legend() # type: ignore
    plt.rcParams.update(
        {
            "font.family": "serif",  # use serif/main font for text elements
            "text.usetex": True,  # use inline math for ticks
            "pgf.rcfonts": False,  # don't setup fonts from rc parameters
        }
    )
    plt.savefig("fig/results.pgf", backend="pgf")  # type: ignore
    plt.savefig("fig/results.pdf", backend="pgf")  # type: ignore


if __name__ == "__main__":
    main()
