import json
import re
from functools import partial
from typing import Any

import click
import matplotlib.pyplot as plt


def safe_get(dict: dict[str, Any], path_list: list[str]) -> Any:
    for key in path_list:
        if key not in dict:
            return None
        dict = dict[key]
    return dict


@click.command()
@click.option(
    "--setting",
    "-s",
    "setting",
    type=click.Choice(["eeg_all", "eeg_bin", "argoverse_all"]),
    required=True,
)
def main(setting: str):
    path_from_settings = {
        "eeg_all": "./results/eeg_profile_horizon40.json",
        "eeg_bin": "./results/eeg_profile_horizon40.json",
        "argoverse_all": "./results/argoverse_profile_horizon30.json",
    }
    database_path = path_from_settings[setting]

    plt.style.use("seaborn-v0_8-bright")

    with open(database_path, "r") as f:
        data = json.load(f)

    def get_matcher(setting: str):
        if setting == "eeg_all" or setting == "argoverse_all":

            def matches(name: str):
                matches = re.match(r"CFRNN", name) is not None
                matches = matches or re.match(r"ConForME(\d+)$", name) is not None
                return matches

            return matches
        elif setting == "eeg_bin":

            def matches(name: str):
                matches = re.match(r"ConForMEBinOptim$", name) is not None
                matches = matches or re.match(r"ConForME20$", name) is not None
                return matches

            return matches
        else:
            raise ValueError(f"Unknown setting {setting}")

    def method_name_map(name: str):
        if re.match(r"CFRNN", name):
            return "CFRNN"
        conforme_match = re.match(r"ConForME(\d+)", name)
        if conforme_match:
            number = int(conforme_match.group(1))
            return f"$ConForME_{{{number}}}$"
        if re.match(r"ConForMEBinOptim", name):
            return "ConForME (best beta)"
        raise ValueError(f"Unknown method {name}")

    matcher = get_matcher(setting)

    _, ax = plt.subplots()  # type: ignore

    for result in data:
        get = partial(safe_get, result)
        method = get(["params", "method"])
        if matcher(method):
            series: list[float] = get(["result", "outputs", "mean_area_per_horizon"])
            stds: list[float] = get(["result", "outputs", "mean_area_per_horizon_std"])
            indexes = list(range(1, len(series) + 1))
            ax.plot(indexes, series, label=method_name_map(method))
            ax.fill_between(
                indexes,
                [a - b for a, b in zip(series, stds)],
                [a + b for a, b in zip(series, stds)],
                alpha=0.2,
            )

    plt.legend()  # type: ignore
    plt.rcParams.update(
        {
            "font.family": "serif",  # use serif/main font for text elements
            "text.usetex": True,  # use inline math for ticks
            "pgf.rcfonts": False,  # don't setup fonts from rc parameters
        }
    )
    plt.savefig(f"results/{setting}.png")  # type: ignore


if __name__ == "__main__":
    main()
