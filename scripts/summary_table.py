import json
from functools import partial

from prettytable import PrettyTable

data_retrive_paths = {
    "model": ["params", "method"],
    "mean_block_size": ["params", "conformal_predictor_params", "mean_block_size"],
    "beta": ["result", "tunnable_params", "beta"],
    "mean_area": ["result", "outputs", "mean_area"],
    "mean_area_std": ["result", "outputs", "mean_area_std"],
    "min_area": ["result", "outputs", "min_area"],
    "min_area_std": ["result", "outputs", "min_area_std"],
    "max_area": ["result", "outputs", "max_area"],
    "max_area_std": ["result", "outputs", "max_area_std"],
    "coverage": ["result", "outputs", "coverage"],
    "coverage_std": ["result", "outputs", "coverage_std"],
    "n_computations": ["result", "tunnable_params", "n_computations"],
    "cpu_time_cal": ["result", "outputs", "mean_cpu_time_cal"],
    "cpu_time_cal_std": ["result", "outputs", "std_cpu_time_cal"],
    "min_cpu_time_cal": ["result", "outputs", "min_cpu_time_cal"],
    "max_cpu_time_cal": ["result", "outputs", "max_cpu_time_cal"],
    "cpu_memory_cal": ["result", "outputs", "mean_cpu_memory_cal"],
    "cpu_memory_cal_std": ["result", "outputs", "std_cpu_memory_cal"],
    "min_cpu_memory_cal": ["result", "outputs", "min_cpu_memory_cal"],
    "max_cpu_memory_cal": ["result", "outputs", "max_cpu_memory_cal"],
    "cpu_time_test": ["result", "outputs", "mean_cpu_time_test"],
    "cpu_time_test_std": ["result", "outputs", "std_cpu_time_test"],
    "min_cpu_time_test": ["result", "outputs", "min_cpu_time_test"],
    "max_cpu_time_test": ["result", "outputs", "max_cpu_time_test"],
    "cpu_memory_test": ["result", "outputs", "mean_cpu_memory_test"],
    "cpu_memory_test_std": ["result", "outputs", "std_cpu_memory_test"],
    "min_cpu_memory_test": ["result", "outputs", "min_cpu_memory_test"],
    "max_cpu_memory_test": ["result", "outputs", "max_cpu_memory_test"],
}

headers = [
    "Model",
    "Coverage",
    "Mean Area",
    "Improvement",
    "Min Area",
    "Max Area",
    "N Evals",
    "Beta",
    "Mean Block Size",
    # "Cpu Time Cal",
    # "Min Cpu Time Cal",
    # "Max Cpu Time Cal",
    # "cpu memory cal",
    # "min cpu memory cal",
    # "max cpu memory cal",
    # "cpu time test",
    # "Min Cpu Time Test",
    # "Max Cpu Time Test",
    # "Cpu Memory Test",
    # "Min Cpu Memory Test",
    # "Max Cpu Memory Test"
]


def get_align(x):
    if x in ["Model", "Improvement"]:
        return "l"
    return "c"


def get_row(get):
    return {
        "Model": get("model"),
        "Beta": format_value("beta", get),
        "Mean Area": format_value("mean_area", get, True),
        "Min Area": format_value("min_area", get, True),
        "Max Area": format_value("max_area", get, True),
        "Coverage": format_value("coverage", get, True),
        "N Evals": get("n_computations"),
        "Cpu Time Cal": format_time("cpu_time_cal", get, True),
        "Min Cpu Time Cal": format_time("min_cpu_time_cal", get),
        "Max Cpu Time Cal": format_time("max_cpu_time_cal", get),
        "Cpu Memory Cal": format_memory("cpu_memory_cal", get, True),
        "Min Cpu Memory Cal": format_memory("min_cpu_memory_cal", get),
        "Max Cpu Memory Cal": format_memory("max_cpu_memory_cal", get),
        "Cpu Time Test": format_time("cpu_time_test", get, True),
        "Min Cpu Time Test": format_time("min_cpu_time_test", get),
        "Max Cpu Time Test": format_time("max_cpu_time_test", get),
        "Cpu Memory Test": format_memory("cpu_memory_test", get, True),
        "Min Cpu Memory Test": format_memory("min_cpu_memory_test", get),
        "Max Cpu Memory Test": format_memory("max_cpu_memory_test", get),
        "Mean Block Size": format_value("mean_block_size", get),
    }


def safe_get_row(data_dict, paths_dict, value):
    path_list = paths_dict[value]
    for key in path_list:
        if not key in data_dict:
            return None
        data_dict = data_dict[key]
    return data_dict


def format_value(field, get, std=False):
    if get(field) is None:
        return "-"
    if std is False:
        return f"{get(field):.2f}"
    std_field = f"{field}_std"
    return f"{get(field):.3f} ± {get(std_field):.3f}"


def wrap_division(denominator, get):
    def division(field):
        return get(field) / denominator

    return division


def format_memory(field, get, std=False):
    wrapped_get = wrap_division(1024**2, get)
    return f"{format_value(field, wrapped_get, std)} MB"


def format_time(field, get, std=False):
    wrapped_get = wrap_division(1e3, get)
    return f"{format_value(field, wrapped_get, std)} ms"


def listmap(*args, **kwargs):
    return list(map(*args, **kwargs))


def get_baseline_mean_area(paths):
    for model in paths:
        if safe_get_row(model, data_retrive_paths, "model") == "CFRNN":
            return safe_get_row(model, data_retrive_paths, "mean_area")


def main():
    database_path = "./results/synthetic_profile_horizon10.json"

    with open(database_path, "r") as f:
        data = json.load(f)

    table = PrettyTable(headers)
    baseline_mean_area = get_baseline_mean_area(data)

    models = []
    for model in data:
        get = partial(safe_get_row, model, data_retrive_paths)
        row = get_row(get)

        def add_improvement(row):
            improvement = (
                (baseline_mean_area - get("mean_area")) / baseline_mean_area * 100
            )
            row["Key"] = improvement
            if row["Model"] == "CFRNN":
                row["Improvement"] = "baseline"
                return row
            row["Improvement"] = f"{improvement:.1f}%"
            return row

        row = add_improvement(row)
        models.append(row)

    models = sorted(models, key=lambda x: x["Key"])

    for model in models:

        def safe_get_row_value(key):
            if key in model:
                return model[key]
            return "-"

        table.add_row(listmap(safe_get_row_value, headers))

    def align(x):
        table.align[x] = get_align(x)

    listmap(align, headers)
    print(table.get_string())


if __name__ == "__main__":
    main()
