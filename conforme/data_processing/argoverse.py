import pickle
import torch
from typing import Tuple
from conformal_time_series.conformal.predictions import Targets2D

dataset_path = "data/argoverse_lanegcn.pkl"


def get_calibration_test() -> Tuple[Targets2D, Targets2D, Targets2D, Targets2D]:
    with open(dataset_path, "rb") as f:
        data_dict = pickle.load(f)

    preds = data_dict["preds"]
    ground_truths = data_dict["gts"]
    ground_truths_accum = []
    preds_accum = []

    for idx in preds:
        # check of nan in preds_idx
        preds_idx = torch.tensor(preds[idx])
        if torch.isnan(preds_idx).any():
            continue
        ground_truths_accum.append(ground_truths[idx])
        preds_accum.append(torch.tensor(preds[idx]))
    preds_accum = torch.stack(preds_accum)
    split_index = int(0.5 * len(preds_accum))
    ground_truths_accum = torch.stack(ground_truths_accum)
    preds_accum = preds_accum.mean(dim=1)

    perm = torch.randperm(len(preds_accum))
    preds_accum = preds_accum[perm]
    ground_truths_accum = ground_truths_accum[perm]
    cal_preds = preds_accum[:split_index]
    cal_gts = ground_truths_accum[:split_index]
    test_preds = preds_accum[split_index:]
    test_gts = ground_truths_accum[split_index:]

    return (
        Targets2D(cal_preds),
        Targets2D(cal_gts),
        Targets2D(test_preds),
        Targets2D(test_gts),
    )
