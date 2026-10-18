"""
SUBMISSION: 
***********
    test_series --prep_X--> (X, catalog) --->

        [DINOv3 ViT-L/16 checkpoint
        ref: https://github.com/ytrsk/OrthoFoundation
        (continued pretraining on 1.2M knee images)
        ├── patch embedding    <LoRA fine-tuned>
        ├── all 24 blocks      <LoRA fine-tuned>
        └── final norm         <fine-tuned>
        +
        SlotHead (learned attention pooling)]<ensemble model_ft_{A/B}.pt>

    --predict--> submission.csv

NOTE: This script assumes each input returns a non-null prediction (validated upstream).

"""

import logging, sys
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

INPUT = Path("/kaggle/input/")
WORKING = Path("/kaggle/working/")
PREP_X_OUT = INPUT / "datasets/ramitnandi/rsna-knee-prep-x5gr-out"  
TRAIN_OUT = INPUT / "notebooks/ramitnandi/rsna-knee-train-ortho"  

sys.path[:0] = [str(PREP_X_OUT), str(TRAIN_OUT)] 
# prioritize over any other module with the same name.
import prep_X
from train import TARGETS, build_ortho_model, load_state, predict

logger = logging.getLogger(__name__)
SUBMIT_RUNS = ['A', 'B'] 
N_SUBMIT = len(SUBMIT_RUNS)


def main() -> pd.DataFrame:
    ## -- X + catalog ----
    X, mask, mask_index = load_state(*prep_X.main("test"))
    N, S = X.shape[:2]
    ## ----

    def predict_run(run_name: str, dev: torch.device) -> np.ndarray:
        """
        - Load the DINOv3-L architecture with OrthoFoundation weights.
        - Overwrite fine-tuned layers using model_ft_{A/B}.pt; keep frozen layers unchanged.
        """
        if dev.type == "cuda":
            torch.cuda.set_device(dev) 

        model = build_ortho_model(n_slot=S).to(dev)
        fine_tuned = torch.load(TRAIN_OUT / f"model_ft_{run_name}.pt", map_location=dev, weights_only=True)
        _, unexpected = model.load_state_dict(fine_tuned, strict=False)
        assert not unexpected
        not_loaded = {name for name, param in model.named_parameters() if param.requires_grad} - set(fine_tuned)
        assert not not_loaded

        run_pred = predict(model, X, mask, np.arange(N), dev) # [[MAIN]]

        logger.info(f"run {run_name}: predicted {N} studies on {dev}")
        return run_pred

    ## -- (T4 x2: cuda:0 -> A | cuda:1 -> B; one model per GPU at a time) ----
    n_gpu = torch.cuda.device_count()
    devices = [torch.device(f"cuda:{i}") for i in range(n_gpu)] if n_gpu else [torch.device("cpu")]
    n_dev = len(devices)

    def predict_runs_on_device(device_index: int) -> list[np.ndarray]:
        return [
            predict_run(SUBMIT_RUNS[run_index], devices[device_index]) 
            for run_index in range(device_index, N_SUBMIT, n_dev)
        ]

    with ThreadPoolExecutor(max_workers=n_dev) as pool:
        run_preds = chain.from_iterable(
            pool.map(predict_runs_on_device, range(n_dev))
            # [[dev1_run1],[dev2_run1]] --> [dev1_run1,dev2_run1];
        )
    ## ---- 

    ## -- rank ensemble ----
    out = np.mean(
        [
            rankdata(run_pred, axis=0) 
            for run_pred in run_preds
        ], axis=0,
        # given our predicted probabilities are not calibrated,
        # the outputs from A/B may be on different scales.
        # hence, average rank is used for submission rather than average score.
        #
        # NOTE: this competition only rewards macro-AUC.
    )
    ## ----

    return pd.DataFrame(out/N, index=mask_index, columns=TARGETS) # [normalize output to the range 0–1]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    
    main().to_csv(WORKING / "submission.csv")
