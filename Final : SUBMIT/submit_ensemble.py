"""
SUBMISSION (ensemble): 
**********************
    test_series --prep_X--> (X, catalog) --->

        [OrthoFoundation ViT-L/16, DoRA on all 24 blocks + SlotHead]<model_ft_{A/B/C/D}.pt>   --rank mean--> ortho
        [DINOv3 ViT-B/16, DoRA on all 12 blocks + SlotHead]<model_ft_{A/B/C/D}.pt>            --rank mean--> dino

    --(ortho + dino) / 2--> submission.csv

Each backbone is read, rebuilt, and predicted with the train code of its own kernel.

NOTE: This script assumes each input returns a non-null prediction (validated upstream).

"""

import importlib.util, logging, sys
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
ORTHO_TRAIN_OUT_AB = INPUT / "notebooks/ramitnandi/rsna-knee-train-ortho-final-ab"  
ORTHO_TRAIN_OUT_CD = INPUT / "notebooks/ramitnandi/rsna-knee-train-ortho-final-cd"  
DINO_TRAIN_OUT = INPUT / "notebooks/ramitnandi/rsna-knee-train-dino-final"  

sys.path[:0] = [str(PREP_X_OUT)] 
import prep_X

spec = importlib.util.spec_from_file_location("train_ortho", ORTHO_TRAIN_OUT_AB / "train.py")
train_ortho = importlib.util.module_from_spec(spec)
sys.modules["train_ortho"] = train_ortho
spec.loader.exec_module(train_ortho)

spec = importlib.util.spec_from_file_location("train_dino", DINO_TRAIN_OUT / "train.py")
train_dino = importlib.util.module_from_spec(spec)
sys.modules["train_dino"] = train_dino
spec.loader.exec_module(train_dino)

logger = logging.getLogger(__name__)


def rank_mean(train, checkpoints: list[Path], test_X: np.ndarray, test_catalog: pd.DataFrame) -> pd.DataFrame:
    """    
    One backbone: run each model using its own training code (2× T4, one model per GPU at a time), 
    then return the average rank across runs.
    """
    X, mask, mask_index = train.load_state(test_X, test_catalog)
    N, S = X.shape[:2]

    def predict_run(checkpoint: Path, dev: torch.device) -> np.ndarray:
        if dev.type == "cuda":
            torch.cuda.set_device(dev) 
        model = train.build_ortho_model(n_slot=S).to(dev)
        fine_tuned = torch.load(checkpoint, map_location=dev, weights_only=True)
        _, unexpected = model.load_state_dict(fine_tuned, strict=False)
        assert not unexpected
        not_loaded = {name for name, param in model.named_parameters() if param.requires_grad} - set(fine_tuned)
        assert not not_loaded
        run_pred = train.predict(model, X, mask, np.arange(N), dev) # [[MAIN]]
        logger.info(f"{checkpoint.parent.name}/{checkpoint.name}: predicted {N} studies on {dev}")
        return run_pred

    n_gpu = torch.cuda.device_count()
    devices = [torch.device(f"cuda:{i}") for i in range(n_gpu)] if n_gpu else [torch.device("cpu")]
    n_dev = len(devices)

    def predict_runs_on_device(device_index: int) -> list[np.ndarray]:
        return [
            predict_run(checkpoints[run_index], devices[device_index]) 
            for run_index in range(device_index, len(checkpoints), n_dev)
        ]

    with ThreadPoolExecutor(max_workers=n_dev) as pool:
        run_preds = list(chain.from_iterable(pool.map(predict_runs_on_device, range(n_dev))))
    out = np.mean([rankdata(run_pred, axis=0) for run_pred in run_preds], axis=0)
    return pd.DataFrame(out, index=mask_index, columns=train.TARGETS)


def main() -> pd.DataFrame:
    """
    - Shared preprocessing.
    - Equal weight per backbone.
    - Average ranks, as predicted probabilities are not calibrated.
    
    NOTE: this competition only rewards macro-AUC.
    """
    test_X, test_catalog = prep_X.main("test")
    ortho = rank_mean(
        train_ortho, 
        [
            ORTHO_TRAIN_OUT_AB / "model_ft_A.pt", ORTHO_TRAIN_OUT_AB / "model_ft_B.pt",
            ORTHO_TRAIN_OUT_CD / "model_ft_C.pt", ORTHO_TRAIN_OUT_CD / "model_ft_D.pt",
        ], 
        test_X, test_catalog,
    )
    dino = rank_mean(
        train_dino, 
        [
            DINO_TRAIN_OUT / "model_ft_A.pt", DINO_TRAIN_OUT / "model_ft_B.pt",
            DINO_TRAIN_OUT / "model_ft_C.pt", DINO_TRAIN_OUT / "model_ft_D.pt",
        ], 
        test_X, test_catalog,
    )
    out = (ortho + dino) / 2
    assert out.shape == ortho.shape == dino.shape
    # pandas aligns on (StudyInstanceUID, target); any label mismatch would grow the shape.
    return out / len(out) ## [normalize output to the range 0–1]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    main().to_csv(WORKING / "submission.csv")
