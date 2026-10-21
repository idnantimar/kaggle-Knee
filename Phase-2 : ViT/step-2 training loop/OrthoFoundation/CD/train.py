"""
CONTINUED TRAINING:
*******************
Continues the A/B kernel with its exact training code (the script it saved)
by calling its `train_run` with run indices 2, 3.

OUTPUT: `model_ft_{C/D}.pt`

"""

import shutil, sys, time
from pathlib import Path

import torch

TRAIN_OUT_AB = Path("/kaggle/input/notebooks/ramitnandi/rsna-knee-train-ortho-final-ab")
LOCAL_PREP_X_OUT = Path("/tmp/rsna-knee-prep-x5gr-out") # local disk: fast I/O.
RUNS_NEW = {
    "C": (0, 1, 3, 4), "D": (0, 1, 2, 4),
}
FINISHED = 2 # A/B

sys.path.insert(0, str(TRAIN_OUT_AB)) # prioritize over any other module with the same name.
import train


def train_runs_on_gpu(gpu_index: int, n_gpu: int):
    """
    T4 x2: cuda:0 -> C;  cuda:1 -> D.
    """
    train.RUNS = RUNS_NEW
    train.PREP_X_OUT = LOCAL_PREP_X_OUT

    T0 = time.time()
    for run_index in range(gpu_index, len(RUNS_NEW), n_gpu):
        train.train_run(run_index + FINISHED, gpu_index, T0)
        # run index 2, 3 --> run name C, D.


if __name__ == "__main__":
    shutil.copytree(train.PREP_X_OUT, LOCAL_PREP_X_OUT)
    n_gpu = torch.cuda.device_count()
    assert n_gpu >= 1, "need at least 1 GPU."
    torch.multiprocessing.spawn(train_runs_on_gpu, args=(n_gpu,), nprocs=min(n_gpu, len(RUNS_NEW)), join=True)
