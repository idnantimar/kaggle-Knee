"""
(ViT backbone + MLP head) fine-tuned on: X = MRI DICOM images, y = LLM-extracted labels.

Inputs (outputs of prep_X.py / prep_y.py kernel run):
*****************************************************
`train_X.npy` -- [n_study, n_slot, n_slice, Height, Width]
`train_catalog.parquet` -- (sl | StudyInstanceUID | slot | SeriesInstanceUID | file_name)
`train_y.parquet` -- (StudyInstanceUID | <target> | <ref> | <weight> | <hard labels>)

Outputs:
********
    `model_ft_{A/B}.pt` -- state_dict corresponding to the best epoch.
    `train.py` -- copy of this script, which the final notebook can import.

"""

from __future__ import annotations

import gc, hashlib, logging, shutil, time, os
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)

SEED = 2026
INPUT = Path("/kaggle/input/")
WORKING = Path("/kaggle/working/")
DATA_ROOT = INPUT / "competitions/rsna-knee-abnormality-detection"
PREP_X_OUT, PREP_Y_OUT = INPUT / "datasets/ramitnandi/rsna-knee-prep-x5gr-out", INPUT / "datasets/ramitnandi/rsna-knee-prep-y-out"
LOCAL_PREP_X_OUT = Path("/tmp/rsna-knee-prep-x5gr-out") # local disk: fast I/O.

ORTHO = INPUT / "models/ramitnandi/orthofoundation/pytorch/default/1/orthofoundation" # +3 percentage points score gain over generic DINOv3 ViT-L/16.

 
TARGETS = [
    "ACL", "MCL",
    "Medial_Meniscus", "Lateral_Meniscus",
    "Medial_OA", "Lateral_OA", "PF_OA",
    "Effusion", "Synovitis",
    "Bakers",
    "Contusion", "Fracture",
] # prep_y.py columns.

# MRI slices are grayscale, while ViT expects R-G-B channels per input image;
# we stack 3 adjacent MRI slices as a single encoder pass.(2.5D: depth in the channels).
RGB = 3

TIME_BUDGET = 10 * 3600.  # sufficient buffer (compared to the Kaggle session limit of 12h).


# ==================================================
# Data
# ====================================
def load_state(X: np.ndarray, catalog: pd.DataFrame) -> tuple:
    """
    Load state from prep_X kernel runs.

    - X --- [n_study, n_slot, N_SLICE, IMG, IMG]
    - catalog --- (sl, StudyInstanceUID, slot, SeriesInstanceUID, file_name)
    X and catalog have row-by-row correspondence.
    """
    n_study, n_slot = X.shape[:2]
    mask = catalog["file_name"].notna().to_numpy().reshape(n_study, n_slot, -1).all(axis=-1)
    # only slots with all slices successfully decoded are considered downstream.
    mask_index = pd.Index(
        catalog["StudyInstanceUID"].to_numpy().reshape(n_study, n_slot, -1)[:, 0, 0],
        # each StudyInstanceUID spans a fixed block of n_slot * N_SLICE rows; 
        # take the first row of each block.
        name="StudyInstanceUID",
    )
    return X, mask, mask_index
 

AUG_ROTATION_DEG = 8.0
AUG_SCALE = 0.08
AUG_SHIFT = 0.05
AUG_INTENSITY = 0.10

def augment(imgs: torch.Tensor) -> torch.Tensor:
    """
    Data augmentation to reduce overfitting:
    - Add small affine transformations.
    - Apply intensity scaling per slot (preserve the relative intensity of R-G-B).
    - No flipping (left and right knees are mirror images of each other).
    """
    lead, unit_dim = imgs.shape[:-3], imgs.shape[-3:]
    x = imgs.reshape(-1, *unit_dim).float() # [batch_len * n_slots, R-G-B, Height, Width]
    n, dev = x.shape[0], x.device

    ## random rotation in [-AUG_ROT°, +AUG_ROT°);
    rot = (torch.rand(n, device=dev) - 0.5) * (AUG_ROTATION_DEG * np.pi / 90)
    ## random zoom-in up to (1 + AUG_SCALE) × 100%;
    sc = 1.0 + torch.rand(n, device=dev) * AUG_SCALE  
    ## random horizontal & vertical translation;
    tx = (torch.rand(n, device=dev) - 0.5) * 2 * AUG_SHIFT
    ty = (torch.rand(n, device=dev) - 0.5) * 2 * AUG_SHIFT
    ## random intensity scaling per R-G-B stack;
    scale = 1.0 + (torch.rand(n, 1, 1, 1, device=dev) - 0.5) * 2 * AUG_INTENSITY

    cos, sin = torch.cos(rot) / sc, torch.sin(rot) / sc
    theta = torch.stack(
        [
            torch.stack([cos, -sin, tx], dim=1),
            torch.stack([sin,  cos, ty], dim=1),
        ], dim=1,
        # inverse affine map: output coordinates --> input coordinates.
    )
    grid = F.affine_grid(
        theta, x.shape, 
        align_corners=False,
        # for a given (x_out, y_out),
        # returns the source (x_in, y_in) = (x_out*cos - y_out*sin + tx, x_out*sin + y_out*cos + ty).
    )
    x = F.grid_sample(
        x, grid, 
        mode="bilinear", padding_mode="border", align_corners=False,
    )
    x = (x * scale).clamp(0, 255)
    return x.reshape(*lead, *unit_dim)

# -------------------------

 
# ==================================================
# Model
# ====================================

from peft import LoraConfig, inject_adapter_in_model
from peft.tuners.lora import LoraLayer
import peft.tuners.lora.torchao as peft_torchao; peft_torchao.is_torchao_available = lambda: False 

LORA_RANK = 8
def _LoRA(layers: list):
    # ref- https://arxiv.org/abs/2402.09353
    return LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_RANK * 2, 
        target_modules=layers, bias="none",
        lora_dropout=0.1,
        use_dora=True,
    )


def build_ortho_model(n_slot: int, checkpoint_path: Path = ORTHO / "OrthoFoundation.pt") -> _Model:
    """
    DINOv3 ViT-L/16 checkpoint
    ref: https://github.com/ytrsk/OrthoFoundation
    (with continued pretraining on 1.2M knee images)
        ├── patch embedding    <trainable DoRA adapter>
        ├── all 24 blocks      <trainable DoRA adapters>
        └── final norm         <trainable>

        +
    SlotHead (attention pooling)
    
    """
    ViT = torch.hub.load(
        os.fspath(ORTHO), 
        "dinov3_vitl16", 
        source="local", pretrained=False,
        # architecture only; no weight download.
    ) 
    checkpoint = torch.load(
        checkpoint_path, 
        map_location="cpu", weights_only=True,
        # plain dict of tensors.
    )  
    # NOTE: different training frameworks may save weights
    # under different top-level keys (e.g., "teacher", "model", "state_dict", etc.);
    # there may also be different prefixes (e.g., "module.", "backbone.", etc.);
    # since this is a local file, we have manually inspected its structure beforehand:
    #   "backbone.<DINOv3 parameter name>" -> tensor
    state_dict = {
        k.removeprefix("backbone."): v 
        for k, v in checkpoint.items()
    }

    ViT.load_state_dict(
        state_dict, strict=True
        # STRICT:
        # the checkpoint tensors must match the ViT-L/16 architecture exactly by name and shape.
    )  
    ViT.requires_grad_(False)
    ViT.norm.requires_grad_(True)
    # base weights and bias are frozen;
    # final norm is trainable.

    inject_adapter_in_model(
        _LoRA(["proj"]), ViT.patch_embed,
        # we use 3 adjacent grayscale MRI slices as a pseudo-RGB stack,
        # hence tune the patch embeddings to adapt from usual RGB image behaviour.
    )
    LoRA_config = _LoRA(["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"])
    for block in ViT.blocks:
        with torch.no_grad():
            block.attn.qkv.bias.mul_(block.attn.qkv.bias_mask)
            # DINOv3 uses shared QKV and masks K-bias on-the-fly during the forward pass,
            # but PEFT DoRA uses the stored, unmasked bias during weight recomputation;
            # apply the mask in place before injecting DoRA so both agree.
            # (the forward output remains unchanged)
        inject_adapter_in_model(LoRA_config, block)

    ViT.eval()
    n_ft = sum(p.numel() for p in ViT.parameters() if p.requires_grad) / 1e6
    logger.info(
        f"""
        backbone: OrthoFoundation-L,
        DoRA (rank {LORA_RANK}) on patch embedding + all {len(ViT.blocks)} blocks ({n_ft:.2f}M trainable params)
        """
    )
    return _Model(ViT, ViT.embed_dim, n_slot)

 
class _Model(nn.Module):
    """ViT backbone + MLP head"""
 
    def __init__(self, backbone, dim, n_slot):
        super().__init__()
        self.backbone = backbone
        self.head = _SlotHead(
            dim * 2, # [CLS | mean(patches)]
            n_slot, len(TARGETS),
        )  
        # standard ImageNet per-channel RGB mean and standard deviation:
        shape = (1, 3, 1, 1) # [image, R-G-B, Height, Width]
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(shape))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(shape))
 
    def forward(self, imgs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        imgs [batch_len, n_slots, R-G-B, Height, Width] uint8 or float in [0, 255], mask [batch_len, n_slots] bool -> logits [batch_len, n_out] float
        """
        B, S = imgs.shape[:2]
        filled = mask.reshape(B * S)
        x = (imgs.reshape(B * S, *imgs.shape[2:])[filled].float() / 255. - self.mean) / self.std
        cls, patch = self.emb(x) 
        # only filled slots (~81%) are passed through the backbone.
        # NOTE: missing slots do not contribute to the final prediction,
        # hence skip the expensive matmuls for them.
        features = torch.cat([cls, patch.mean(dim=1)], dim=1)
        slot_features = features.new_zeros(B * S, features.shape[1])
        slot_features[filled] = features
        return self.head(slot_features.reshape(B, S, -1), mask)
 
    def emb(self, x: torch.Tensor) -> tuple:
        """
        The encoder pass of scaled inputs [batch_len * n_slots, R-G-B, Height, Width] happens here.
        Extracts:
            CLS [batch_len * n_slots, dim],
            patches [batch_len * n_slots, n_patch, dim]
        """
        ViT = self.backbone
        t, (H, W) = ViT.prepare_tokens_with_masks(x)
        rope = ViT.rope_embed(H=H, W=W) if (ViT.rope_embed is not None) else None
        for block in ViT.blocks:
            t = block(t, rope)
        t = ViT.norm(t)
        # [batch_len * n_slots, CLS | storage tokens | patch tokens, dim] embeddings.
        return t[:, 0].float(), t[:, -(H * W):].float()
 
    def train(self, mode=True):
        """
        - backbone always in `eval` mode (no dropout / stochastic depth),
          except DoRA dropout, which follows `mode`.
        - head in `train` mode if mode=True.
        """
        super().train(mode)
        for name, module in self.backbone.named_modules():
            if not (isinstance(module, LoraLayer) or ".lora_dropout." in name):
                module.training = False
        return self

class _SlotHead(nn.Module):
    """
    Each StudyInstanceUID contains multiple slots with varying configurations, each contributing an RGB-like stack during an encoder pass.

    Not all image slots are equally important for each diagnosis.
    e.g.,
    - Cruciate ligaments and menisci are well evaluated on sagittal/coronal images, while patellofemoral cartilage is better evaluated on axial images.
    - Fluid-sensitive sequences with fat suppression are central to detecting edema, effusion.

    SlotHead serves as a diagnosis-specific attention pooling layer: 
    instead of averaging the slot embeddings, it learns to assign different weights to different slots for each diagnosis.
    """
    def __init__(
            self, 
            dim, # embedding dimensions
            n_slot, # number of MRI-slots in input
            n_out, # multi-label classification; each returns one logit
            hidden=256, # latent dimension
            r=0.2 # dropout rate
        ):
        super().__init__()
        self.proj = nn.Sequential(
            OrderedDict(
                norm=nn.LayerNorm(dim), 
                linear=nn.Linear(dim, hidden), 
                activation=nn.GELU(),
            )
            # for each (StudyInstanceUID, slot):
            #   ViT embedding --normalized--> hidden layer --activation--> .
        )
        self.slot_emb = nn.Parameter(
            torch.randn(n_slot, hidden) * 0.02
            # slot-level bias; does not vary with StudyInstanceUID. 
        )
        self.query = nn.Parameter(
            torch.randn(n_out, hidden) * 0.02
            # same feature may receive different importance for different label classes.
        )
        self.drop = nn.Dropout(r)
        self.out = nn.Linear(hidden, n_out)
        self.hidden = hidden
 
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x [batch_len, n_slots, dim] float, mask [batch_len, n_slots] bool -> logits [batch_len, n_out] float
        """
        x_ = self.proj(x) + self.slot_emb
        mask = mask.unsqueeze(1) 
        w = (
            torch.einsum(
                "bsh,oh->bos", x_, self.query,
                # (label, slot)-wise weighted sum of features.
            ) / (self.hidden ** 0.5)
        ).masked_fill(~mask, -1e4).softmax(dim=-1)
        # - `softmax` ensures w sums to 1 across slots.
        # - `masked_fill` ensures approximately w=0 for empty-slot placeholders.
        wx = torch.einsum(
            "bos,bsh->boh", w, x_,
            # label-wise weighted average of slot embeddings.
        )
        # logit for each diagnosis is derived independently from its corresponding embedding row;
        # different label classes do not interact with each other.
        return torch.einsum("boh,oh->bo", self.drop(wx), self.out.weight) + self.out.bias
 
# -------------------------

 
# ==================================================
# Train / Predict
# ==================================== 

from concurrent.futures import ThreadPoolExecutor

def _prefetch[Key, Value](load: Callable[[Key], Value], keys: Iterable[Key]) -> Iterator[tuple[Key, Value]]:
    """
    (asynchronous prefetching)

    Loads the next data chunk in a background thread while the main thread processes the current chunk.
    In the long run, each step takes `max(load, compute)` runtime rather than `load + compute`.
    """
    keys = iter(keys)
    try:
        key = next(keys)
    except StopIteration:
        return
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(load, key) # <first key>.
        for next_key in keys:
            next_future = pool.submit(load, next_key)
            yield key, future.result()
            key, future = next_key, next_future
        yield key, future.result() # <last key>.


EVAL_BATCH = 48
@torch.no_grad()
def predict(
        model, 
        X: np.ndarray, mask: np.ndarray | torch.Tensor, 
        idx: np.ndarray, 
        dev: torch.device,
        restore_weights: bool = False,
    ) -> np.ndarray:
    """
    - X [n_study, n_slot, n_slice, Height, Width]
    - mask [n_study, n_slots]; numpy is moved to `dev` once, a tensor already on `dev` is used as is.

    For any (StudyInstanceUID, slot), there can be multiple R-G-B stacks, 
    each contributing one prediction.
    The final prediction is the average logit (no sigmoid).
    """
    model.eval()
    n_group = X.shape[-3] // RGB
    H, W = X.shape[-2:]
    mask = torch.as_tensor(mask, device=dev) 
    is_cuda = dev.type == "cuda"
    out = []
    lora_layers = [module for module in model.modules() if isinstance(module, LoraLayer)]
    if restore_weights:
        base_weights = [layer.base_layer.weight.detach().clone() for layer in lora_layers]

    for layer in lora_layers:
        layer.merge()
        # fold LoRA into the base weight once,
        # instead of recomputing `W + BA` on every eval forward pass for `len(idx) / EVAL_BATCH`.

    batches = [idx[b:b + EVAL_BATCH] for b in range(0, len(idx), EVAL_BATCH)]
    for uid, X_batch in _prefetch(lambda uid: X[uid], batches): # [[MAIN]]
        X_batch = torch.tensor(X_batch, device=dev) 
        B, S = X_batch.shape[:2]
        # fold groups into the batch: [B, S, n_group * RGB, H, W] --> [B * n_group, S, RGB, H, W]
        X_batch = X_batch.reshape(B, S, n_group, RGB, H, W).transpose(1, 2).reshape(B * n_group, S, RGB, H, W)
        m = mask[torch.tensor(uid, device=dev)].repeat_interleave(n_group, dim=0) 
        with torch.autocast("cuda", enabled=is_cuda):
            logits = model(X_batch, m).float()

        # [B * n_group, n_out] --> [B, n_group, n_out] --mean over groups--> [B, n_out] logits.
        out.append(logits.reshape(B, n_group, -1).mean(dim=1))

    if restore_weights: # [[needed when training resumes]]
        for layer, base_weight in zip(lora_layers, base_weights):
            layer.unmerge()
            # restore the exact frozen weights before resuming training;
            # `unmerge` alone may introduce round-off errors.
            layer.base_layer.weight.data.copy_(base_weight)

    return torch.cat(out).cpu().numpy() if out else np.zeros((0, len(TARGETS)), np.float32) # append batches together.
 

EPOCHS = 25
BATCH_STUDIES, ACCUMULATION_STEPS = 4, 2 
GRADIENT_CLIP_NORM = 1.0
LR_HEAD, LR_BACKBONE, LR_PATCH_EMBED = 1e-3, 2e-4, 4e-5
RUNS = {
    "A": (0, 1, 2, 3), "B": (1, 2, 3, 4),
    # run -> training folds (4/5); 
    # the remaining (1/5) folds are held out.
} ## NOTE: every fold is used at least once for training.


def train_runs_on_gpu(gpu_index: int, n_gpu: int):
    """
    One process per GPU (cuda:gpu_index), launched in parallel via torch.multiprocessing.spawn(...);
    each process trains its share of RUNS one after another:
        T4 x2: cuda:0 -> A;  cuda:1 -> B.
    """
    global PREP_X_OUT
    PREP_X_OUT = LOCAL_PREP_X_OUT
    T0 = time.time()
    # TIME_BUDGET covers all runs of this process (one Kaggle session), not each run separately.
    for run_index in range(gpu_index, len(RUNS), n_gpu):
        train_run(run_index, gpu_index, T0)


def train_run(run_index: int, gpu_index: int, T0: float):
    """
    One run on cuda:gpu_index.
    """
    run_name = chr(run_index + 65)
    # name [0, 1, ...] ---> [A, B, ...]
    train_folds = RUNS[run_name]
    np.random.seed(SEED + run_index)
    torch.manual_seed(SEED + run_index)
    # ensemble diversity: A/B differ in initialization, augmentation, and batch order.

    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [{run_name}] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    has_cuda = True
    device = torch.device(f"cuda:{gpu_index}")
    torch.cuda.set_device(device)

    ## -- image data ----
    X, mask, mask_index = load_state(
        X = np.load(PREP_X_OUT / "train_X.npy", mmap_mode="r"),
        catalog = pd.read_parquet(PREP_X_OUT / "train_catalog.parquet"),
    ) # the raw pixel array and metadata.
    n_group = X.shape[-3] // RGB 
    N, S = X.shape[:2]
    logger.info(f"X {X.shape}; slots filled {int(mask.sum(axis=None))}/{mask.size}")
    ## ----
 
    ## -- targets ----
    target = pd.read_parquet(PREP_Y_OUT / "train_y.parquet").reindex(mask_index)

    Y = target[TARGETS].to_numpy(np.float32)
    W = target[[t + "__w" for t in TARGETS]].to_numpy(np.float32)
    labels = target[[t + "__label" for t in TARGETS]].to_numpy(bool)  
    ref = target["ref_flag"].to_numpy(bool) # flag studies having human-annotated labels.

    skip = ~ W.any(axis=1) # skip studies with all-zero weights; they do not contribute to training.
    logger.info(f"supervised {(~skip).sum()} of {N} studies")

    Y_device, W_device = torch.tensor(Y, device=device), torch.tensor(W, device=device)
    mask_device = torch.tensor(mask, device=device)
    # move Y, W, mask from CPU to GPU once (< 1 MB).
    # X is huge (10 GB+) and stays on disk; move to GPU chunk-wise as needed.
    ## ----

    ## -- train/validation 80/20 splits ----
    report = pd.read_csv(
        DATA_ROOT / "train.csv",
        usecols=["StudyInstanceUID", "Report"], dtype="string",
        index_col="StudyInstanceUID",
    ).rename(index=str.strip)["Report"].fillna("")
    folds = np.array(
        [
            int(
                hashlib.md5(r.encode("utf-8")).hexdigest()[:8], 
                16
            ) % 5
            for r in report.loc[mask_index]
        ]
        # deterministic bucketing into 5 folds;
        # duplicate reports get the same bucket to avoid leakage.
    )  
    in_train_folds = np.isin(folds, train_folds)
    validation_set = np.flatnonzero(~skip & ~in_train_folds)
    train_set = np.flatnonzero(~skip & in_train_folds)
    n_train, n_validation = len(train_set), len(validation_set)
    logger.info(f"train folds {train_folds}: train {n_train} / holdout {n_validation} studies.")
    val_labels = labels[validation_set] 
    # NOTE: hard-labels are only used for validation AUC tracking;
    # training loss is always derived from soft-scores.

    is_val_ref = ref[validation_set] 
    validation_ref = validation_set[is_val_ref]
    # indices in the validation set that have reference labels.
    val_ref_labels = labels[validation_ref]
    n_val_ref = len(validation_ref)
    logger.info(f"annotation check: {n_val_ref} of {ref.sum()} annotated studies are in the holdout.")
    ## ----

    ## -- fine-tuning ----
    scaler = torch.amp.GradScaler("cuda", enabled=has_cuda)
    model = build_ortho_model(n_slot=S).to(device)

    batch_head = np.arange(
        0, n_train - BATCH_STUDIES + 1, 
        BATCH_STUDIES,
    ) # first element in a batch.
    n_batches = len(batch_head) - len(batch_head) % ACCUMULATION_STEPS
    batch_head = batch_head[:n_batches]
    # make the number of batches a multiple of ACCUMULATION_STEPS.
    #
    # due to GPU memory constraints, we need a small BATCH_STUDIES
    # and use ACCUMULATION_STEPS to avoid noisy gradients;
    # the effective batch size becomes BATCH_STUDIES * ACCUMULATION_STEPS per gradient-descent update.
    accumulation_start_id, accumulation_end_id = 0, ACCUMULATION_STEPS - 1
    n_updates = n_batches // ACCUMULATION_STEPS
    assert n_updates > 0, "`n_updates` must be > 0; otherwise, `batch_head` is [] and no training occurs."

    for block in model.backbone.blocks:
        # forward pass is dominated by matmuls;
        # compile the blocks for faster training epochs.
        block.compile(dynamic=True) 
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    MLP_head, ViT = model.head, model.backbone
    block_DoRA = [module for module in ViT.blocks.modules() if isinstance(module, LoraLayer)]
    patch_DoRA = ViT.patch_embed.proj
    opt = torch.optim.AdamW([
        {
            "params": [MLP_head.slot_emb, MLP_head.query, MLP_head.proj.linear.weight, MLP_head.out.weight],
            "lr": LR_HEAD, "weight_decay": 0.02,
        },
        {
            "params": [MLP_head.proj.norm.weight, MLP_head.proj.norm.bias, MLP_head.proj.linear.bias, MLP_head.out.bias],
            "lr": LR_HEAD, "weight_decay": 0.0,
        },
        {
            "params": [layer.lora_A["default"].weight for layer in block_DoRA] + [layer.lora_B["default"].weight for layer in block_DoRA],
            "lr": LR_BACKBONE, "weight_decay": 0.02,
        },
        {
            "params": [layer.lora_magnitude_vector["default"].weight for layer in block_DoRA] + [ViT.norm.weight, ViT.norm.bias],
            "lr": LR_BACKBONE, "weight_decay": 0.0,
        },
        {
            "params": [patch_DoRA.lora_A["default"].weight, patch_DoRA.lora_B["default"].weight],
            "lr": LR_PATCH_EMBED, "weight_decay": 0.02,
        },
        {
            "params": [patch_DoRA.lora_magnitude_vector["default"].weight],
            "lr": LR_PATCH_EMBED, "weight_decay": 0.0,
        },
    ]) # no weight decay on Linear-layer biases, LayerNorm affine parameters, and DoRA magnitude vectors.
    assert (
        sorted(id(param) for group in opt.param_groups for param in group["params"]) 
        == sorted(id(param) for param in model.parameters() if param.requires_grad)
    ), "optimizer groups must contain every trainable parameter exactly once (and nothing frozen)."
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[group["lr"] for group in opt.param_groups],
        epochs=EPOCHS,
        steps_per_epoch=n_updates, 
        pct_start=0.15, anneal_strategy='cos',
        # cosine learning-rate scheduler.
    )

    def _read_train_batch(key):
        batch, g_id = key
        return X[batch, :, g_id * RGB:(g_id + 1) * RGB]

    best, best_state = -1.0, None
    # schedule the learning rate by EPOCHS, then skip the final few epochs
    # where the learning rate is too small for meaningful updates.
    for epoch in range(EPOCHS - 5): # [[MAIN]]
        model.train()
        current_permutation = np.random.permutation(train_set)
        batches = [
            (
                np.sort(current_permutation[b:b + BATCH_STUDIES]), 
                np.random.randint(n_group)
            )
            for b in batch_head
            # each (study, slot) contains multiple slices as R-G-B stacks:
            #   [x_l-1, x_l, x_l+1 | l-1, l, l+1 | m-1, m, m+1 | h-1, h, h+1 | x_h-1, x_h, x_h+1]
            #   <group-extreme_low>  <group-low> <group-medium> <group-high> <group-extreme_high>
            # for each batch in the training loop, we randomly pick only one group.
        ] 
        
        tot, nstep = 0.0, 0
        for (batch, _), X_batch in _prefetch(_read_train_batch, batches):
            imgs = augment(torch.tensor(X_batch, device=device))
            batch_device = torch.tensor(batch, device=device)
            m, y, w = mask_device[batch_device], Y_device[batch_device], W_device[batch_device]

            if nstep % ACCUMULATION_STEPS == accumulation_start_id:
                opt.zero_grad(set_to_none=True)

            with torch.autocast("cuda", enabled=has_cuda):
                loss = (
                    F.binary_cross_entropy_with_logits(
                        model(imgs, m), y, 
                        reduction="none",
                    ) * w
                    # weighted BCE loss on soft labels.
                    # equivalent to `BCEWithLogitsLoss(...)`,
                    # but each sample has different weight vectors.
                ).mean() / ACCUMULATION_STEPS
                # mean loss of the effective batch.
            scaler.scale(loss).backward()
            tot += loss.item()

            if nstep % ACCUMULATION_STEPS == accumulation_end_id:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    [param for group in opt.param_groups for param in group["params"]], 
                    GRADIENT_CLIP_NORM,
                )
                scaler.step(opt)
                scaler.update()
                scheduler.step()

            nstep += 1
 
        holdout_pred = predict(
            model, 
            X, mask_device, validation_set, 
            device, restore_weights=True,
        )
        holdout_label_auc = roc_auc_score(val_labels, holdout_pred, average=None) 
        holdout_auc = np.nanmean(holdout_label_auc)
        holdout_ref_auc = np.nanmean(roc_auc_score(val_ref_labels, holdout_pred[is_val_ref], average=None)) if n_val_ref else np.nan
        label_auc_text = "; ".join(f"{target} {auc:.4f}" for target, auc in zip(TARGETS, holdout_label_auc))
        logger.info(
            f"""
            EPOCH {epoch + 1}/{EPOCHS} | loss {tot / n_updates:.4f} | holdout macro auc {holdout_auc:.4f} | annotated studies (n={n_val_ref}) macro auc {holdout_ref_auc:.4f}
            holdout auc per label: {label_auc_text}"""
        )
                
        # best epoch is chosen based on full holdout AUC;
        # the annotated reference studies (~ 1.3% * 1/5) are too few to make a decision.
        if holdout_auc > best:
            best = holdout_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k in trainable}
            # only store the unfrozen parameters.
        if time.time() - T0 > TIME_BUDGET:
            logger.info("time budget reached")
            break
 
    torch.save(best_state, WORKING / f"model_ft_{run_name}.pt")
    logger.info(f"best holdout AUC {best:.4f}; saved model_ft_{run_name}.pt")
    del model, opt, scheduler, scaler
    gc.collect()
    ## ----
 
# -------------------------


# ++++++++++++++++++++++++++++++++++++++++++++++++


# [[THIS ENSURES THAT THE SUBMISSION SCRIPT CAN IMPORT THE EXACT MODEL ARCHITECTURE USED DURING FINE-TUNING.]]

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    shutil.copytree(PREP_X_OUT, LOCAL_PREP_X_OUT)
    n_gpu = torch.cuda.device_count()
    assert n_gpu >= 1, "need at least 1 GPU."
    torch.multiprocessing.spawn(train_runs_on_gpu, args=(n_gpu,), nprocs=min(n_gpu, len(RUNS)), join=True)
    shutil.copy(__file__, WORKING / "train.py")