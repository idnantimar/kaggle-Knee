"""
(ViT backbone + MLP head) fine-tuned on: X = MRI DICOM images, y = LLM-extracted labels.

Inputs (outputs of prep_X.py / prep_y.py kernel run):
*****************************************************
train_X.npy -- [n_study, n_slot, n_slice, Height, Width]
train_catalog.parquet -- (sl | StudyInstanceUID | slot | SeriesInstanceUID | file_name)
train_y.parquet -- (StudyInstanceUID | <target> | <ref> | <weight> | <hard labels>)

Outputs:
********
    model_ft.pt -- state_dict corresponding to the best epoch.
    train.py -- copy of this script, which the final notebook can import.

NOTE: The last few ViT blocks and the MLP head are trainable; all other layers are frozen.

"""

from __future__ import annotations

import gc, hashlib, logging, shutil, time, os
from pathlib import Path

import numpy as np, pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

SEED = 2026
INPUT = Path("/kaggle/input/")
WORKING = Path("/kaggle/working/")
DATA_ROOT = INPUT / "competitions/rsna-knee-abnormality-detection"
PREP_X_OUT, PREP_Y_OUT = INPUT / "notebooks/ramitnandi/rsna-knee-prep-x", INPUT / "notebooks/ramitnandi/rsna-knee-prep-y"
ORTHO = INPUT / "models/ramitnandi/orthofoundation/pytorch/default/1/orthofoundation"
 
 
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

TIME_BUDGET = 8 * 3600.  # sufficient buffer (compared to the Kaggle session limit of 12h).

def macro_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Macro ROC-AUC across label columns (excluding degenerate classes)."""
    return np.nanmean(
        roc_auc_score(y, p, average=None)
    )


# ==================================================
# Data
# ====================================
def load_split(prep_dir: Path, split: str) -> tuple:
    """
    Load state from prep_X kernel runs.
    """
    X = np.load(prep_dir / f"{split}_X.npy", mmap_mode="r")
    catalog = pd.read_parquet(prep_dir / f"{split}_catalog.parquet").sort_values(["sl", "slot"]) 
    # NOTE: X must be written in increasing order of (sl, slot).
    mask = (
        catalog.assign(ok=catalog["file_name"].notna())
        .groupby(["StudyInstanceUID", "slot"], sort=False)["ok"].all()  
        .unstack("slot", sort=False).astype(bool) 
    )
    assert mask.shape == X.shape[:2]
    assert (mask.index == catalog["StudyInstanceUID"].unique()).all()  
    assert (mask.columns == np.arange(X.shape[1])).all() 
    # better to assert the index order, though trivial;
    # otherwise, any change in pandas behaviour could silently produce
    # incorrect downstream outputs.
    return X, mask.to_numpy(bool), mask.index 
 

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
        # returns the source (x_in, y_in) =
        #   (x_out*cos - y_out*sin + tx, x_out*sin + y_out*cos + ty).
    )
    x = F.grid_sample(
        x, grid, 
        mode="bilinear", padding_mode="border", align_corners=False,
    )
    x = (x * scale).clamp(0, 255).round().to(imgs.dtype)
    return x.reshape(*lead, *unit_dim)

# -------------------------

 
# ==================================================
# Model
# ====================================
DINOV3_REPO, ORTHO_WEIGHTS = ORTHO / "dinov3", ORTHO / "OrthoFoundation.pt"
TUNE_BLOCKS = 4  
ORTHO_ROPE = True 


def build_ortho_model(n_slot: int) -> Model:
    """
    DINOv3 ViT-L/16 checkpoint
    ref: https://github.com/ytrsk/OrthoFoundation
    (with continued pretraining on 1.2M knee images)
        ├── early blocks       <frozen>
        ├── ...
        ├── last TUNE_BLOCKS   <trainable>
        └── final norm         <trainable>

        +
    SlotHead (attention pooling)
    
    """
    ViT = torch.hub.load(
        os.fspath(
            DINOV3_REPO.parent 
            # directory containing `hubconf.py`.
        ), 
        "dinov3_vitl16", 
        source="local", pretrained=False,
        # architecture only; no weight download.
    ) 
    checkpoint = torch.load(
        ORTHO_WEIGHTS, 
        map_location="cpu", weights_only=True,
        # plain dict of tensors.
    )  
    # NOTE: Different training frameworks may save weights
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
        # The checkpoint tensors must match the ViT-L/16 architecture exactly by name and shape.
    )  
    for param in ViT.parameters():
        param.requires_grad = False
    n_blocks = len(ViT.blocks)
    n_tune = min(n_blocks, TUNE_BLOCKS)
    for ft_layers in [*ViT.blocks[n_blocks - n_tune:], ViT.norm]:
        for param in ft_layers.parameters():
            param.requires_grad = True
    ViT.eval()
    n_ft = sum(p.numel() for p in ViT.parameters() if p.requires_grad) / 1e6
    logger.info(
        f"""
        backbone: OrthoFoundation-L, {n_blocks} blocks,
        fine-tuning last {n_tune} ({n_ft:.1f}M params), rope={ORTHO_ROPE}
        """
    )
    return Model(ViT, ViT.embed_dim, n_slot)


class SlotHead(nn.Module):
    """
    Each StudyInstanceUID contains multiple slots with varying configurations, each contributing an RGB-like stack during an encoder pass.

    Not all image slots are equally important for each diagnosis.
    e.g.,
    - Cruciate ligaments and menisci are well evaluated on sagittal and coronal images, while patellofemoral cartilage is better evaluated on axial images.
    - Fluid-sensitive sequences with fat suppression are central to detecting edema, effusion, and tears.

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
            nn.LayerNorm(dim), 
            nn.Linear(dim, hidden), 
            nn.GELU(),
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
            ) / self.hidden ** 0.5
        ).masked_fill(~mask, -1e4).softmax(-1)
        wx = torch.einsum(
            "bos,bsh->boh", w, x_,
            # label-wise weighted average of slot embeddings.
        )
        # logit for each diagnosis is derived independently from its corresponding embedding row;
        # different label classes do not interact with each other.
        return torch.einsum("boh,oh->bo", self.drop(wx), self.out.weight) + self.out.bias

 
class Model(nn.Module):
    """ViT backbone + MLP head"""
 
    def __init__(self, backbone, dim, n_slot):
        super().__init__()
        self.backbone = backbone
        self.head = SlotHead(
            dim * 2, # [CLS | mean(patches)]
            n_slot, len(TARGETS),
        )  
        # standard ImageNet per-channel RGB mean and standard deviation:
        shape = (1, 3, 1, 1) # [image, R-G-B, Height, Width]
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(shape))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(shape))
 
    def forward(self, imgs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        imgs [batch_len, n_slots, R-G-B, Height, Width] uint, mask [batch_len, n_slots] bool -> logits [batch_len, n_out] float
        """
        B, S = imgs.shape[:2]
        x = (imgs.reshape(B * S, *imgs.shape[2:]).float().div_(255.) - self.mean) / self.std
        cls, patch = self.emb(x) 
        return self.head(
            torch.cat([cls, patch.mean(1)], dim=1).reshape(B, S, -1), 
            mask,
        )
 
    def emb(self, x: torch.Tensor) -> tuple:
        """
        The encoder pass of scaled inputs [batch_len * n_slots, R-G-B, Height, Width] happens here.
        The initial blocks stay frozen; only the last few ViT blocks are fine-tuned.
        Extracts:
            CLS [batch_len * n_slots, dim],
            mean(patches [batch_len * n_slots, n_patch, dim])
        """
        ViT = self.backbone
        k0 = max(len(ViT.blocks) - TUNE_BLOCKS, 0)
        with torch.no_grad(): ## [[frozen blocks: 0, 1, ..., k0-1]]
            t, (H, W) = ViT.prepare_tokens_with_masks(x)
            rope = ViT.rope_embed(H=H, W=W) if (ORTHO_ROPE and ViT.rope_embed is not None) else None
            for block in ViT.blocks[:k0]:
                t = block(t, rope)
        for block in ViT.blocks[k0:]: ## [[fine-tuned layers: k0, k0+1, ...]]
            t = block(t, rope)
        t = ViT.norm(t)
        # [batch_len * n_slots, CLS | storage tokens | patch tokens, dim] embeddings.
        return t[:, 0].float(), t[:, -(H * W):].float()
 
    def train(self, mode=True):
        """
        - backbone always in `eval` mode (no dropout / stochastic depth).
        - head in `train` mode if mode=True.
        """
        super().train(mode)
        self.backbone.eval()  
        return self
 
# -------------------------

 
# ==================================================
# Train / Predict
# ==================================== 
EVAL_BATCH = 8

@torch.no_grad()
def predict(
        model, 
        X: np.ndarray, mask: np.ndarray, 
        idx: np.ndarray, 
        dev: torch.device,
    ) -> np.ndarray:
    """
    - X [n_study, n_slot, n_slice, Height, Width]
    - mask [n_study, n_slots]

    For any (StudyInstanceUID, slot), there can be multiple R-G-B stacks, 
    each contributing one prediction.
    The final prediction is `Sigmoid(average logits)`.
    """
    model.eval()
    n_slice = X.shape[-3]
    is_cuda = dev.type == "cuda"
    out = []
    for b in range(0, len(idx), EVAL_BATCH):
        uid = idx[b:b + EVAL_BATCH]
        m = torch.from_numpy(mask[uid]).to(dev)
        logits = []
        for g in range(0, n_slice, RGB):
            X_g = torch.from_numpy(
                np.ascontiguousarray(X[uid, :, g:g + RGB])
            ).to(dev)
            with torch.autocast("cuda", enabled=is_cuda):
                logits.append(model(X_g, m).float())
        out.append(torch.stack(logits, dim=1))  
    ## [len(idx), n_group, n_out] --mean over groups--> 
    ##      [len(idx), n_out] --sigmoid (element-wise)--> probabilities.
    if not out:
        return np.zeros((0, len(TARGETS)), np.float32)
    return torch.cat(out).mean(dim=1).sigmoid().cpu().numpy()
 

EPOCHS = 10
BATCH_STUDIES = 8
LR_HEAD, LR_BACKBONE, WEIGHT_DECAY = 1e-3, 2e-5, 0.02

def main():
    T0 = time.time()
    np.random.seed(SEED)
    torch.manual_seed(SEED)
 
    X, mask, mask_index = load_split(PREP_X_OUT, "train") # the raw pixel array and metadata.
    n_group = X.shape[-3] // RGB 
    N = len(X)
    logger.info(f"X {X.shape}; slots filled {int(mask.sum(axis=None))}/{mask.size}")
 
    ## -- targets ----
    target = pd.read_parquet(PREP_Y_OUT / "train_y.parquet").reindex(mask_index)

    Y = target[TARGETS].to_numpy(np.float32)
    W = target[[t + "__w" for t in TARGETS]].to_numpy(np.float32)
    labels = target[[t + "__label" for t in TARGETS]].to_numpy(bool)  
    ref = target["ref_flag"].to_numpy(bool) # flag studies having human-annotated labels.

    skip = ~ W.any(axis=1) # skip studies with all-zero weights; they do not contribute to training.
    logger.info(f"supervised {(~skip).sum()} of {N} studies")
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
        # deterministic bucketing: 0 | 1, 2, 3, 4;
        # duplicate reports get the same bucket to avoid leakage.
    )  
    validation_set = np.flatnonzero(~skip & (folds == 0))
    train_set = np.flatnonzero(~skip & (folds != 0))
    n_train, n_validation = len(train_set), len(validation_set)
    logger.info(f"train {n_train} / holdout {n_validation} studies")

    validation_ref = validation_set[ref[validation_set]] # indices out of the validation set that have reference labels.
    val_ref_labels = labels[validation_ref]
    n_val_ref = len(validation_ref)
    logger.info(f"annotation check: {n_val_ref} of {ref.sum()} annotated studies are in the holdout")
    val_labels = labels[validation_set] 
    # NOTE: hard-labels are only used for validation AUC tracking;
    # training loss is always derived from soft-scores.

    ## ----

    ## -- fine-tuning ----
    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")
    scaler = torch.amp.GradScaler("cuda", enabled=has_cuda)
    model = build_ortho_model(n_slot=X.shape[1]).to(device)
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    opt = torch.optim.AdamW(
        [
            {"params": model.head.parameters(), "lr": LR_HEAD},
            {"params": [param for param in model.backbone.parameters() if param.requires_grad], "lr": LR_BACKBONE},
        ], weight_decay=WEIGHT_DECAY,
    )
    steps = max(EPOCHS * (n_train // BATCH_STUDIES), 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[LR_HEAD, LR_BACKBONE],
        total_steps=steps, pct_start=0.15,
    )
    
    best, best_state = -1.0, None
 
    for epoch in range(EPOCHS): # [[MAIN]]
        model.train()
        current_permutation = np.random.permutation(train_set)
        tot, nstep = 0.0, 0
        for b_start in range(0, n_train - BATCH_STUDIES + 1, BATCH_STUDIES):
            batch = np.sort(current_permutation[b_start:b_start + BATCH_STUDIES])
            g_id = np.random.randint(n_group) 
            # each (study, slot) contains multiple slices as R-G-B stacks:
            #   [l-1, l, l+1 | m-1, m, m+1 | h-1, h, h+1]
            #      <group-l>    <group-m>     <group-h>
            # for each batch in the training loop, we randomly pick only one group.
            imgs = augment(
                torch.from_numpy(
                    np.ascontiguousarray(X[batch, :, g_id * RGB:(g_id + 1) * RGB])
                ).to(device)
            )
            m = torch.from_numpy(mask[batch]).to(device)
            y = torch.from_numpy(Y[batch]).to(device)
            w = torch.from_numpy(W[batch]).to(device)
            with torch.autocast("cuda", enabled=has_cuda):
                loss = (
                    # weighted BCE loss on soft-labels.
                    F.binary_cross_entropy_with_logits(
                        model(imgs, m), y, 
                        reduction="none",
                    ) * w
                ).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            tot += loss.item()
            nstep += 1
 
        holdout_auc = macro_auc(
            val_labels, 
            predict(model, X, mask, validation_set, device)
        )
        holdout_ref_auc = macro_auc(
            val_ref_labels, 
            predict(model, X, mask, validation_ref, device)
        ) if len(validation_ref) else np.nan
        logger.info(
            f"""EPOCH {epoch + 1}/{EPOCHS} | loss {tot / max(nstep, 1):.4f}; holdout auc {holdout_auc:.4f}; holdout annotated studies(n={n_val_ref}) auc {holdout_ref_auc:.4f};""")
                
        # best epoch is chosen based on full holdout AUC;
        # the annotated reference studies (~ 1.3% * 1/5) are too few to make a decision.
        if holdout_auc > best:
            best = holdout_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k in trainable}
            # only store the unfrozen parameters.
        if time.time() - T0 > TIME_BUDGET:
            logger.info("time budget reached")
            break
 
    torch.save(best_state, WORKING / "model_ft.pt")
    logger.info(f"best holdout AUC {best:.4f}; saved model_ft.pt")
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

    main()
    shutil.copy(__file__, WORKING / "train.py")