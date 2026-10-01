"""
Prepare the training images:
****************************
[1.] Enlist all DICOM file names per (StudyInstanceUID, SeriesInstanceUID) and gather series-level information.
[2.] For a StudyInstanceUID, choose a SeriesInstanceUID with a given MRI configuration.
[3.] Order the DICOM files by their physical coordinates rather than file names, under every (StudyInstanceUID, SeriesInstanceUID).
[4.] Left and Right knees are anatomically mirror images of each other; normalize the data to the Left knee view whenever applicable.
[5.] Sample a few DICOM files per (StudyInstanceUID, SeriesInstanceUID) rather than processing all MRI slices; mainly to reduce runtime.
[6.] Store:
    X.npy --- [n_study, n_slot, N_SLICE, IMG, IMG]
    catalog.parquet --- (sl, StudyInstanceUID, slot, SeriesInstanceUID, file_name)

"""

from __future__ import annotations
from typing import Literal, Annotated
from collections import defaultdict
import logging, os, shutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path

import numpy as np, pandas as pd
import pydicom
import cv2 ; cv2.setNumThreads(1)


logger = logging.getLogger(__name__)


DATA_ROOT = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
WORKING = Path("/kaggle/working/")

BUFFER = Annotated[
    np.ndarray[tuple[int, int, int], np.dtype[np.uint8]],
    "(n_slice, Height, Width)"
    # contiguous memory for a (StudyInstanceUID, SeriesInstanceUID).
]
JOB_i = Annotated[
    tuple[np.uint16, str, np.uint8, pd.Series],
    """
    (
        sl_no, StudyInstanceUID, slot_id,
        SeriesInstanceUID-level [e.g., path, list of file names, etc.] metadata
    )
    """
    # corresponds to one row of output
]

# MRI slices are grayscale, while ViT expects R-G-B channels per input image;
# we stack 3 adjacent MRI slices as a single encoder pass.
N_GROUP_MAX = 3 # number of [R|G|B]-like stacks per (StudyInstanceUID, SeriesInstanceUID)
N_SLICE = 3 * N_GROUP_MAX
# Total number of slices used per (StudyInstanceUID, SeriesInstanceUID): 3 * N_GROUP_MAX
SLICE_BAND = (0.15, 0.85) 
# only the middle 15–85% of each ordered slice stack is sampled per (StudyInstanceUID, SeriesInstanceUID);
# edge slices are assumed to contain mostly background noise.
MID_SLICE_IDX = N_SLICE // 2 # single-slice representative of a (StudyInstanceUID, SeriesInstanceUID)


# ==================================================
# 1. Headers
# ====================================
HDR_THREADS = 16 
 

def HEADERS(split) -> pd.DataFrame:
    """
    Gather series-level information.
    """
    items = [(p.parent.name, p.name, p) for p in (DATA_ROOT / f"{split}_series").glob("*/*")]
    # (StudyInstanceUID, SeriesInstanceUID, path/to/series)
    with ThreadPoolExecutor(HDR_THREADS) as pool:
        H_dcm = (
            pd.DataFrame(
                pool.map(_probe, items),
                # each row corresponds to a _probe(...) call 
                # for a unique (StudyInstanceUID, SeriesInstanceUID) pair.
            )
            .astype(
                {
                    "StudyInstanceUID":'string', "SeriesInstanceUID":'string', 
                    "n_slices":int, "dir":object, "files":object,
                }
            )
            .set_index(["StudyInstanceUID", "SeriesInstanceUID"])
        )
    H_csv = pd.read_csv(
        DATA_ROOT / f"{split}_series.csv",
        index_col=["StudyInstanceUID", "SeriesInstanceUID"],
        dtype={
            "StudyInstanceUID":'string', "SeriesInstanceUID":'string',
            "Anatomical_Plane":'category', "Fluid_Sensitive":bool, "Fat_Suppression":bool,
        }, 
    )

    return (
        pd.merge(
            H_dcm.rename(index=str.strip), H_csv.rename(index=str.strip),
            left_index=True, right_index=True,
            # ASSUMPTION: there is no index mismatch between
            #    `*_series/` and `*_series.csv`
            # provided by the organisers.
        )
    )
 
def _probe(item) -> dict:
    study, series, path = item
    files = [f.name for f in path.glob("*.dcm")]
    return {
        "StudyInstanceUID": study, "SeriesInstanceUID": series, 
        "dir": path, "files": files, 
        "n_slices": len(files),
    }


# [[DICOM files contain critical geometric information in their metadata]]
# ****************************************************************************
#   X-axis: -ve <--- Patient's Right ---0--- Patient's Left ---> +ve
#   Y-axis: -ve <--- Patient's Face  ---0--- Patient's Back ---> +ve
#   Z-axis: -ve <--- Patient's Feet  ---0--- Patient's Head ---> +ve
#
# `ImagePositionPatient` (x mm, y mm, z mm)
#  --------------------
# Gives the 3D coordinates of the top-left (first) pixel of an MRI slice.
#
# `ImageOrientationPatient` [h_x, h_y, h_z | v_x, v_y, v_z]
#  -----------------------
# (h_x, h_y, h_z) is the unit vector in the direction of moving across the image (along a row).
# (v_x, v_y, v_z) is the unit vector in the direction of moving down the image (along a column).
#
# NOTE: MRI slices within a series are typically parallel 2D sections of a 3D volume.
# Hence, ImageOrientationPatient remains constant for a (StudyInstanceUID, SeriesInstanceUID),
# while ImagePositionPatient changes from slice to slice, primarily along the axis perpendicular to the image plane.
# Exception: scout/localizer or radial/multi-angle acquisitions may have varying image orientations within the same series.
#
# The actual physical distance is determined by 
# `PixelSpacing` (row_spacing mm/px, column_spacing mm/px).
 
 # -------------------------


# ==================================================
# 2. Slots
# ====================================
# {name: (Anatomical_Plane,Fluid_Sensitive,Fat_Suppression)}
SLOTS = {
    "SAG_FLUID_FS": ("Sagittal", True, True),
    "COR_FLUID_FS": ("Coronal", True, True),
    "AX_FLUID_FS": ("Axial", True, True),
    "SAG_NONFLUID_NOFS": ("Sagittal", False, False),
    "COR_NONFLUID_NOFS": ("Coronal", False, False),
    "AX_NONFLUID_NOFS": ("Axial", False, False),
}


def pick_slots(header:pd.DataFrame) -> dict:
    """
    For a StudyInstanceUID, 
    choose a SeriesInstanceUID with a given (Anatomical_Plane, Fluid_Sensitive, Fat_Suppression) configuration.

    When multiple qualifying SeriesInstanceUIDs exist under the same StudyInstanceUID,
    the series with the greater number of slices wins.
    """
    out = {}
    for uid, g in header.groupby(level="StudyInstanceUID"):
        out[uid] = {}
        for name, (plane, fluid, fatsuppr) in SLOTS.items():
            candidate = g[
                (g["Anatomical_Plane"] == plane) &
                ((g["Fluid_Sensitive"] | g["Fat_Suppression"]) == (fluid | fatsuppr))
            ]
            # ref - https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737312
            # NOTE: most clinically useful fluid-sensitive MSK sequences are also fat-suppressed;
            # hence, the model has never seen 
            # combinations like (Fluid_Sensitive=True, Fat_Suppression=False) or (Fluid_Sensitive=False, Fat_Suppression=True)
            # in its training data.
            # if such a rare configuration is encountered at test time, the model tries to route it
            # to the closest configuration seen during training.
            #
            # though an exact match is prioritized first.
            if not candidate.empty:
                out[uid][name] = (
                    candidate
                    .assign(exact=((candidate["Fluid_Sensitive"] == fluid) & (candidate["Fat_Suppression"] == fatsuppr)))
                    .nlargest(1, ["exact", "n_slices"])
                    .index.get_level_values("SeriesInstanceUID").item()
                )
            else:
                # placeholder to preserve slot position
                out[uid][name] = None 

    return out

# -------------------------

 
# ==================================================
# 3. Order
# ====================================
ORDER_TAGS = [
    "ImagePositionPatient", "ImageOrientationPatient",
]  # Type-1 metadata; mandatory with a value for classic single-frame MRI slices.


def _order_slices(HEADER_UIDrow: pd.Series) -> list:
    """
    Order the DICOM files by their physical coordinates rather than file names, 
    under every (StudyInstanceUID, SeriesInstanceUID).
    """
    files = HEADER_UIDrow["files"]
    try:
        geo = [
            pydicom.dcmread(
                HEADER_UIDrow["dir"] / f, 
                force=True, specific_tags=ORDER_TAGS,
            ) for f in files
        ]
        ImagePositionPatient = [
            np.asarray(ds.ImagePositionPatient, dtype=float) for ds in geo
        ]
        ImageOrientationPatient = np.asarray(geo[0].ImageOrientationPatient, dtype=float)
        if any(
            not np.allclose(ds.ImageOrientationPatient, ImageOrientationPatient, atol=1e-3)
            for ds in geo[1:]
        ):
            return files
            # NOTE: if the images do not have a common orientation, projection-based ordering is not possible; 
            # the series is returned in its original input order as is.
        
        SliceNormal = np.cross(
            ImageOrientationPatient[:3], ImageOrientationPatient[3:]
            # `ImageOrientationPatient` [h_x, h_y, h_z | v_x, v_y, v_z]
            # (h_x, h_y, h_z) and (v_x, v_y, v_z) are two unit vectors lying on the image plane.
            # their cross product gives the perpendicular direction to the image plane.
        )
        order = np.argsort([p @ SliceNormal for p in ImagePositionPatient])
        # sort the projection of each slice's first-pixel coordinate onto the slice-normal axis
    except Exception:
        return files

    return [files[i] for i in order] # files in sorted order


# -------------------------


# ==================================================
# 4. Resolve Laterality
# ====================================
L_R = {"L":+1, "R":-1}
LAT_MIN_OFFSET_MM = 20. 


def laterality(
        meta_X: pd.DataFrame,
    ) -> dict[str, Literal[-1, +1] | None]:
    """
    Left and right knees are anatomically 'MIRROR IMAGES' of each other.
    Knowing the MRI laterality is crucial for normalization.
    """
 
    tags, centre_x = defaultdict(set), defaultdict(list) 
    # `tags` captures the set of L/R tags present per StudyInstanceUID by combining all (StudyInstanceUID, SeriesInstanceUID) pairs.
    # `centre_x` captures the list of x-coordinate centres per StudyInstanceUID by combining all (StudyInstanceUID, SeriesInstanceUID) pairs.

    for (StudyInstanceUID, _), g in meta_X.groupby(["StudyInstanceUID", "SeriesInstanceUID"], sort=False):
        for t in g["Laterality"]:
            t = str(t).strip().upper()[:1]
            if t in L_R:
                tags[StudyInstanceUID].add(L_R[t])
        try:
            r = g.iloc[MID_SLICE_IDX]
            ImagePositionPatient, ImageOrientationPatient, PixelSpacing = (
                r["ImagePositionPatient"], r["ImageOrientationPatient"], r["PixelSpacing"]
            )
            x = float(
                ImagePositionPatient[0]  # x-coordinate of the top-left corner of the Width-by-Height image
                + ImageOrientationPatient[0] * PixelSpacing[1] * r["Columns"] / 2  # x-displacement by moving Width/2
                + ImageOrientationPatient[3] * PixelSpacing[0] * r["Rows"] / 2  # x-displacement by moving Height/2
            )
        except Exception:
            # e.g. placeholder rows for a series that failed to decode,
            # or a slot that was not picked.
            pass
        else:
            centre_x[StudyInstanceUID].append(x)  
 
    out = {}
    # Case-I: metadata provides unique laterality; choose from {"L": +1, "R": -1}.
    # Case-II: metadata is ambiguous across slices/SeriesInstanceUIDs of the same StudyInstanceUID; reset laterality to None.
    #
    # Case-III: metadata does not provide laterality;
    #   then check where the x-coordinate of the image centre falls.
    #   try to determine laterality from the sign of x-centre (except for cases very close to the zero-line).
    for StudyInstanceUID in meta_X["StudyInstanceUID"].unique():
        if StudyInstanceUID in tags:
            out[StudyInstanceUID] = next(iter(tags[StudyInstanceUID])) if len(tags[StudyInstanceUID]) == 1 else None   
        elif StudyInstanceUID in centre_x:
            m = np.median(centre_x[StudyInstanceUID])
            if abs(m) < LAT_MIN_OFFSET_MM:
                out[StudyInstanceUID] = None
            else:
                out[StudyInstanceUID] = int(np.sign(m))
        else:
            out[StudyInstanceUID] = None
 
    return out

def to_LeftKnee(
        img: BUFFER, 
        plane: Literal["Coronal", "Axial", "Sagittal"], side: Literal[-1, +1] | None,
    ) -> BUFFER:
    """
    Convert Right knee to Left knee view.

    - Coronal (front-to-back) series: flip the image horizontally.
    - Axial (top-to-bottom) series: flip the image horizontally.
    - Sagittal (side-to-side) series: reverse the stack ordering.

    """
    if side != -1:
        # if laterality is unresolved or already left knee,
        # keep the data as-is.
        return img
    
    match plane:
        case "Coronal" | "Axial":
            return img[:, :, ::-1]
        case "Sagittal":
            return img[::-1]
        case _:
            return img
        
# -------------------------


# ==================================================
# 5. Sample + Decode
#
# NOTE: PIXEL PITCH = CROP_MM / CACHE_IMG; 
# finer detail below the pixel pitch is irreversibly lost.
# ====================================
CACHE_IMG = 224 
# cache resolution (matches OrthoFoundation's pretraining resolution of 224 px)
CROP_MM_H, CROP_MM_W = (140.,140.)
# every image is centre-cropped to a row-by-column square,
# keeping the knee at a fixed physical scale across all scans.
#
# given a fixed resolution, a smaller field-of-view preserves better details.
# - ~92% of images are square (H = W); for the remaining, H/W: p25=0.97, p75=1.12.
# - combining H and W, the observed image sizes (resolution * pixel spacing): p5=140mm, p50=160mm, p80=170mm, p95=190mm.

IMG_TAGS = [
    "PixelSpacing", "Rows", "Columns", 
    "Laterality", 
    "ImagePositionPatient", "ImageOrientationPatient",
]  # per-slice metadata, captured only for the chosen N_SLICE slices for a given (StudyInstanceUID, SeriesInstanceUID)
META_COLS = {
    "sl": np.uint16 ,"StudyInstanceUID": 'string', "slot": np.uint8, "SeriesInstanceUID": 'string',
    "file_name": 'string',
    **{tag: object for tag in IMG_TAGS}
}


def read_Series(job_row: JOB_i) -> tuple[BUFFER | None, list[dict]]:
    """
    Stack multiple representative MRI slices corresponding to a given (StudyInstanceUID, SeriesInstanceUID)
    in a contiguous buffer as a 3D array (N_SLICE, Height, Width).

    Per-slice metadata is captured as a list of dictionaries.
    """
    sl, study, slot, HEADER_UIDrow = job_row

    if HEADER_UIDrow is None:
        # no series picked for the given slot
        return None, [{"sl": sl, "StudyInstanceUID": study, "slot": slot}] * N_SLICE

    series = HEADER_UIDrow.name[-1]
    files_ordered = _order_slices(HEADER_UIDrow)
    if not files_ordered:
        # no files to decode
        return None, [{"sl": sl, "StudyInstanceUID": study, "slot": slot, "SeriesInstanceUID": series}] * N_SLICE
    
    n_1 = len(files_ordered) - 1
    lo, hi = np.floor(SLICE_BAND[0] * n_1), np.ceil(SLICE_BAND[1] * n_1)
    centres = np.linspace(lo, hi, N_GROUP_MAX).round()
    idx = np.clip((centres[:, None] + np.array([-1, 0, 1])), 0, n_1).astype(int).ravel()
    # l-1, l, l+1 | m-1, m, m+1 | h-1, h, h+1
 
    # only store successfully decoded slices
    decoded = {
        i: data for i in np.unique(idx) 
        if (data := _decode_dcm(HEADER_UIDrow["dir"] / files_ordered[i])) is not None
    }
    if not decoded:
        # no slice from the series could be decoded
        return None, [{"sl": sl, "StudyInstanceUID": study, "slot": slot, "SeriesInstanceUID": series}] * N_SLICE
    planes = []
    metadata = []
    for i in idx:
        j = min(decoded, key=lambda j: abs(j - i))
        # abs(j - i) >= 0;
        # exact match j=i whenever i is in decoded,
        # otherwise use the nearest decoded slice index as a fallback.
        current_plane, current_meta = decoded[j]
        planes.append(current_plane)
        metadata.append(
            {
                "sl": sl, "StudyInstanceUID": study, 
                "slot": slot, "SeriesInstanceUID": series,
                "file_name": files_ordered[j], 
            } | current_meta
        )

    # keeping the knee at a fixed physical scale across all scans.
    planes = [_centre_crop(p, m) for p, m in zip(planes, metadata)]

    # one intensity scale for the whole slot rather than per-slice normalization,
    # preserving the relative scale across slices under a given (StudyInstanceUID, SeriesInstanceUID).
    lo_v, hi_v = np.percentile(
        np.concatenate(
            [
                p[r::2, c::2].ravel()
                for k, p in enumerate(planes)
                for r, c in [divmod(k % 4, 2)]
                # use 25% of pixels for quantile estimation,
                # with rotating offsets to reduce systematic bias.
            ]
        ),
        [1, 99],
    )
    scale = max(hi_v - lo_v, 1e-6) / 255 # [[8-bit greyscale 0-255]]

    # stack the slices under a given (StudyInstanceUID, SeriesInstanceUID)
    # into a contiguous memory buffer, while creating a new outermost axis.
    buffer = np.stack(
        [
            cv2.resize(
                np.clip((p - lo_v) / scale, 0, 255),
                (CACHE_IMG, CACHE_IMG),
                interpolation=cv2.INTER_AREA if min(p.shape) > CACHE_IMG else cv2.INTER_LINEAR,
                # shrinking: area averaging; enlarging: bilinear;
                # weighted average of values in [0, 255] --> output in [0, 255]
            ).round()
            for p in planes
        ],
        dtype=np.uint8, casting="unsafe",
    )
 
    return (
        buffer, metadata, 
        # buffer and metadata hold the slices in the same order.
    )

def _decode_dcm(path) -> tuple[np.ndarray, dict] | None:
    # rescaled float32 pixels of one DICOM slice with its meadata, 
    # or None if it the decode fails.
    try:
        ds = pydicom.dcmread(path, force=True)
        base = ds.pixel_array.astype(np.float32)
        m, c = (ds.get("RescaleSlope", 1) or 1), (ds.get("RescaleIntercept", 0) or 0)
        return float(c) + base * float(m), {t: ds.get(t) for t in IMG_TAGS}
    except Exception:
        return None

def _centre_crop(img: np.ndarray, meta: dict) -> np.ndarray:
    # crop an image around its centre, unless
    #  - the image is already smaller than the crop box, i.e. nothing to crop
    #  - the derived crop box is absurdly small (arbitrary default 16-by-16)
    try:
        rSpacing, cSpacing = meta["PixelSpacing"]
        crop_px = np.ceil(CROP_MM_H / rSpacing), np.ceil(CROP_MM_W / cSpacing)
    except Exception:
        crop_px = (None, None)
    h, w = [int(n_px) if 16 < (n_px or 0) < size else size for n_px, size in zip(crop_px, img.shape)]
    y0, x0 = (img.shape[0] - h) // 2, (img.shape[1] - w) // 2
    return img[y0:y0 + h, x0:x0 + w]
 
# -------------------------



# ==================================================
# 6. Run & Store
# ====================================
CHUNK = 1024
PIX_PROCESS = os.cpu_count()

 
def main(split: str, out: Path):
    """
    {split}_series --> out/X.npy, out/catalog.parquet written to disk
    """
    header = HEADERS(split) # series-level info
    studies = pick_slots(header)
    studies = dict(sorted(studies.items()))
    # {StudyInstanceUID: {slot_name: SeriesInstanceUID}} ; sorted by StudyInstanceUID
    logger.info(f"{len(studies)} study; {len(header)} series.")  

    ## -- allocate memory ----
    out.mkdir(parents=True, exist_ok=True)
    X = np.lib.format.open_memmap(
        out / f"{split}_X.npy", mode="w+", dtype=np.uint8,
        shape=(len(studies), len(SLOTS), N_SLICE, CACHE_IMG, CACHE_IMG),
        # [[X can be several GB in size]]
        # avoid loading the entire array into RAM.
        # keep the data on disk and access it through a memory-mapped array.
    )
    meta_X = []
    # X only captures the pixel array;
    # UID + metadata (e.g., height, width, laterality, geometry etc.)
    # are crucial for interpreting the buffer downstream.
    ## ----
 
    ## -- enlist all (StudyInstanceUID, slot_name, header_row) ----
    jobs = [
        (
            i,
            StudyInstanceUID, j, 
            header.loc[(StudyInstanceUID, SeriesInstanceUID)] if SeriesInstanceUID else None
        )
        for i, (StudyInstanceUID, SLOTS_i) in enumerate(studies.items())
        for j, SeriesInstanceUID in enumerate(SLOTS_i.values())
    ]
    valid_jobs = sum(1 for _, _, _, h in jobs if h is not None)
    num_jobs = len(jobs)
    logger.info(f"{valid_jobs} series to be processed out of {num_jobs} memory allocations.")
    ## ----

    ## -- fill buffer ----
    c = 0
    with ProcessPoolExecutor(PIX_PROCESS) as pool:
        for c0 in range(0, num_jobs, CHUNK):
            # bounds the images held in RAM at a time
            block = jobs[c0:c0 + CHUNK]
            current_out = pool.map(
                read_Series, block, 
                chunksize=8, 
            )
            for (sl, _, slot, _), (img, meta) in zip(block, current_out):
                meta_X.extend(meta)
                if img is not None:
                    X[sl, slot] = img
                    c += 1
            logger.info(f"decoded jobs: {c0} - {min(c0 + CHUNK, num_jobs)} | success {c}")
    X.flush()
    meta_X = (
        pd.DataFrame(meta_X)
        .astype(META_COLS)
    )
    # NOTE:
    # - SeriesInstanceUID is None when no series is picked for the slot.
    # - file_name is None when decoding fails.
    # The placeholder for UID still exists for consistency with X.
    catalog = ["sl", "StudyInstanceUID", "slot", "SeriesInstanceUID", "file_name"]
    meta_X[catalog].to_parquet(
        out / f"{split}_catalog.parquet",
        compression='zstd',
    )
    ## ----

    ## -- normalise laterality inplace ----
    lat = laterality(meta_X)
    lat_counter = defaultdict(int)
    planes = [p for p, _, _ in SLOTS.values()]
    filled = (
        meta_X.loc[meta_X["file_name"].notna(), ["sl", "slot", "StudyInstanceUID"]]
        .drop_duplicates()
        # one row per successful (study, slot)
    )  
    for row in filled.itertuples(index=False):
        lat_row = lat[row.StudyInstanceUID]
        if lat_row == -1:
            # disk I/O only for required cases.
            X[row.sl, row.slot] = to_LeftKnee(
                X[row.sl, row.slot], 
                planes[row.slot], lat_row
            )
        lat_counter[lat_row] += 1
    X.flush()
    logger.info(
        f"already left knee {lat_counter[1]} | laterality unresolved {lat_counter[None]} | converted right->left view {lat_counter[-1]}"
    )
    # laterality is defined at StudyInstanceUID level;
    # hence can be normalised only after all (StudyInstanceUID, SeriesInstanceUID) are processed.
    ## ----


# ++++++++++++++++++++++++++++++++++++++++++++++++




# - When pushed as its own kernel, the script processes the training data
# and writes `train_X.npy` and `train_catalog.parquet` to disk.
# It also keeps a copy of the script as `prep_X.py` in the output.

# - The final submission notebook can access `prep_X.py` 
# from the output of this kernel run
# and import `main()`.

# [[THIS ENSURES THAT THE TRAINING AND TEST SETS ARE PROCESSED THROUGH THE SAME PIPELINE.]]

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    main("train", WORKING)
    shutil.copy(__file__, WORKING / "prep_X.py")