"""
train.py — Train Fast SCNN v2 from scratch on a GeoTIFF orthomosaic
            labelled with a solar-panel Shapefile.

Usage
-----
    python train.py
    python train.py --ortho Data/Orthomosaic_Patisen.tif \
                    --shp   Data/Panneaux_Patisen.shp
    python train.py --epochs 80 --batch_size 8 --tile_size 512

Outputs (all in --output_dir)
-------------------------------
    best_model.h5       Best weights (highest val IoU)
    training_log.csv    Per-epoch metrics
    training_curves.png Loss + IoU plots
"""

import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import geopandas as gpd
import tensorflow as tf
from rasterio.features import rasterize as rio_rasterize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_list import fast_scnn_2 as _fast_scnn_2

# ── Defaults ──────────────────────────────────────────────────────────────────
ORTHO_PATH = "Data/Orthomosaic_Patisen.tif"
SHP_PATH   = "Data/Panneaux_Patisen.shp"
OUTPUT_DIR = "trained_models/patisen"

TILE_SIZE        = 512
STRIDE           = 256
BATCH_SIZE       = 8
EPOCHS           = 200
LR               = 1e-4
VAL_FRAC         = 0.2
PANEL_W          = 15.0  # loss weight for the panel class vs background
PANEL_OVERSAMPLE = 4     # repeat panel tiles N× per epoch to balance batches


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train Fast SCNN v2 on solar panels")
    p.add_argument("--ortho", nargs='+', default=[ORTHO_PATH],
                   help="One or more GeoTIFF orthomosaics (one per site)")
    p.add_argument("--shp",   nargs='+', default=[SHP_PATH],
                   help="Matching shapefiles, one per --ortho")
    p.add_argument("--output_dir",  default=OUTPUT_DIR)
    p.add_argument("--tile_size",   type=int,   default=TILE_SIZE)
    p.add_argument("--stride",      type=int,   default=STRIDE)
    p.add_argument("--batch_size",  type=int,   default=BATCH_SIZE)
    p.add_argument("--epochs",      type=int,   default=EPOCHS)
    p.add_argument("--lr",          type=float, default=LR)
    p.add_argument("--val_frac",    type=float, default=VAL_FRAC)
    p.add_argument("--panel_weight",    type=float, default=PANEL_W,
                   help="Loss weight for panel pixels relative to background")
    p.add_argument("--panel_oversample", type=int, default=PANEL_OVERSAMPLE,
                   help="Repeat panel tiles N× per epoch to improve batch balance")
    p.add_argument("--max_tiles_per_site", type=int, default=0,
                   help="Cap tiles per site (0=unlimited). Use ~5000 for very large sites like Malicounda.")
    p.add_argument("--model", default="fast_scnn_2",
                   choices=["fast_scnn_2", "unet_resnet50", "unet_resnet50_attn", "unet_efficientnetb4"],
                   help="Model architecture to train")
    p.add_argument("--freeze_encoder_epochs", type=int, default=20,
                   help="(unet_resnet50 only) Epochs with frozen ResNet50 encoder before fine-tuning")
    p.add_argument("--resume", action="store_true",
                   help="Load best_model.weights.h5 from output_dir before training")
    p.add_argument("--resume_weights", default=None,
                   help="Load weights from this path before training (e.g. fine-tune from v2 weights)")
    p.add_argument("--skip_phase1", action="store_true",
                   help="Skip frozen Phase 1 and go directly to Phase 2 (requires --resume)")
    p.add_argument("--initial_epoch", type=int, default=None,
                   help="Starting epoch for Phase 2 when using --skip_phase1 (default: freeze_encoder_epochs)")
    p.add_argument("--freeze_encoder_bn", action="store_true",
                   help="Keep BatchNorm layers frozen in encoder during Phase 2 (recommended when batch_size<=2)")
    p.add_argument("--mixed_precision", action="store_true",
                   help="Enable float16 mixed precision — halves activation memory, allows batch_size=4 on 16GB GPU")
    p.add_argument("--max_panel_tiles", type=int, default=0,
                   help="Cap total panel tiles across all sites at N, distributed proportionally by panel pixel count. 0=unlimited.")
    p.add_argument("--loss", default="combo",
                   choices=["combo", "focal_tversky"],
                   help="Loss function: 'combo' (wBCE+Dice) or 'focal_tversky' (better for hard examples)")
    return p.parse_args()


# ── GPU ───────────────────────────────────────────────────────────────────────
def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[GPU] {len(gpus)} device(s): {[g.name for g in gpus]} — memory growth enabled")
    else:
        print("[GPU] None found — running on CPU (slow)")
    return bool(gpus)


# ── Data: rasterization ───────────────────────────────────────────────────────
def rasterize_labels(ortho_path: str, shp_path: str) -> np.ndarray:
    """Burn shapefile polygons onto a binary uint8 mask matching the ortho."""
    with rasterio.open(ortho_path) as src:
        height, width = src.height, src.width
        transform = src.transform
        raster_crs = src.crs

    gdf = gpd.read_file(shp_path)
    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)

    shapes = [(geom, 255) for geom in gdf.geometry if geom is not None]
    mask = rio_rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )
    n_panel = int((mask == 255).sum())
    total   = height * width
    print(f"[Mask] {n_panel:,} panel px / {total:,} total ({n_panel/total*100:.1f}%)")
    return mask


# ── Data: site loader ─────────────────────────────────────────────────────────
_MAX_RAM_GB = 10.0  # uncompressed RGB uint8 RAM limit (not file size)

def load_site(ortho_path: str, shp_path: str) -> dict:
    """Load one site. Images whose uncompressed uint8 RGB fits under _MAX_RAM_GB
    are fully loaded into RAM for fast tile access; larger ones use windowed reads."""
    file_gb = os.path.getsize(ortho_path) / 1e9
    with rasterio.open(ortho_path) as src:
        height, width, n_bands = src.height, src.width, src.count
    print(f"       {width}×{height} px | {n_bands} bands | {file_gb:.1f} GB on disk")

    mask = rasterize_labels(ortho_path, shp_path)

    # Estimate uncompressed RAM: height × width × 3 bands × 1 byte (uint8)
    ram_gb = height * width * 3 / 1e9
    if ram_gb <= _MAX_RAM_GB:
        t0 = time.time()
        with rasterio.open(ortho_path) as src:
            bands = [1, 2, 3] if src.count >= 3 else [1, 1, 1]
            rgb = np.moveaxis(src.read(bands), 0, -1)
        print(f"       Loaded into RAM in {time.time()-t0:.1f}s")
        return {'rgb': rgb, 'mask': mask, 'path': ortho_path,
                'height': height, 'width': width}
    else:
        print(f"       Too large for RAM — windowed tile reading enabled")
        return {'rgb': None, 'mask': mask, 'path': ortho_path,
                'height': height, 'width': width}


# ── Data: tile indices ─────────────────────────────────────────────────────────
def get_tile_indices(height, width, tile_size, stride, val_frac,
                     mask: np.ndarray | None = None):
    """
    Return (train_indices, val_indices) — lists of (row, col) top-left corners.

    Strategy: random shuffle of all tiles, then 80/20 split.
    Tiles with at least one panel pixel are prioritised into the split so that
    validation always contains panel examples.
    """
    import random

    all_idx = []
    for row in range(0, height - tile_size + 1, stride):
        for col in range(0, width - tile_size + 1, stride):
            all_idx.append((row, col))

    # Separate panel tiles from background-only tiles
    if mask is not None:
        panel_idx = [(r, c) for r, c in all_idx
                     if mask[r:r+tile_size, c:c+tile_size].max() > 0]
        bg_idx    = [(r, c) for r, c in all_idx
                     if mask[r:r+tile_size, c:c+tile_size].max() == 0]
    else:
        panel_idx, bg_idx = all_idx, []

    # Shuffle both groups with a fixed seed for reproducibility
    rng = random.Random(42)
    rng.shuffle(panel_idx)
    rng.shuffle(bg_idx)

    # Take val_frac from panel tiles first, then from background tiles
    n_val_panel = max(1, int(len(panel_idx) * val_frac))
    n_val_bg    = max(0, int(len(bg_idx)    * val_frac))

    val_idx   = panel_idx[:n_val_panel] + bg_idx[:n_val_bg]
    train_idx = panel_idx[n_val_panel:] + bg_idx[n_val_bg:]

    # Re-shuffle train so panel and background tiles are interleaved
    rng.shuffle(train_idx)

    return train_idx, val_idx


# ── Data: TF Dataset ──────────────────────────────────────────────────────────
@tf.function
def _augment(img, msk):
    """Geometric + photometric augmentation; img and msk transformed jointly."""
    # Stack along channel axis so geometric ops stay in sync
    combined = tf.concat([img, msk], axis=-1)          # (H, W, 5)
    combined = tf.image.random_flip_left_right(combined)
    combined = tf.image.random_flip_up_down(combined)
    k = tf.random.uniform((), 0, 4, dtype=tf.int32)
    combined = tf.image.rot90(combined, k)
    img = combined[..., :3]
    msk = combined[..., 3:]
    # Photometric: image only
    img = tf.image.random_brightness(img, max_delta=0.15)
    img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, msk


def make_dataset(sites: list, indices: list, tile_size: int, batch_size: int,
                 augment: bool = False, shuffle: bool = False, n_outputs: int = 1):
    """
    Multi-site tf.data.Dataset.
    n_outputs=1  → returns (img, mask)            for standard models
    n_outputs=3  → returns (img, (mask, mask, mask)) for Attention U-Net deep supervision
    """
    site_ids = np.array([s for s, r, c in indices], dtype=np.int32)
    rows     = np.array([r for s, r, c in indices], dtype=np.int32)
    cols     = np.array([c for s, r, c in indices], dtype=np.int32)

    def load_tile(sid_t, r_tensor, c_tensor):
        sid = int(sid_t.numpy())
        r   = int(r_tensor.numpy())
        c   = int(c_tensor.numpy())
        site = sites[sid]

        if site['rgb'] is not None:
            img = site['rgb'][r:r+tile_size, c:c+tile_size].astype(np.float32) / 255.0
        else:
            with rasterio.open(site['path']) as src:
                bands = [1, 2, 3] if src.count >= 3 else [1, 1, 1]
                window = rasterio.windows.Window(c, r, tile_size, tile_size)
                img = np.moveaxis(src.read(bands, window=window), 0, -1).astype(np.float32) / 255.0

        msk = site['mask'][r:r+tile_size, c:c+tile_size]
        msk_oh = np.stack(
            [(msk == 0).astype(np.float32),
             (msk == 255).astype(np.float32)],
            axis=-1,
        )
        return img, msk_oh

    def map_fn(sid, r, c):
        img, msk = tf.py_function(load_tile, [sid, r, c], [tf.float32, tf.float32])
        img.set_shape((tile_size, tile_size, 3))
        msk.set_shape((tile_size, tile_size, 2))
        return img, msk

    def map_fn_deep_sup(sid, r, c):
        img, msk = tf.py_function(load_tile, [sid, r, c], [tf.float32, tf.float32])
        img.set_shape((tile_size, tile_size, 3))
        msk.set_shape((tile_size, tile_size, 2))
        # deep supervision: same ground-truth for all 3 outputs
        return img, {"main_output": msk, "aux_d3_output": msk, "aux_d4_output": msk}

    ds = tf.data.Dataset.from_tensor_slices((site_ids, rows, cols))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(indices), reshuffle_each_iteration=True)
    _map = map_fn_deep_sup if n_outputs == 3 else map_fn
    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        if n_outputs == 3:
            def _augment_deep(img, y_dict):
                aug_img, aug_msk = _augment(img, y_dict["main_output"])
                return aug_img, {"main_output": aug_msk,
                                 "aux_d3_output": aug_msk,
                                 "aux_d4_output": aug_msk}
            ds = ds.map(_augment_deep, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ── Metrics ───────────────────────────────────────────────────────────────────
def panel_iou(y_true, y_pred):
    """IoU for the panel class. Returns 0 (not 1) when no panels are present,
    so empty tiles do not inflate the metric to a false 1.0."""
    y_true_bin = tf.cast(tf.equal(tf.argmax(y_true, axis=-1), 1), tf.float32)
    y_pred_bin = tf.cast(tf.equal(tf.argmax(y_pred, axis=-1), 1), tf.float32)
    intersection = tf.reduce_sum(y_true_bin * y_pred_bin)
    union = tf.reduce_sum(y_true_bin) + tf.reduce_sum(y_pred_bin) - intersection
    return tf.where(tf.equal(union, 0.0), 0.0, intersection / union)


# ── Loss ──────────────────────────────────────────────────────────────────────
def make_combo_loss(panel_weight: float):
    """Weighted categorical BCE + Dice on the panel class."""
    def combo_loss(y_true, y_pred):
        weight_map = y_true[..., 0] * 1.0 + y_true[..., 1] * panel_weight
        bce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        w_bce = tf.reduce_mean(bce * weight_map)

        true_p = y_true[..., 1]
        pred_p = y_pred[..., 1]
        smooth = 1.0
        intersection = tf.reduce_sum(true_p * pred_p)
        dice = 1.0 - (2.0 * intersection + smooth) / (
            tf.reduce_sum(true_p) + tf.reduce_sum(pred_p) + smooth
        )
        return w_bce + dice

    combo_loss.__name__ = "combo_loss"
    return combo_loss


def make_focal_tversky_loss(panel_weight: float, alpha: float = 0.3,
                             beta: float = 0.7, gamma: float = 1.33):
    """
    Focal Tversky Loss — designed for highly imbalanced segmentation.
    alpha = FP penalty weight, beta = FN penalty weight (beta > alpha favours recall).
    gamma > 1 down-weights easy (background) examples and focuses training on hard
    (missed panel) pixels. Combined with weighted BCE for numerical stability.
    """
    def focal_tversky_loss(y_true, y_pred):
        true_p = y_true[..., 1]
        pred_p = y_pred[..., 1]
        smooth = 1e-6
        tp = tf.reduce_sum(true_p * pred_p)
        fp = tf.reduce_sum((1.0 - true_p) * pred_p)
        fn = tf.reduce_sum(true_p * (1.0 - pred_p))
        tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
        ftl = tf.pow(1.0 - tversky, 1.0 / gamma)

        weight_map = y_true[..., 0] * 1.0 + y_true[..., 1] * panel_weight
        bce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        w_bce = tf.reduce_mean(bce * weight_map)
        return ftl + 0.5 * w_bce

    focal_tversky_loss.__name__ = "focal_tversky_loss"
    return focal_tversky_loss


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(tile_size: int, lr: float, panel_weight: float,
                model_name: str = "fast_scnn_2", loss_name: str = "combo"):
    """Build and compile the requested model. Returns (model, encoder_or_None)."""
    encoder = None
    if model_name == "fast_scnn_2":
        model = _fast_scnn_2.fast_scnn_v2(
            input_shape=(tile_size, tile_size, 3),
            batch_size=None,
            n_labels=2,
            model_summary=False,
        )
    elif model_name == "unet_resnet50":
        from model_list import unet_resnet50 as _unet_r50
        model, encoder = _unet_r50.build_unet_resnet50(
            input_shape=(tile_size, tile_size, 3),
            n_labels=2,
        )
    elif model_name == "unet_resnet50_attn":
        from model_list import unet_resnet50_attn as _unet_attn
        model, encoder = _unet_attn.build_unet_resnet50_attn(
            input_shape=(tile_size, tile_size, 3),
            n_labels=2,
        )
    elif model_name == "unet_efficientnetb4":
        from model_list import unet_efficientnetb4 as _unet_eff
        model, encoder = _unet_eff.build_unet_efficientnetb4(
            input_shape=(tile_size, tile_size, 3),
            n_labels=2,
        )
    else:
        raise ValueError(f"Unknown model: {model_name!r}")

    # Attention U-Net outputs [main, aux_d3, aux_d4] — use weighted multi-output loss
    is_deep_sup = model_name == "unet_resnet50_attn"
    loss_fn_single = (make_focal_tversky_loss(panel_weight)
                      if loss_name == "focal_tversky"
                      else make_combo_loss(panel_weight))
    if is_deep_sup:
        loss_fn = {
            "main_output":  loss_fn_single,
            "aux_d3_output": loss_fn_single,
            "aux_d4_output": loss_fn_single,
        }
        loss_weights = {"main_output": 1.0, "aux_d3_output": 0.4, "aux_d4_output": 0.2}
        metrics_cfg  = {"main_output": [panel_iou]}
    else:
        loss_fn      = loss_fn_single
        loss_weights = None
        metrics_cfg  = [panel_iou]
    print(f"[Model] Loss: {loss_name}{'  deep_supervision=ON' if is_deep_sup else ''}")
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
        opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
        print("[Model] LossScaleOptimizer enabled for mixed precision")
    model.compile(
        optimizer=opt,
        loss=loss_fn,
        loss_weights=loss_weights,
        metrics=metrics_cfg,
    )
    return model, encoder


# ── Plotting ──────────────────────────────────────────────────────────────────
def save_curves(history, output_dir: str):
    hist = history.history
    iou_key     = next((k for k in hist if "iou" in k and not k.startswith("val_")), None)
    val_iou_key = ("val_" + iou_key) if iou_key and ("val_" + iou_key) in hist else None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(hist["loss"],     label="train")
    axes[0].plot(hist["val_loss"], label="val")
    axes[0].set_title("Loss  (weighted BCE + Dice)")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if iou_key and val_iou_key:
        axes[1].plot(hist[iou_key],     label="train")
        axes[1].plot(hist[val_iou_key], label="val")
        axes[1].set_title("IoU  (panel class)")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[Plot] Saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    setup_gpu()

    if args.mixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("[GPU] Mixed precision enabled — float16 activations, float32 weights/gradients")

    if len(args.ortho) != len(args.shp):
        raise ValueError(f"--ortho ({len(args.ortho)}) and --shp ({len(args.shp)}) must match")

    # 1. Load all sites
    sites = []
    for i, (op, sp) in enumerate(zip(args.ortho, args.shp)):
        print(f"\n[Site {i+1}/{len(args.ortho)}] {os.path.basename(op)}")
        sites.append(load_site(op, sp))

    # 2. Tile indices per site, tagged with site_id
    import random as _random
    all_train_idx = []
    all_val_idx   = []
    for i, site in enumerate(sites):
        train_i, val_i = get_tile_indices(
            site['height'], site['width'],
            args.tile_size, args.stride, args.val_frac, mask=site['mask']
        )
        if args.max_tiles_per_site > 0:
            rng = _random.Random(42 + i)
            if len(train_i) > args.max_tiles_per_site:
                train_i = rng.sample(train_i, args.max_tiles_per_site)
            n_val_cap = max(1, int(args.max_tiles_per_site * args.val_frac))
            if len(val_i) > n_val_cap:
                val_i = rng.sample(val_i, n_val_cap)
        all_train_idx.extend([(i, r, c) for r, c in train_i])
        all_val_idx.extend([(i, r, c)   for r, c in val_i])

    # 3. Oversample panel tiles across all sites
    panel_train = [(s, r, c) for s, r, c in all_train_idx
                   if sites[s]['mask'][r:r+args.tile_size, c:c+args.tile_size].max() > 0]
    bg_train    = [(s, r, c) for s, r, c in all_train_idx
                   if sites[s]['mask'][r:r+args.tile_size, c:c+args.tile_size].max() == 0]

    # Proportional panel tile capping: distribute max_panel_tiles by mask pixel count per site
    if args.max_panel_tiles > 0 and panel_train:
        by_site = {}
        for s, r, c in panel_train:
            by_site.setdefault(s, []).append((s, r, c))
        site_px = [int(sites[s]['mask'].sum()) for s in range(len(sites))]
        total_px = sum(site_px[s] for s in by_site)
        print(f"\n[Data] Capping panel tiles → {args.max_panel_tiles} total (proportional by panel pixels):")
        capped = []
        for s_idx in sorted(by_site):
            tiles = by_site[s_idx]
            w = site_px[s_idx] / total_px if total_px > 0 else 1.0 / len(by_site)
            cap = max(1, round(args.max_panel_tiles * w))
            chosen = _random.Random(99 + s_idx).sample(tiles, min(cap, len(tiles)))
            capped.extend(chosen)
            print(f"         Site {s_idx} ({os.path.basename(args.ortho[s_idx])}): "
                  f"{len(tiles)} avail → {len(chosen)} kept  ({w:.1%} weight, cap={cap})")
        panel_train = capped

    oversampled = panel_train * args.panel_oversample + bg_train
    _random.Random(42).shuffle(oversampled)
    print(f"\n[Data] {len(args.ortho)} site(s) combined — train: {len(all_train_idx)} raw "
          f"({len(panel_train)} panel × {args.panel_oversample} + {len(bg_train)} bg "
          f"= {len(oversampled)} oversampled), val: {len(all_val_idx)}")
    if oversampled:
        est = args.batch_size * len(panel_train) * args.panel_oversample / len(oversampled)
        print(f"       Expected panel tiles/batch: {est:.1f} / {args.batch_size}")

    # 4. TF Datasets — deep supervision needs 3 y targets
    _n_out = 3 if args.model == "unet_resnet50_attn" else 1
    train_ds = make_dataset(sites, oversampled, args.tile_size, args.batch_size,
                            augment=True,  shuffle=True,  n_outputs=_n_out)
    val_ds   = make_dataset(sites, all_val_idx,  args.tile_size, args.batch_size,
                            augment=False, shuffle=False, n_outputs=_n_out)

    # 5. Model
    print(f"\n[Model] {args.model}  input={args.tile_size}×{args.tile_size}  "
          f"lr={args.lr}  panel_weight={args.panel_weight}  loss={args.loss}")
    model, encoder = build_model(args.tile_size, args.lr, args.panel_weight,
                                  args.model, args.loss)
    model.summary(line_length=90)

    # Freeze encoder during warm-up phase (U-Net variants only)
    is_deep_sup = args.model == "unet_resnet50_attn"
    if encoder is not None and args.freeze_encoder_epochs > 0:
        encoder.trainable = False
        _loss_s = (make_focal_tversky_loss(args.panel_weight)
                   if args.loss == "focal_tversky"
                   else make_combo_loss(args.panel_weight))
        if is_deep_sup:
            _loss_cfg = {"main_output": _loss_s, "aux_d3_output": _loss_s, "aux_d4_output": _loss_s}
            _lw = {"main_output": 1.0, "aux_d3_output": 0.4, "aux_d4_output": 0.2}
            _met = {"main_output": [panel_iou]}
        else:
            _loss_cfg, _lw, _met = _loss_s, None, [panel_iou]
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
            loss=_loss_cfg, loss_weights=_lw, metrics=_met,
        )
        print(f"[Model] Encoder frozen for first {args.freeze_encoder_epochs} epochs")

    if args.resume_weights:
        if os.path.exists(args.resume_weights):
            model.load_weights(args.resume_weights)
            print(f"[Resume] Loaded weights from {args.resume_weights}")
        else:
            print(f"[Resume] WARNING: {args.resume_weights} not found — starting from scratch")
    elif args.resume:
        resume_path = os.path.join(args.output_dir, "best_model.weights.h5")
        if os.path.exists(resume_path):
            model.load_weights(resume_path)
            print(f"[Resume] Loaded weights from {resume_path}")
        else:
            print(f"[Resume] WARNING: {resume_path} not found — starting from scratch")

    # 6. Callbacks
    best_weights_path = os.path.join(args.output_dir, "best_model.weights.h5")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_weights_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=25,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=50,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(args.output_dir, "training_log.csv"),
            append=True,
        ),
    ]

    # 7. Train
    print(f"\n[Train] epochs={args.epochs}  batch={args.batch_size}  "
          f"steps/epoch~{len(oversampled)//args.batch_size}")

    phase1_epochs = (
        args.freeze_encoder_epochs
        if encoder is not None and args.freeze_encoder_epochs > 0
        else args.epochs
    )

    def _unfreeze_encoder_for_phase2(enc, freeze_bn):
        """Unfreeze encoder conv weights, optionally keeping BN frozen."""
        enc.trainable = True
        if freeze_bn:
            bn_frozen = 0
            for layer in enc.layers:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = False
                    bn_frozen += 1
            print(f"[Train] BN layers in encoder frozen ({bn_frozen} layers) — using ImageNet BN stats")
        else:
            print(f"[Train] Encoder fully unfrozen including BN layers")

    def _build_phase2_loss():
        loss2 = (make_focal_tversky_loss(args.panel_weight)
                 if args.loss == "focal_tversky"
                 else make_combo_loss(args.panel_weight))
        if is_deep_sup:
            lc = {"main_output": loss2, "aux_d3_output": loss2, "aux_d4_output": loss2}
            lw = {"main_output": 1.0, "aux_d3_output": 0.4, "aux_d4_output": 0.2}
            mt = {"main_output": [panel_iou]}
        else:
            lc, lw, mt = loss2, None, [panel_iou]
        return lc, lw, mt

    # --skip_phase1: jump directly to Phase 2 (used when resuming after OOM mid-Phase-2)
    if args.skip_phase1 and encoder is not None and args.freeze_encoder_epochs > 0:
        start_ep = args.initial_epoch if args.initial_epoch is not None else phase1_epochs
        print(f"\n[Train] --skip_phase1: jumping to Phase 2 at epoch {start_ep}, lr={args.lr / 3:.2e}")
        _unfreeze_encoder_for_phase2(encoder, args.freeze_encoder_bn)
        _lc2, _lw2, _mt2 = _build_phase2_loss()
        _opt2 = tf.keras.optimizers.Adam(learning_rate=args.lr / 3)
        if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
            _opt2 = tf.keras.mixed_precision.LossScaleOptimizer(_opt2)
        model.compile(
            optimizer=_opt2,
            loss=_lc2, loss_weights=_lw2, metrics=_mt2,
        )
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            initial_epoch=start_ep,
            callbacks=callbacks,
            verbose=1,
        )
    else:
        # Phase 1 — frozen encoder warm-up (U-Net) or full training (Fast SCNN)
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=phase1_epochs,
            callbacks=callbacks,
            verbose=1,
        )

        # Phase 2 — unfreeze encoder and fine-tune with lower LR (U-Net only)
        if encoder is not None and args.freeze_encoder_epochs > 0 and args.epochs > args.freeze_encoder_epochs:
            print(f"\n[Train] Unfreezing encoder — fine-tuning with lr={args.lr / 3:.2e}")
            _unfreeze_encoder_for_phase2(encoder, args.freeze_encoder_bn)
            _lc2, _lw2, _mt2 = _build_phase2_loss()
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr / 3),
                loss=_lc2, loss_weights=_lw2, metrics=_mt2,
            )
            history2 = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=args.epochs,
                initial_epoch=phase1_epochs,
                callbacks=callbacks,
                verbose=1,
            )
            for k in history.history:
                history.history[k].extend(history2.history.get(k, []))

    # 8. Save curves and report
    save_curves(history, args.output_dir)

    best_val_loss = min(history.history.get("val_loss", [float("inf")]))
    iou_key = next((k for k in history.history if "iou" in k and not k.startswith("val_")), None)
    best_val_iou  = max(history.history.get("val_" + iou_key, [0.0])) if iou_key else 0.0
    print(f"\n[Done] Best val loss : {best_val_loss:.4f}")
    print(f"       Best val IoU  : {best_val_iou:.4f}")
    print(f"       Weights saved : {best_weights_path}")
    model_type = "unet_resnet50" if args.model == "unet_resnet50" else "fast_scnn_v2"
    print(f"\nTo run inference with these weights:")
    print(f"  python detect_geotiff.py ortho.tif "
          f"--model_name {best_weights_path} --model_type {model_type}")


if __name__ == "__main__":
    main()
