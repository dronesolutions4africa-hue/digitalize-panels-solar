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
    p.add_argument("--ortho",       default=ORTHO_PATH)
    p.add_argument("--shp",         default=SHP_PATH)
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
    p.add_argument("--model", default="fast_scnn_2",
                   choices=["fast_scnn_2", "unet_resnet50"],
                   help="Model architecture to train")
    p.add_argument("--freeze_encoder_epochs", type=int, default=20,
                   help="(unet_resnet50 only) Epochs with frozen ResNet50 encoder before fine-tuning")
    p.add_argument("--resume", action="store_true",
                   help="Load best_model.weights.h5 from output_dir before training")
    return p.parse_args()


# ── GPU ───────────────────────────────────────────────────────────────────────
def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print(f"[GPU] {len(gpus)} device(s): {[g.name for g in gpus]}")
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


def make_dataset(rgb: np.ndarray, mask: np.ndarray,
                 indices, tile_size: int, batch_size: int,
                 augment: bool = False, shuffle: bool = False):
    """
    Build a tf.data.Dataset that extracts (img, one-hot-mask) tiles on the fly
    from the in-RAM numpy arrays `rgb` (H,W,3 uint8) and `mask` (H,W uint8).
    """
    rows = np.array([r for r, c in indices], dtype=np.int32)
    cols = np.array([c for r, c in indices], dtype=np.int32)

    def load_tile(r_tensor, c_tensor):
        r = int(r_tensor.numpy())
        c = int(c_tensor.numpy())
        img = rgb[r:r+tile_size, c:c+tile_size].astype(np.float32) / 255.0
        msk = mask[r:r+tile_size, c:c+tile_size]
        msk_oh = np.stack(
            [(msk == 0).astype(np.float32),
             (msk == 255).astype(np.float32)],
            axis=-1,
        )
        return img, msk_oh

    def map_fn(r, c):
        img, msk = tf.py_function(load_tile, [r, c], [tf.float32, tf.float32])
        img.set_shape((tile_size, tile_size, 3))
        msk.set_shape((tile_size, tile_size, 2))
        return img, msk

    ds = tf.data.Dataset.from_tensor_slices((rows, cols))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(indices), reshuffle_each_iteration=True)
    ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
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
    """
    Returns a loss = weighted categorical BCE + Dice on the panel class.
    panel_weight upweights panel pixels in the BCE term to fight class imbalance.
    """
    def combo_loss(y_true, y_pred):
        # Pixel-level weight map: background=1, panel=panel_weight
        weight_map = y_true[..., 0] * 1.0 + y_true[..., 1] * panel_weight
        bce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        w_bce = tf.reduce_mean(bce * weight_map)

        # Dice loss on the panel class (index 1)
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


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(tile_size: int, lr: float, panel_weight: float, model_name: str = "fast_scnn_2"):
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
    else:
        raise ValueError(f"Unknown model: {model_name!r}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=make_combo_loss(panel_weight),
        metrics=[panel_iou],
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

    # 1. Load orthomosaic into RAM (uint8 — ~950 MB for 18 695×16 883)
    print(f"\n[Data] Loading {args.ortho} …")
    t0 = time.time()
    with rasterio.open(args.ortho) as src:
        # Read RGB bands only (ignore alpha / extra bands)
        band_indices = [1, 2, 3] if src.count >= 3 else [1, 1, 1]
        rgb = src.read(band_indices)              # (3, H, W) uint8
        rgb = np.moveaxis(rgb, 0, -1)            # (H, W, 3)
        height, width = src.height, src.width
    print(f"       {width}×{height} px loaded in {time.time()-t0:.1f}s")

    # 2. Rasterize shapefile labels
    print(f"\n[Data] Rasterizing {args.shp} …")
    mask = rasterize_labels(args.ortho, args.shp)

    # 3. Tile indices (random panel-aware split)
    train_idx, val_idx = get_tile_indices(
        height, width, args.tile_size, args.stride, args.val_frac, mask=mask
    )

    # Oversample panel tiles to balance each training batch
    import random as _random
    panel_train = [(r, c) for r, c in train_idx
                   if mask[r:r+args.tile_size, c:c+args.tile_size].max() > 0]
    bg_train    = [(r, c) for r, c in train_idx
                   if mask[r:r+args.tile_size, c:c+args.tile_size].max() == 0]
    oversampled = panel_train * args.panel_oversample + bg_train
    _random.Random(42).shuffle(oversampled)
    print(f"[Data] Tiles — train: {len(train_idx)} raw "
          f"({len(panel_train)} panel × {args.panel_oversample} + {len(bg_train)} bg "
          f"= {len(oversampled)} oversampled), val: {len(val_idx)}")
    est_panel_per_batch = args.batch_size * len(panel_train) * args.panel_oversample / len(oversampled)
    print(f"       Expected panel tiles/batch: {est_panel_per_batch:.1f} / {args.batch_size}")

    # 4. TF Datasets
    train_ds = make_dataset(
        rgb, mask, oversampled,
        args.tile_size, args.batch_size,
        augment=True, shuffle=True,
    )
    val_ds = make_dataset(
        rgb, mask, val_idx,
        args.tile_size, args.batch_size,
        augment=False, shuffle=False,
    )

    # 5. Model
    print(f"\n[Model] {args.model}  input={args.tile_size}×{args.tile_size}  "
          f"lr={args.lr}  panel_weight={args.panel_weight}")
    model, encoder = build_model(args.tile_size, args.lr, args.panel_weight, args.model)
    model.summary(line_length=90)

    # Freeze ResNet50 encoder during warm-up phase (U-Net only)
    if encoder is not None and args.freeze_encoder_epochs > 0:
        encoder.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
            loss=make_combo_loss(args.panel_weight),
            metrics=[panel_iou],
        )
        print(f"[Model] Encoder frozen for first {args.freeze_encoder_epochs} epochs")

    if args.resume:
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
            patience=15,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=40,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(args.output_dir, "training_log.csv")
        ),
    ]

    # 7. Train
    print(f"\n[Train] epochs={args.epochs}  batch={args.batch_size}  "
          f"steps/epoch~{len(oversampled)//args.batch_size}")

    # Phase 1 — frozen encoder warm-up (U-Net) or full training (Fast SCNN)
    phase1_epochs = (
        args.freeze_encoder_epochs
        if encoder is not None and args.freeze_encoder_epochs > 0
        else args.epochs
    )
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase1_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Phase 2 — unfreeze encoder and fine-tune with lower LR (U-Net only)
    if encoder is not None and args.freeze_encoder_epochs > 0 and args.epochs > args.freeze_encoder_epochs:
        print(f"\n[Train] Unfreezing encoder — fine-tuning with lr={args.lr / 10:.2e}")
        encoder.trainable = True
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr / 10),
            loss=make_combo_loss(args.panel_weight),
            metrics=[panel_iou],
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
