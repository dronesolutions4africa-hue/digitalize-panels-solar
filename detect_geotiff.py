"""
detect_geotiff.py — Solar panel detection on GeoTIFF orthomosaics

Usage:
  python detect_geotiff.py \
    --input  path/to/rgb_orthomosaic.tif \
    --output_shp path/to/output/panels.shp \
    --output_img path/to/output/panels_viz.tif \
    --model_type fast_scnn_v2 \
    --model_name fast_scnn_2.h5 \
    --tile_size 256 \
    --overlap 32 \
    --min_area_m2 2.0

Model types:
  fast_scnn_v2    (recommended, IoU 0.8224)
  seg_resnet_v2
  segnet_4ed
  segnet_original

Weight files in trained_models/:
  fast_scnn_2.h5, fast_scnn_1.h5, fast_scnn_original.h5
  seg_resnet_1.h5, seg_resnet_2.h5, segnet_1.h5, segnet_2.h5, segnet_original.h5

Large raster handling (>50 MP):
  Switches automatically to non-overlapping tile mode with disk-backed binary mask
  (np.memmap). Visualization written in 2 000-row chunks. Vectorization done in
  5 000-row strips via rasterio.features.shapes. See ERROR_LOG ERROR-004.

NOTE: solar_panel_detection.py has no exportable build_model function — model
creation is inlined here per ERROR_LOG ERROR-001.
"""

import argparse
import logging
import math
import os
import shutil
import sys
import tempfile

import cv2
import geopandas as gpd
import numpy as np
import rasterio
import rasterio.enums
import rasterio.features
from affine import Affine
from rasterio.features import shapes as rio_shapes
from rasterio.windows import Window
from shapely.geometry import Polygon, shape
from tqdm import tqdm

# model_list imports — inline build_model, see ERROR_LOG ERROR-001
from model_list import fast_scnn_2 as _fast_scnn_2
from model_list import segnet_0 as _segnet_0
from model_list import segnet_1 as _segnet_1
from model_list import segnet_3 as _segnet_3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("detect_geotiff")

# Rasters larger than this use non-overlapping tiles + memmap binary mask
LARGE_RASTER_MP = 50  # megapixels


# ── Model construction ─────────────────────────────────────────────────────────

def build_model(model_type: str, tile_size: int = 256):
    """
    Build and return a compiled Keras model.
    solar_panel_detection.py creates models inline — no exportable build_model
    exists (ERROR_LOG ERROR-001). This function replicates that logic.
    """
    input_shape = (tile_size, tile_size, 3)
    kwargs = dict(input_shape=input_shape, batch_size=1, n_labels=2, model_summary=False)

    if model_type == "fast_scnn_v2":
        return _fast_scnn_2.fast_scnn_v2(**kwargs)
    elif model_type == "seg_resnet_v2":
        return _segnet_3.segnet_resnet_v2(**kwargs)
    elif model_type == "segnet_4ed":
        return _segnet_1.segnet_4_encoder_decoder(**kwargs)
    elif model_type == "segnet_original":
        return _segnet_0.segnet_original(**kwargs)
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            "Choose: fast_scnn_v2 | seg_resnet_v2 | segnet_4ed | segnet_original"
        )


def load_model(model_type: str, model_name: str, tile_size: int = 256):
    """Build model architecture and load pre-trained weights (.h5)."""
    logger.info(f"Loading model: type={model_type}, weights={model_name}")
    model = build_model(model_type, tile_size)

    candidates = [
        model_name,
        os.path.join("trained_models", model_name),
        model_name + ".h5",
        os.path.join("trained_models", model_name + ".h5"),
    ]
    weights_path = next((p for p in candidates if os.path.exists(p)), None)
    if weights_path is None:
        raise FileNotFoundError(
            f"Weights not found for '{model_name}'. "
            "Available in trained_models/: fast_scnn_2.h5, fast_scnn_1.h5, fast_scnn_original.h5, "
            "seg_resnet_1.h5, seg_resnet_2.h5, segnet_1.h5, segnet_2.h5, segnet_original.h5"
        )

    model.load_weights(weights_path)
    logger.info(f"Weights loaded from {weights_path}")
    return model


# ── Normalisation (non-RGB / float rasters) ────────────────────────────────────

def compute_norm_params(src):
    """
    Compute 2nd-98th percentile stretch for band 1 via decimated read.
    Returns (p2, p98) so the full raster is never loaded at once.
    """
    decimate = 16
    small = src.read(
        1,
        out_shape=(1, max(1, src.height // decimate), max(1, src.width // decimate)),
        resampling=rasterio.enums.Resampling.nearest,
    ).astype(np.float32).ravel()
    valid = small[np.isfinite(small) & (small != 0)]
    if len(valid) == 0:
        valid = small
    p2, p98 = np.percentile(valid, [2, 98])
    if p98 == p2:
        p98 = p2 + 1.0
    logger.info(f"Norm params (2nd-98th pct of band 1): [{p2:.3f}, {p98:.3f}]")
    return float(p2), float(p98)


# ── Tiling generator ───────────────────────────────────────────────────────────

def tile_geotiff(src, tile_size: int, overlap: int, norm_params=None):
    """
    Generator — yields (window, tile_array, (orig_w, orig_h)) one tile at a time.
    tile_array is (tile_size, tile_size, 3) float32 in [0, 1], padded at edges.
    Handles 3-band uint8 RGB (divide by 255) and any other format (band-1 stretch).
    """
    step = tile_size - overlap
    width, height = src.width, src.height
    is_rgb = src.count >= 3 and src.dtypes[0] == "uint8"
    logger.info(
        f"Raster: {width}x{height}px — tile={tile_size}, overlap={overlap}, step={step} | "
        f"mode={'RGB/255' if is_rgb else 'band1-stretch'}"
    )

    p2, p98 = norm_params if norm_params else (0.0, 255.0)
    scale = max(p98 - p2, 1.0)

    for row_off in range(0, height, step):
        for col_off in range(0, width, step):
            orig_w = min(tile_size, width - col_off)
            orig_h = min(tile_size, height - row_off)
            win = Window(col_off, row_off, orig_w, orig_h)

            if is_rgb:
                tile = src.read([1, 2, 3], window=win)             # (3, h, w) uint8
                tile = np.moveaxis(tile, 0, -1).astype(np.float32) # (h, w, 3)
                tile /= 255.0
            else:
                band1 = src.read(1, window=win).astype(np.float32) # (h, w)
                band1 = np.clip((band1 - p2) / scale, 0.0, 1.0)
                tile = np.stack([band1, band1, band1], axis=-1)     # (h, w, 3)

            if orig_h < tile_size or orig_w < tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=np.float32)
                padded[:orig_h, :orig_w] = tile
                tile = padded

            yield win, tile, (orig_w, orig_h)


# ── Inference ──────────────────────────────────────────────────────────────────

def run_inference(
    model, src, tile_size: int, overlap: int, batch_size: int = 8, threshold: float = 0.5
):
    """
    Run tiled inference and return a binary mask (uint8, 0 or 255).

    Small rasters (<= LARGE_RASTER_MP MP):
      Overlapping tiles with weighted-average blending. Float32 arrays in RAM.

    Large rasters (> LARGE_RASTER_MP MP):
      Non-overlapping tiles (overlap forced to 0) to avoid 10+ GB float32
      accumulation arrays. Binary mask stored as np.memmap on disk (~2.5 GB
      for a 48kx51k raster). See ERROR_LOG ERROR-004.

    Returns: (binary_mask, tmpdir)
      tmpdir: str path to clean up, or None if no temp dir was used.
    """
    height, width = src.height, src.width
    n_pixels = height * width
    is_large = n_pixels > LARGE_RASTER_MP * 1_000_000

    if is_large:
        effective_overlap = 0   # no accumulation arrays needed
        logger.info(
            f"Large raster ({width}x{height}, {n_pixels/1e6:.0f} MP) — "
            f"non-overlapping tile mode (overlap forced to 0). "
            f"Binary mask will be stored as memmap on disk."
        )
    else:
        effective_overlap = overlap

    step = tile_size - effective_overlap
    n_y = len(range(0, height, step))
    n_x = len(range(0, width, step))
    n_tiles = n_y * n_x
    n_batches = math.ceil(n_tiles / batch_size)
    est_min = n_batches / 2.5 / 60
    logger.info(
        f"Tiles: {n_tiles} ({n_batches} batches of {batch_size}) — "
        f"estimated ~{est_min:.0f} min at 2.5 batch/s on CPU"
    )

    norm_params = (
        None if (src.count >= 3 and src.dtypes[0] == "uint8")
        else compute_norm_params(src)
    )

    # Allocate output arrays
    tmpdir = None
    if is_large:
        tmpdir = tempfile.mkdtemp(prefix="detect_geotiff_")
        binary_mask = np.memmap(
            os.path.join(tmpdir, "binary.dat"),
            dtype=np.uint8, mode="w+", shape=(height, width),
        )
        logger.info(
            f"memmap binary mask: {tmpdir}/binary.dat "
            f"({n_pixels / 1e9:.2f} GB on disk)"
        )
    else:
        full_mask = np.zeros((height, width), dtype=np.float32)
        weight_map = np.zeros((height, width), dtype=np.float32)

    tiles_gen = tile_geotiff(src, tile_size, effective_overlap, norm_params)
    prob_max = 0.0
    panel_px = 0

    logger.info(f"Running inference on {n_tiles} tiles (batch_size={batch_size})...")

    with tqdm(total=n_batches, desc="inference") as pbar:
        while True:
            batch_metas, batch_imgs = [], []
            for _ in range(batch_size):
                item = next(tiles_gen, None)
                if item is None:
                    break
                win, tile, orig_wh = item
                batch_metas.append((win, orig_wh))
                batch_imgs.append(tile)
            if not batch_imgs:
                break

            batch_arr = np.stack(batch_imgs, axis=0)  # (B, 256, 256, 3)
            preds = model.predict(batch_arr, verbose=0)

            # 2-class softmax → panel probability; sigmoid fallback
            if preds.shape[-1] == 2:
                preds = preds[..., 1]
            else:
                preds = preds[..., 0]

            prob_max = max(prob_max, float(preds.max()))

            for i, (win, (orig_w, orig_h)) in enumerate(batch_metas):
                col, row = int(win.col_off), int(win.row_off)
                patch = preds[i, :orig_h, :orig_w]

                if is_large:
                    bm = (patch >= threshold).astype(np.uint8) * 255
                    binary_mask[row: row + orig_h, col: col + orig_w] = bm
                    panel_px += int(np.sum(bm == 255))
                else:
                    full_mask[row: row + orig_h, col: col + orig_w] += patch
                    weight_map[row: row + orig_h, col: col + orig_w] += 1.0

            pbar.update(1)

    if is_large:
        binary_mask.flush()
    else:
        weight_map = np.where(weight_map == 0, 1.0, weight_map)
        full_mask /= weight_map
        prob_max = float(full_mask.max())
        binary_mask = (full_mask >= threshold).astype(np.uint8) * 255
        panel_px = int(np.sum(binary_mask == 255))

    logger.info(f"Max model probability: {prob_max:.3f}")
    logger.info(
        f"Binary mask ready — {panel_px} panel pixels "
        f"({panel_px / n_pixels * 100:.2f}% of raster)"
    )
    return binary_mask, tmpdir


# ── Vectorisation ──────────────────────────────────────────────────────────────

def mask_to_polygons(
    binary_mask: np.ndarray,
    transform,
    crs,
    min_area_m2: float,
) -> gpd.GeoDataFrame:
    """
    Convert binary mask to georeferenced polygons.

    Large rasters: rasterio.features.shapes in 5 000-row strips (no full-array
    load). Small rasters: morphological cleanup + cv2 contours.
    """
    height, width = binary_mask.shape
    is_large = (height * width) > LARGE_RASTER_MP * 1_000_000
    logger.info("Vectorizing mask to polygons...")
    geometries = []

    if is_large:
        chunk_rows = 5000
        for r in range(0, height, chunk_rows):
            h = min(chunk_rows, height - r)
            chunk = np.array(binary_mask[r: r + h], dtype=np.uint8)
            if not chunk.any():
                continue
            # Offset the transform so strip pixel (0,0) maps to original (row=r, col=0)
            strip_transform = transform * Affine.translation(0, r)
            for geom_dict, val in rio_shapes(
                chunk, mask=(chunk == 255), transform=strip_transform
            ):
                poly = shape(geom_dict)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if not poly.is_empty:
                    geometries.append(poly)
        logger.info(f"Polygons (rasterio.features.shapes): {len(geometries)}")

    else:
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        logger.info(f"Raw contours: {len(contours)}")
        skipped = 0
        for cnt in contours:
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) < 3:
                skipped += 1
                continue
            pixel_coords = approx.squeeze().tolist()
            if not isinstance(pixel_coords[0], list):
                skipped += 1
                continue
            geo_coords = [
                rasterio.transform.xy(transform, py, px, offset="center")
                for px, py in pixel_coords
            ]
            if geo_coords[0] != geo_coords[-1]:
                geo_coords.append(geo_coords[0])
            if len(geo_coords) < 4:
                skipped += 1
                continue
            poly = Polygon(geo_coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                skipped += 1
                continue
            geometries.append(poly)
        logger.info(f"Polygons extracted: {len(geometries)}, skipped: {skipped}")

    if not geometries:
        logger.warning("No polygons extracted from mask")
        return gpd.GeoDataFrame(columns=["geometry", "area_m2"], crs=crs)

    gdf = gpd.GeoDataFrame(geometry=geometries, crs=crs)

    if crs.is_projected:
        gdf["area_m2"] = gdf.geometry.area
    else:
        gdf["area_m2"] = gdf.geometry.area * (111_320.0 ** 2)

    before = len(gdf)
    gdf = gdf[gdf["area_m2"] >= min_area_m2].reset_index(drop=True)
    logger.info(
        f"After area filter (>= {min_area_m2} m2): {len(gdf)} polygons "
        f"(removed {before - len(gdf)})"
    )
    return gdf


# ── Output writers ─────────────────────────────────────────────────────────────

def save_shapefile(gdf: gpd.GeoDataFrame, output_path: str) -> None:
    """Save GeoDataFrame as Shapefile, preserving the raster CRS."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    out = gdf.copy()
    out["panel_id"] = range(1, len(out) + 1)
    out.to_file(output_path, driver="ESRI Shapefile")
    logger.info(f"Shapefile saved: {output_path} ({len(out)} panels)")
    logger.info(f"  CRS:  {out.crs}")
    logger.info(f"  Bbox: {out.total_bounds}")


def save_visualization(src, binary_mask: np.ndarray, output_path: str) -> None:
    """
    Write a 3-band GeoTIFF with detected panels overlaid in red.
    Writes in 2 000-row chunks to handle rasters of any size.
    Handles both RGB uint8 and non-RGB / float32 inputs.
    """
    logger.info(f"Saving visualization to {output_path}...")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    height, width = src.height, src.width
    is_rgb = src.count >= 3 and src.dtypes[0] == "uint8"

    p2, p98, scale = 0.0, 255.0, 255.0
    if not is_rgb:
        small = src.read(
            1,
            out_shape=(1, min(500, height), min(500, width)),
            resampling=rasterio.enums.Resampling.nearest,
        ).astype(np.float32).ravel()
        valid = small[np.isfinite(small) & (small != 0)]
        if len(valid) > 0:
            p2 = float(np.percentile(valid, 2))
            p98 = float(np.percentile(valid, 98))
        scale = max(p98 - p2, 1.0)

    chunk_rows = 2000
    # BIGTIFF=YES required when output exceeds 4 GB (e.g. 48k×51k×3 = 7.4 GB)
    with rasterio.open(
        output_path, "w",
        driver="GTiff",
        height=height, width=width,
        count=3, dtype="uint8",
        crs=src.crs, transform=src.transform,
        compress="lzw",
        tiled=True, blockxsize=512, blockysize=512,
        BIGTIFF="YES",
    ) as dst:
        for r in range(0, height, chunk_rows):
            h = min(chunk_rows, height - r)
            win = Window(0, r, width, h)

            if is_rgb:
                rgb = src.read([1, 2, 3], window=win)
                viz = rgb.astype(np.float32)
            else:
                b1 = src.read(1, window=win).astype(np.float32)
                gray = np.clip((b1 - p2) / scale * 255.0, 0, 255)
                viz = np.stack([gray, gray, gray], axis=0)

            panel = np.array(binary_mask[r: r + h]) == 255
            viz[0][panel] = viz[0][panel] * 0.5 + 127.5
            viz[1][panel] = viz[1][panel] * 0.5
            viz[2][panel] = viz[2][panel] * 0.5
            viz = np.clip(viz, 0, 255).astype(np.uint8)
            dst.write(viz, window=win)

    logger.info(f"Visualization saved: {output_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Solar panel detection on GeoTIFF orthomosaics -> Shapefile + viz GeoTIFF"
    )
    parser.add_argument("--input",       required=True, help="Input GeoTIFF path")
    parser.add_argument("--output_shp",  required=True, help="Output shapefile (.shp)")
    parser.add_argument("--output_img",  required=True, help="Output visualization GeoTIFF")
    parser.add_argument(
        "--model_type", default="fast_scnn_v2",
        choices=["fast_scnn_v2", "seg_resnet_v2", "segnet_4ed", "segnet_original"],
    )
    parser.add_argument(
        "--model_name", default="fast_scnn_2.h5",
        help="Weight file in trained_models/ (see ERROR_LOG ERROR-002)",
    )
    parser.add_argument("--tile_size",   type=int,   default=256)
    parser.add_argument("--overlap",     type=int,   default=32,
                        help="Overlap in px (ignored for rasters >50 MP, forced to 0)")
    parser.add_argument("--batch_size",  type=int,   default=8,
                        help="Inference batch size — reduce to 4 or 2 on OOM")
    parser.add_argument("--min_area_m2", type=float, default=2.0,
                        help="Min panel area in m2 to retain")
    parser.add_argument("--threshold",   type=float, default=0.5,
                        help="Probability threshold for panel class (0-1). "
                             "Higher = fewer but more precise detections.")
    parser.add_argument("--save_mask_tif", default=None,
                        help="Optional path to save the raw binary mask as a GeoTIFF "
                             "(single band uint8, 0/255) for watershed post-processing.")
    args = parser.parse_args()

    logger.info("Solar panel detection pipeline started")
    logger.info(f"  Input:      {args.input}")
    logger.info(f"  Output SHP: {args.output_shp}")
    logger.info(f"  Output IMG: {args.output_img}")
    logger.info(f"  Model:      {args.model_type} / {args.model_name}")
    logger.info(f"  Tile:       {args.tile_size}px, overlap={args.overlap}px")

    with rasterio.open(args.input) as src:
        logger.info(
            f"Raster opened — CRS: {src.crs} | "
            f"Size: {src.width}x{src.height}px ({src.width*src.height/1e6:.0f} MP) | "
            f"Bands: {src.count} | Dtypes: {src.dtypes} | "
            f"Res: {src.res[0]:.6f}x{src.res[1]:.6f} units | "
            f"Bounds: {src.bounds}"
        )
        if src.count < 3:
            logger.warning(
                f"Input has only {src.count} band(s) — expected RGB. "
                "Band 1 will be replicated to 3 channels with percentile stretch."
            )

        model = load_model(args.model_type, args.model_name, args.tile_size)
        logger.info(f"  Threshold:  {args.threshold}")
        binary_mask, tmpdir = run_inference(
            model, src, args.tile_size, args.overlap, args.batch_size, args.threshold
        )

        try:
            if args.save_mask_tif:
                logger.info(f"Saving binary mask GeoTIFF → {args.save_mask_tif}")
                os.makedirs(os.path.dirname(os.path.abspath(args.save_mask_tif)), exist_ok=True)
                with rasterio.open(
                    args.save_mask_tif, "w",
                    driver="GTiff", height=src.height, width=src.width,
                    count=1, dtype="uint8",
                    crs=src.crs, transform=src.transform,
                    compress="lzw", tiled=True, blockxsize=512, blockysize=512,
                    BIGTIFF="YES",
                ) as dst_mask:
                    chunk = 2000
                    for r in range(0, src.height, chunk):
                        h = min(chunk, src.height - r)
                        dst_mask.write(
                            np.array(binary_mask[r:r+h]).reshape(1, h, src.width),
                            window=rasterio.windows.Window(0, r, src.width, h),
                        )
                logger.info("Binary mask GeoTIFF saved.")

            gdf = mask_to_polygons(
                binary_mask, src.transform, src.crs, args.min_area_m2
            )
            if len(gdf) > 0:
                save_shapefile(gdf, args.output_shp)
            else:
                logger.warning("No panels detected — shapefile not saved")

            save_visualization(src, binary_mask, args.output_img)

        finally:
            if tmpdir and os.path.exists(tmpdir):
                # On Windows, memmap file handles must be released before rmtree
                del binary_mask
                import gc; gc.collect()
                shutil.rmtree(tmpdir, ignore_errors=True)
                logger.info(f"Cleaned up temp dir: {tmpdir}")

    logger.info(f"Pipeline complete — {len(gdf)} panels detected")


if __name__ == "__main__":
    main()
