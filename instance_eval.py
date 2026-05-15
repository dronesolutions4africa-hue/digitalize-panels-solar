"""
instance_eval.py — Évaluation au niveau instance (panneau individuel).

Pour chacun des N panneaux du shapefile GT, on vérifie si le masque prédit
couvre ≥ iou_thresh de sa surface → panneau "détecté".

Sorties :
  - Métriques : rappel / précision / F1 par panneau
  - Shapefile des panneaux détectés (géométries GT)
  - Shapefile des panneaux manqués
  - Visualisation GeoTIFF (vert=détecté, rouge=manqué)

Usage :
    python instance_eval.py \
        --mask_tif   output/patisen/binary_mask.tif \
        --gt_shp     Data/Panneaux_Patisen.shp \
        --ortho      Data/Orthomosaic_Patisen.tif \
        --output_dir output/patisen/instance_eval \
        --coverage   0.5
"""

import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import rasterio
import rasterio.features
import rasterio.windows
import geopandas as gpd
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mask_tif",   required=True)
    p.add_argument("--gt_shp",     required=True)
    p.add_argument("--ortho",      required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--coverage",   type=float, default=0.5,
                   help="Fraction minimale du panneau couverte par la prédiction")
    return p.parse_args()


def evaluate_instances(mask_tif, gt_shp, coverage_thresh):
    """
    Pour chaque polygone GT, rasterise localement et compare au masque prédit.
    Retourne un tableau booléen : detected[i] = True si panneau i couvert.
    """
    with rasterio.open(mask_tif) as src:
        transform = src.transform
        crs       = src.crs
        H, W      = src.height, src.width
        px_w      = abs(transform.a)  # degrés/pixel en x
        px_h      = abs(transform.e)  # degrés/pixel en y

    gdf = gpd.read_file(gt_shp)
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    detected = np.zeros(len(gdf), dtype=bool)

    with rasterio.open(mask_tif) as src:
        for i, row in enumerate(tqdm(gdf.itertuples(), total=len(gdf),
                                      desc="éval instances")):
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            # Boîte englobante du panneau en pixels
            minx, miny, maxx, maxy = geom.bounds
            col0 = max(0, int((minx - transform.c) / px_w) - 1)
            row0 = max(0, int((transform.f - maxy) / px_h) - 1)
            col1 = min(W, int((maxx - transform.c) / px_w) + 2)
            row1 = min(H, int((transform.f - miny) / px_h) + 2)

            if col1 <= col0 or row1 <= row0:
                continue

            h_local = row1 - row0
            w_local = col1 - col0
            win = rasterio.windows.Window(col0, row0, w_local, h_local)

            # Rasterise ce seul panneau localement
            local_transform = rasterio.windows.transform(win, transform)
            gt_local = rasterio.features.rasterize(
                [(geom, 1)],
                out_shape=(h_local, w_local),
                transform=local_transform,
                fill=0, dtype=np.uint8, all_touched=True,
            )

            # Lecture du masque prédit dans cette fenêtre
            pred_local = src.read(1, window=win)  # 0 ou 255

            gt_pixels   = int(gt_local.sum())
            if gt_pixels == 0:
                continue

            covered = int(((pred_local == 255) & (gt_local == 1)).sum())
            if covered / gt_pixels >= coverage_thresh:
                detected[i] = True

    return gdf, detected, crs


def save_shapefiles(gdf, detected, output_dir, crs):
    os.makedirs(output_dir, exist_ok=True)

    det_gdf  = gdf[detected].copy().reset_index(drop=True)
    miss_gdf = gdf[~detected].copy().reset_index(drop=True)

    det_gdf["status"]  = "detected"
    miss_gdf["status"] = "missed"

    det_path  = os.path.join(output_dir, "detected_panels.shp")
    miss_path = os.path.join(output_dir, "missed_panels.shp")

    det_gdf.to_file(det_path,  driver="ESRI Shapefile")
    miss_gdf.to_file(miss_path, driver="ESRI Shapefile")

    print(f"  Détectés  → {det_path}  ({len(det_gdf)} panneaux)")
    print(f"  Manqués   → {miss_path} ({len(miss_gdf)} panneaux)")
    return det_gdf, miss_gdf


def save_viz(ortho, det_gdf, miss_gdf, mask_tif, output_dir):
    viz_path = os.path.join(output_dir, "instance_eval_viz.tif")
    print(f"[Viz] Génération {viz_path}…")

    with rasterio.open(mask_tif) as msrc:
        transform, crs = msrc.transform, msrc.crs
        H, W = msrc.height, msrc.width

        # Rasterise détectés (vert) et manqués (rouge)
        shapes_det  = [(g, 2) for g in det_gdf.geometry  if g and not g.is_empty]
        shapes_miss = [(g, 1) for g in miss_gdf.geometry if g and not g.is_empty]

        label_mask = rasterio.features.rasterize(
            shapes_miss + shapes_det,
            out_shape=(H, W), transform=transform,
            fill=0, dtype=np.uint8, all_touched=True,
        )

    with rasterio.open(ortho) as src:
        os.makedirs(output_dir, exist_ok=True)
        with rasterio.open(
            viz_path, "w", driver="GTiff",
            height=H, width=W, count=3, dtype="uint8",
            crs=src.crs, transform=src.transform,
            compress="lzw", tiled=True, blockxsize=512, blockysize=512,
            BIGTIFF="YES",
        ) as dst:
            chunk = 2000
            for r in tqdm(range(0, H, chunk), desc="viz"):
                h = min(chunk, H - r)
                win = rasterio.windows.Window(0, r, W, h)
                rgb = src.read([1, 2, 3], window=win).astype(np.float32)
                lbl = label_mask[r:r+h]

                missed   = lbl == 1
                detected = lbl == 2

                # Manqués → rouge
                rgb[0][missed] = rgb[0][missed] * 0.3 + 178
                rgb[1][missed] = rgb[1][missed] * 0.3
                rgb[2][missed] = rgb[2][missed] * 0.3

                # Détectés → vert
                rgb[0][detected] = rgb[0][detected] * 0.3
                rgb[1][detected] = rgb[1][detected] * 0.3 + 178
                rgb[2][detected] = rgb[2][detected] * 0.3

                dst.write(np.clip(rgb, 0, 255).astype(np.uint8), window=win)

    print(f"[Viz] Sauvegardé → {viz_path}")


def main():
    args = parse_args()
    t0 = time.time()

    print(f"\n[Config] coverage_thresh = {args.coverage*100:.0f}% du panneau couvert")
    print(f"         mask_tif = {args.mask_tif}")

    gdf, detected, crs = evaluate_instances(
        args.mask_tif, args.gt_shp, args.coverage
    )

    n_total    = len(gdf)
    n_detected = int(detected.sum())
    n_missed   = n_total - n_detected
    recall     = n_detected / n_total

    print(f"\n{'═'*52}")
    print(f"  Panneaux GT total         : {n_total:>6}")
    print(f"  Panneaux détectés (≥{args.coverage*100:.0f}%) : {n_detected:>6}  ({recall*100:.1f}%)")
    print(f"  Panneaux manqués          : {n_missed:>6}  ({(1-recall)*100:.1f}%)")
    print(f"{'═'*52}\n")

    det_gdf, miss_gdf = save_shapefiles(gdf, detected, args.output_dir, crs)
    save_viz(args.ortho, det_gdf, miss_gdf, args.mask_tif, args.output_dir)

    print(f"\n[Terminé] en {(time.time()-t0)/60:.1f} min")
    print(f"  Résultats → {args.output_dir}/")


if __name__ == "__main__":
    main()
