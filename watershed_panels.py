"""
watershed_panels.py — Évalue le modèle pixel par pixel et sépare les blobs en
panneaux individuels via distance transform + watershed.

Usage:
    python watershed_panels.py \
        --mask_tif   output/patisen/binary_mask.tif \
        --gt_shp     Data/Panneaux_Patisen.shp \
        --ortho      Data/Orthomosaic_Patisen.tif \
        --output_shp output/patisen/panels_watershed.shp \
        --output_img output/patisen/panels_watershed_viz.tif \
        --peak_dist  45 \
        --min_panel_px 2000
"""

import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import rasterio
import rasterio.features
import rasterio.windows
import geopandas as gpd
from affine import Affine
from rasterio.features import shapes as rio_shapes
from scipy import ndimage as ndi
from shapely.geometry import shape
from tqdm import tqdm


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mask_tif",      required=True,  help="Binary mask GeoTIFF (0/255)")
    p.add_argument("--gt_shp",        required=True,  help="Ground truth shapefile")
    p.add_argument("--ortho",         required=True,  help="Original orthomosaic (pour viz)")
    p.add_argument("--output_shp",    required=True,  help="Shapefile de sortie watershed")
    p.add_argument("--output_img",    required=True,  help="Visualisation GeoTIFF de sortie")
    p.add_argument("--peak_dist",     type=int,   default=45,
                   help="Distance min en pixels entre centres de panneaux (≈ moitié largeur panneau)")
    p.add_argument("--min_panel_px",  type=int,   default=2000,
                   help="Surface min d'un panneau en pixels (évite les fragments)")
    p.add_argument("--chunk_rows",    type=int,   default=4000,
                   help="Lignes traitées par bande pour le watershed (mémoire)")
    p.add_argument("--margin",        type=int,   default=200,
                   help="Marge de recouvrement entre bandes en pixels")
    return p.parse_args()


# ── Évaluation pixel ──────────────────────────────────────────────────────────
def pixel_evaluation(mask_tif: str, gt_shp: str):
    """Calcule TP/FP/FN/IoU/Recall/Precision pixel à pixel."""
    print("\n[Éval pixel] Chargement du masque prédit…")
    with rasterio.open(mask_tif) as src:
        H, W = src.height, src.width
        transform, crs = src.transform, src.crs

    print("[Éval pixel] Rasterisation du ground truth…")
    gdf = gpd.read_file(gt_shp)
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    shapes_gt = [(geom, 1) for geom in gdf.geometry if geom is not None]
    gt_mask = rasterio.features.rasterize(
        shapes_gt, out_shape=(H, W), transform=transform,
        fill=0, dtype=np.uint8, all_touched=True,
    )

    print("[Éval pixel] Calcul des métriques par bande…")
    TP = FP = FN = TN = 0
    chunk = 4000
    with rasterio.open(mask_tif) as src:
        for r in tqdm(range(0, H, chunk), desc="métriques"):
            h = min(chunk, H - r)
            pred = src.read(1, window=rasterio.windows.Window(0, r, W, h)).astype(bool)
            gt   = gt_mask[r:r+h].astype(bool)
            TP  += int((pred &  gt).sum())
            FP  += int((pred & ~gt).sum())
            FN  += int((~pred &  gt).sum())
            TN  += int((~pred & ~gt).sum())

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    iou       = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n{'─'*50}")
    print(f"  Pixels vrais panneaux (GT)   : {TP+FN:>12,}")
    print(f"  Pixels prédits panneaux      : {TP+FP:>12,}")
    print(f"  Vrais positifs (TP)          : {TP:>12,}")
    print(f"  Faux positifs (FP)           : {FP:>12,}")
    print(f"  Faux négatifs (FN — ratés)   : {FN:>12,}")
    print(f"  Précision  (pas de faux+)    : {precision*100:>10.1f}%")
    print(f"  Rappel     (panneaux couverts): {recall*100:>10.1f}%")
    print(f"  IoU pixel                    : {iou*100:>10.1f}%")
    print(f"  F1-score                     : {f1*100:>10.1f}%")
    print(f"{'─'*50}\n")
    return gt_mask, transform, crs, H, W


# ── Watershed par bande ────────────────────────────────────────────────────────
def watershed_strip(binary_strip: np.ndarray, peak_dist: int, min_panel_px: int):
    """
    Applique distance transform + watershed sur une bande 2D uint8 (0/255).
    Retourne un masque uint8 des instances séparées (255=panneau, 0=fond).
    """
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    binary = binary_strip == 255

    if binary.sum() == 0:
        return np.zeros_like(binary_strip)

    # Distance transform : chaque pixel panneau = distance au fond le plus proche
    dist = ndi.distance_transform_edt(binary)

    # Pics locaux = centres des panneaux individuels
    coords = peak_local_max(dist, min_distance=peak_dist, labels=binary)
    markers = np.zeros(binary.shape, dtype=np.int32)
    for idx, (r, c) in enumerate(coords, start=1):
        markers[r, c] = idx

    # Watershed depuis chaque centre vers l'extérieur
    labels = watershed(-dist, markers, mask=binary)

    # Supprimer les instances trop petites
    result = np.zeros_like(binary_strip)
    for lbl in np.unique(labels):
        if lbl == 0:
            continue
        if (labels == lbl).sum() >= min_panel_px:
            result[labels == lbl] = 255

    return result


# ── Watershed sur tout le raster par bandes ───────────────────────────────────
def apply_watershed(mask_tif: str, peak_dist: int, min_panel_px: int,
                    chunk_rows: int, margin: int, transform, crs, H, W):
    """Parcourt le masque en bandes, applique watershed, collecte les polygones."""
    print(f"[Watershed] peak_dist={peak_dist}px, min_panel_px={min_panel_px}px")
    print(f"            bandes de {chunk_rows}px + marge {margin}px")

    geometries = []

    with rasterio.open(mask_tif) as src:
        for r_start in tqdm(range(0, H, chunk_rows), desc="watershed"):
            # Bande avec marge
            r0 = max(0, r_start - margin)
            r1 = min(H, r_start + chunk_rows + margin)
            h  = r1 - r0

            strip = src.read(1, window=rasterio.windows.Window(0, r0, W, h))

            if strip.max() == 0:
                continue

            ws_strip = watershed_strip(strip, peak_dist, min_panel_px)

            # Zone utile sans les marges
            useful_r0 = r_start - r0
            useful_r1 = min(r1, r_start + chunk_rows) - r0
            ws_useful  = ws_strip[useful_r0:useful_r1]

            if ws_useful.max() == 0:
                continue

            # Transform de la bande utile
            strip_transform = transform * Affine.translation(0, r_start)

            for geom_dict, val in rio_shapes(
                ws_useful.astype(np.uint8),
                mask=(ws_useful == 255),
                transform=strip_transform,
            ):
                poly = shape(geom_dict)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if not poly.is_empty:
                    geometries.append(poly)

    print(f"[Watershed] {len(geometries)} polygones extraits")
    return geometries


# ── Sauvegarde shapefile ───────────────────────────────────────────────────────
def save_results(geometries, crs, output_shp, min_panel_px, transform):
    """Filtre par aire et sauvegarde."""
    import geopandas as gpd
    if not geometries:
        print("[Sortie] Aucun polygone — shapefile non créé")
        return gpd.GeoDataFrame(columns=["geometry"], crs=crs)

    gdf = gpd.GeoDataFrame(geometry=geometries, crs=crs)

    # Aire en m² (CRS projeté ou approximation géographique)
    if crs.is_projected:
        gdf["area_m2"] = gdf.geometry.area
    else:
        # Résolution pixel en degrés → m via facteur ~1.18 cm/px
        px_m = abs(transform.a) * 111_320.0   # degrés → mètres (approx lat=15°)
        px_area = px_m ** 2
        gdf["area_m2"] = gdf.geometry.area / (abs(transform.a) ** 2) * px_area

    min_m2 = min_panel_px * (abs(transform.a) * 111_320.0) ** 2
    before = len(gdf)
    gdf = gdf[gdf["area_m2"] >= min_m2 * 0.3].reset_index(drop=True)
    gdf["panel_id"] = range(1, len(gdf) + 1)

    os.makedirs(os.path.dirname(os.path.abspath(output_shp)), exist_ok=True)
    gdf.to_file(output_shp, driver="ESRI Shapefile")
    print(f"[Sortie] {len(gdf)} panneaux sauvegardés → {output_shp}")
    print(f"         (supprimé {before - len(gdf)} trop petits)")
    return gdf


# ── Visualisation ──────────────────────────────────────────────────────────────
def save_viz(ortho: str, gdf, output_img: str, crs):
    """Superpose les polygones détectés sur l'orthomosaïque."""
    print(f"[Viz] Génération de {output_img}…")
    if gdf is None or len(gdf) == 0:
        print("[Viz] Aucun panneau — viz ignorée")
        return

    with rasterio.open(ortho) as src:
        H, W = src.height, src.width
        # Rasterise les polygones détectés
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)
        shapes_det = [(geom, 255) for geom in gdf.geometry]
        det_mask = rasterio.features.rasterize(
            shapes_det, out_shape=(H, W), transform=src.transform,
            fill=0, dtype=np.uint8, all_touched=False,
        )

        os.makedirs(os.path.dirname(os.path.abspath(output_img)), exist_ok=True)
        with rasterio.open(
            output_img, "w", driver="GTiff",
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
                panel = det_mask[r:r+h] == 255
                # Panneaux détectés → surbrillance verte
                rgb[0][panel] = rgb[0][panel] * 0.4
                rgb[1][panel] = rgb[1][panel] * 0.4 + 153
                rgb[2][panel] = rgb[2][panel] * 0.4
                dst.write(np.clip(rgb, 0, 255).astype(np.uint8), window=win)
    print(f"[Viz] Sauvegardé → {output_img}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    t0 = time.time()

    # 1. Évaluation pixel
    gt_mask, transform, crs, H, W = pixel_evaluation(args.mask_tif, args.gt_shp)
    del gt_mask  # libère la RAM

    # 2. Watershed
    geometries = apply_watershed(
        args.mask_tif, args.peak_dist, args.min_panel_px,
        args.chunk_rows, args.margin, transform, crs, H, W,
    )

    # 3. Sauvegarde
    gdf = save_results(geometries, crs, args.output_shp, args.min_panel_px, transform)

    # 4. Visualisation
    save_viz(args.ortho, gdf, args.output_img, crs)

    print(f"\n[Terminé] {len(gdf)} panneaux détectés en {(time.time()-t0)/60:.1f} min")
    print(f"  Shapefile : {args.output_shp}")
    print(f"  Viz       : {args.output_img}")


if __name__ == "__main__":
    main()
