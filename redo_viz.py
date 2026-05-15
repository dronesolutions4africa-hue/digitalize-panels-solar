"""Re-run only save_visualization using the preserved binary.dat memmap."""
import gc
import logging
import numpy as np
import rasterio
import rasterio.enums
from rasterio.windows import Window
import os, sys

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("redo_viz")

BINARY_DAT  = r"C:\Users\user\AppData\Local\Temp\detect_geotiff_oyl8xxd6\binary.dat"
INPUT_TIF   = "ortho.tif"
OUTPUT_VIZ  = "output/ortho/panels_viz.tif"

with rasterio.open(INPUT_TIF) as src:
    height, width = src.height, src.width
    logger.info(f"Raster: {width}x{height}, CRS={src.crs}")

    binary_mask = np.memmap(BINARY_DAT, dtype=np.uint8, mode="r", shape=(height, width))
    panel_px = int(np.sum(binary_mask == 255))
    logger.info(f"Binary mask loaded — {panel_px} panel pixels ({panel_px/(height*width)*100:.2f}%)")

    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_VIZ)), exist_ok=True)
    chunk_rows = 2000

    logger.info(f"Writing visualization to {OUTPUT_VIZ} (BIGTIFF)...")
    with rasterio.open(
        OUTPUT_VIZ, "w",
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
            rgb = src.read([1, 2, 3], window=win)
            viz = rgb.astype(np.float32)
            panel = (np.array(binary_mask[r: r + h]) == 255)
            viz[0][panel] = viz[0][panel] * 0.5 + 127.5
            viz[1][panel] = viz[1][panel] * 0.5
            viz[2][panel] = viz[2][panel] * 0.5
            viz = viz.clip(0, 255).astype(np.uint8)
            dst.write(viz, window=win)
            if (r // chunk_rows) % 5 == 0:
                logger.info(f"  row {r}/{height} ({r/height*100:.0f}%)")

    logger.info(f"Done: {OUTPUT_VIZ}")

del binary_mask
gc.collect()
