"""Creates a synthetic 768x768 RGB GeoTIFF with simulated solar panel clusters for testing."""
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import os

os.makedirs("test_geotiff", exist_ok=True)

H, W = 768, 768
rng = np.random.default_rng(42)

# Background: greenish (vegetation)
rgb = rng.integers(40, 120, (3, H, W), dtype=np.uint8)
rgb[1] = np.clip(rgb[1].astype(int) + 40, 0, 255).astype(np.uint8)

# Six simulated solar panel clusters (blue-grey rectangles)
panels = [
    (80,  80,  160, 140),
    (200, 50,  310, 100),
    (400, 200, 510, 280),
    (100, 400, 200, 460),
    (550, 350, 650, 430),
    (300, 550, 420, 620),
]
for (c0, r0, c1, r1) in panels:
    rgb[0, r0:r1, c0:c1] = rng.integers(100, 140, (r1-r0, c1-c0), dtype=np.uint8)
    rgb[1, r0:r1, c0:c1] = rng.integers(100, 140, (r1-r0, c1-c0), dtype=np.uint8)
    rgb[2, r0:r1, c0:c1] = rng.integers(140, 180, (r1-r0, c1-c0), dtype=np.uint8)

# 0.15 m/px, near Brisbane (EPSG:32754 UTM zone 54S)
transform = from_bounds(
    502000, 6953000,
    502000 + W * 0.15, 6953000 + H * 0.15,
    W, H
)
crs = CRS.from_epsg(32754)

out = "test_geotiff/rgb_test.tif"
with rasterio.open(
    out, "w", driver="GTiff",
    height=H, width=W, count=3, dtype="uint8",
    crs=crs, transform=transform, compress="lzw"
) as dst:
    dst.write(rgb)

print("Created:", out, "  (" + str(W) + "x" + str(H) + "px, EPSG:32754, 0.15m/px)")
print("Panel clusters:", len(panels))
