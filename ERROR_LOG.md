# ERROR_LOG — detect_geotiff.py implementation

## 2026-05-06

---

### ERROR-001 — No importable `build_model` in solar_panel_detection.py

**Status:** LOGGED

**Finding:** `solar_panel_detection.py` does not export a `build_model` function.
Model creation is performed inline per `model_type` using direct module imports from `model_list/`:

| model_type (str) | actual call |
|---|---|
| `fast_scnn_v2` | `fast_scnn_2.fast_scnn_v2(input_shape, batch_size=1, n_labels=2, model_summary=False)` |
| `seg_resnet_v2` | `segnet_3.segnet_resnet_v2(input_shape, batch_size=1, n_labels=2, model_summary=False)` |
| `segnet_4ed` | `segnet_1.segnet_4_encoder_decoder(input_shape, batch_size=1, n_labels=2, model_summary=False)` |
| `segnet_original` | `segnet_0.segnet_original(input_shape, batch_size=1, n_labels=2, model_summary=False)` |

**APPROACH_CHANGE:** `from solar_panel_detection import build_model` is not usable.
A local `build_model(model_type: str, tile_size: int)` function is defined directly
in `detect_geotiff.py` that imports from `model_list` subpackage as above.

---

### ERROR-002 — Model name mismatch (task spec vs actual weight filenames)

**Status:** LOGGED

**Finding:** Task spec uses `--model_name fast_scnn_v2_best` as default. Actual weight
files present in `trained_models/` are:

| filename | architecture |
|---|---|
| `fast_scnn_2.h5` | Fast SCNN v2 — best model (IoU 0.8224) |
| `fast_scnn_1.h5` | Fast SCNN v1 |
| `fast_scnn_original.h5` | Original Fast SCNN |
| `seg_resnet_1.h5` | SegNet ResNet v1 |
| `seg_resnet_2.h5` | SegNet ResNet v2 |
| `segnet_1.h5` | SegNet 4 encoder-decoder |
| `segnet_2.h5` | SegNet variant |
| `segnet_original.h5` | Original SegNet |

**APPROACH_CHANGE:** Default `--model_name` changed to `fast_scnn_2.h5` (actual filename).
`load_model()` resolves the path in order: argument as-is → under `trained_models/` → with `.h5`
appended. Raises `FileNotFoundError` with the available filenames listed if none resolves.

---

### ERROR-003 — ortho_convertie.tif has 2 bands (float32), not 3-band RGB uint8

**Status:** LOGGED

**Finding:** `ortho_convertie.tif` has:
- Band 1: float32, range [1.0, 78.13], mean 26.5 — luminance/intensity data
- Band 2: float32, only values {0.0, 1.0} — binary validity mask

The model expects 3-band uint8 RGB tiles (values 0–255 divided by 255 → [0,1]).

**APPROACH_CHANGE:**
- `compute_norm_params(src)`: samples band 1 at decimated resolution to compute
  2nd–98th percentile stretch parameters (avoids loading the full 13 k×13 k raster).
- `tile_geotiff`: if `src.count < 3`, reads band 1 and replicates it to 3 channels;
  applies percentile stretch (not /255) for float32 data.
- `save_visualization`: if `src.count < 3`, builds 3-channel grayscale from band 1
  for the viz background instead of reading RGB bands directly.
- `main()`: replaces hard `sys.exit(1)` with a warning + continuation for <3-band inputs.

---

### ERROR-004 — ortho.tif is 48 027×51 341 px (2.46 GP) — OOM on in-RAM accumulation

**Status:** LOGGED

**Finding:** `ortho.tif` is 4-band uint8 RGBA, EPSG:32628, 0.017 m/px.
At 2.46 gigapixels, the current code does two fatal things:
1. `tiles_data = list(tile_geotiff(...))` loads 37 788 tiles × 256×256×3×4 B ≈ **9.5 GB** into RAM at once.
2. `full_mask` + `weight_map` as float32 in RAM = **2 × 9.86 GB = 19.7 GB**.
3. `src.read([1,2,3])` in `save_visualization` = **7.4 GB** in one shot.

**APPROACH_CHANGE:**
- Add `LARGE_RASTER_MP = 50` threshold (megapixels).
- Above threshold: **non-overlapping tile mode** — threshold each batch directly to uint8 binary_mask (no float32 accumulation arrays needed).
- `binary_mask` stored as `np.memmap` on disk (~2.47 GB temp file).
- `tile_geotiff` becomes a generator consumed batch-by-batch (no `list()`).
- `save_visualization` writes in 2 000-row chunks.
- `mask_to_polygons` uses `rasterio.features.shapes` in 5 000-row strips.
- Estimated inference time on CPU: ~30 min for 4 724 batches.
- `tmpdir` cleaned up in `finally` block.

---

### ERROR-005 — save_visualization crashes: TIFF 4 GB limit + memmap PermissionError on Windows

**Status:** LOGGED

**Finding (run on ortho.tif 48 027×51 341):**

A) `rasterio.errors.RasterioIOError: TIFFAppendToStrip: Maximum TIFF file size exceeded. Use BIGTIFF=YES`
   Output viz GeoTIFF = 48 027×51 341×3 bytes ≈ 7.4 GB, exceeds standard TIFF 4 GB limit.
   Fix: add `BIGTIFF="YES"` to `rasterio.open()` creation options in `save_visualization`.

B) `PermissionError [WinError 32]` on `shutil.rmtree(tmpdir)` — the `np.memmap` for
   `binary.dat` is still open when `finally` block runs on Windows (mmap file handles
   are not released until the object is garbage-collected).
   Fix: explicitly `del binary_mask` and `gc.collect()` before cleanup in `main()`.

**GOOD NEWS:** inference and shapefile both completed successfully before the crash:
   - 414 267 040 panel pixels (16.80% of raster)
   - 411 polygons saved to output/ortho/panels.shp
   - tmpdir still on disk (rmtree failed → binary.dat preserved for viz retry)

---

*No ATTEMPTED_FIX entries — all errors logged before fixes applied.*
