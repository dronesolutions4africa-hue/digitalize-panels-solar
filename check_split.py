import sys
sys.path.insert(0, ".")
from train import rasterize_labels, get_tile_indices

mask = rasterize_labels("Data/Orthomosaic_Patisen.tif", "Data/Panneaux_Patisen.shp")
H, W = mask.shape
tr, va = get_tile_indices(H, W, 512, 256, 0.2, mask)

panel_in_val = sum(1 for r, c in va if mask[r:r+512, c:c+512].max() > 0)
panel_in_tr  = sum(1 for r, c in tr if mask[r:r+512, c:c+512].max() > 0)
bg_in_val    = len(va) - panel_in_val
bg_in_tr     = len(tr) - panel_in_tr

print(f"Train: {len(tr)} tiles  ({panel_in_tr} panel tiles, {bg_in_tr} background)")
print(f"Val  : {len(va)} tiles  ({panel_in_val} panel tiles, {bg_in_val} background)")
print(f"Panel tile ratio in val: {panel_in_val/len(va)*100:.1f}%")
