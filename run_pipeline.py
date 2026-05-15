"""
run_pipeline.py — Pipeline d'entraînement auto-escaladant.

Étape 1 : Entraîne Fast SCNN v2  (rapide, 7.9 MB)
Étape 2 : Si val IoU < seuil → entraîne U-Net + ResNet50 (ImageNet pre-trained)

Usage
-----
    python run_pipeline.py
    python run_pipeline.py --iou_threshold 0.85 --epochs 150
    python run_pipeline.py --ortho Data/mon_ortho.tif --shp Data/mes_panneaux.shp
"""

import argparse
import csv
import os
import subprocess
import sys


def read_best_iou(csv_path: str) -> float:
    if not os.path.exists(csv_path):
        return 0.0
    best = 0.0
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        val_key = None
        for row in reader:
            if val_key is None:
                val_key = next(
                    (k for k in row if 'val' in k and 'iou' in k.lower()), None
                )
            if val_key and row.get(val_key, ''):
                try:
                    best = max(best, float(row[val_key]))
                except ValueError:
                    pass
    return best


def run_train(model_name: str, output_dir: str, base_args: list) -> int:
    cmd = [
        sys.executable, "train.py",
        "--model",      model_name,
        "--output_dir", output_dir,
    ] + base_args
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"[Pipeline] {' '.join(cmd)}")
    print(f"{sep}\n")
    return subprocess.run(cmd).returncode


def main():
    p = argparse.ArgumentParser(description="Auto-escalating solar-panel training pipeline")
    p.add_argument("--ortho",         default="Data/Orthomosaic_Patisen.tif")
    p.add_argument("--shp",           default="Data/Panneaux_Patisen.shp")
    p.add_argument("--iou_threshold", type=float, default=0.85,
                   help="Val IoU target — escalate to U-Net if not reached")
    p.add_argument("--epochs",        type=int,   default=200)
    p.add_argument("--batch_size",    type=int,   default=8)
    p.add_argument("--tile_size",     type=int,   default=512)
    p.add_argument("--freeze_encoder_epochs", type=int, default=20,
                   help="U-Net phase 1: epochs with frozen ResNet50 encoder")
    args = p.parse_args()

    base = [
        "--ortho",      args.ortho,
        "--shp",        args.shp,
        "--epochs",     str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--tile_size",  str(args.tile_size),
    ]

    # ── Étape 1 : Fast SCNN v2 ────────────────────────────────────────────────
    out1 = "trained_models/patisen_fast_scnn"
    rc = run_train("fast_scnn_2", out1, base)
    if rc != 0:
        sys.exit(f"[Pipeline] ERREUR : Fast SCNN a échoué (exit {rc})")

    best_iou = read_best_iou(os.path.join(out1, "training_log.csv"))
    print(f"\n[Pipeline] Fast SCNN v2 → meilleur val IoU = {best_iou:.4f}")

    if best_iou >= args.iou_threshold:
        print(f"[Pipeline] Objectif atteint (>= {args.iou_threshold}). TERMINÉ.")
        print(f"           Poids : {out1}/best_model.weights.h5")
        return

    # ── Étape 2 : U-Net + ResNet50 ────────────────────────────────────────────
    print(f"\n[Pipeline] IoU {best_iou:.4f} < {args.iou_threshold} "
          f"→ escalade vers U-Net + ResNet50.")

    out2 = "trained_models/patisen_unet_resnet50"
    unet_args = base + ["--freeze_encoder_epochs", str(args.freeze_encoder_epochs)]
    rc = run_train("unet_resnet50", out2, unet_args)
    if rc != 0:
        sys.exit(f"[Pipeline] ERREUR : U-Net a échoué (exit {rc})")

    best_iou2 = read_best_iou(os.path.join(out2, "training_log.csv"))
    print(f"\n[Pipeline] U-Net ResNet50 → meilleur val IoU = {best_iou2:.4f}")

    winner_dir  = out1 if best_iou >= best_iou2 else out2
    winner_iou  = max(best_iou, best_iou2)
    winner_arch = "Fast SCNN v2" if winner_dir == out1 else "U-Net ResNet50"
    print(f"\n[Pipeline] Meilleur modèle : {winner_arch}")
    print(f"           IoU  : {winner_iou:.4f}")
    print(f"           Poids: {winner_dir}/best_model.weights.h5")


if __name__ == "__main__":
    main()
