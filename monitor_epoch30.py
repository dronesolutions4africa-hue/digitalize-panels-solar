#!/usr/bin/env python3
"""
monitor_epoch30.py — Surveillance automatique epoch 30.
Si val_iou < 0.42 a l'epoch 30 : redémarre avec oversample=1, lr=1e-5, resume.
"""
import csv
import datetime
import os
import subprocess
import time

BASE = "/mnt/c/Users/user/Downloads/solar-panels-detection-master/solar-panels-detection-master"
CSV_PATH = f"{BASE}/trained_models/p2_attn_unet_3sites/training_log.csv"
SCRIPT = "/home/solar/run_p2_gpu.sh"
LOG = f"{BASE}/trained_models/p2_attn_unet_3sites/monitor.log"
TARGET_EPOCH = 30
VAL_THRESHOLD = 0.42
CHECK_INTERVAL = 1800  # 30 minutes


def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    line = f"{ts}  {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def read_last_epoch_row(target_epoch):
    """Retourne le dernier row CSV avec epoch == target_epoch."""
    found = None
    try:
        with open(CSV_PATH, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    if int(float(row["epoch"])) == target_epoch:
                        found = row
                except (ValueError, KeyError):
                    pass
    except FileNotFoundError:
        pass
    return found

def get_current_max_epoch():
    max_ep = -1
    try:
        with open(CSV_PATH, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ep = int(float(row["epoch"]))
                    if ep > max_ep:
                        max_ep = ep
                except (ValueError, KeyError):
                    pass
    except FileNotFoundError:
        pass
    return max_ep

def rewrite_script_for_intervention():
    with open(SCRIPT) as f:
        content = f.read()
    content = content.replace("--panel_oversample 3", "--panel_oversample 1")
    content = content.replace("--lr 1e-4", "--lr 1e-5")
    if "--resume" not in content:
        content = content.replace(
            "--output_dir",
            "--resume \\\n  --skip_phase1 \\\n  --initial_epoch 30 \\\n  --output_dir"
        )
    with open(SCRIPT, "w") as f:
        f.write(content)
    log("Script mis a jour: oversample=1, lr=1e-5, resume+skip_phase1+initial_epoch=30")

def kill_training():
    try:
        subprocess.run(["pkill", "-f", "train.py"], check=False)
        time.sleep(5)
        log("Ancien processus train.py arrete")
    except Exception as e:
        log(f"Erreur pkill: {e}")

def restart_training():
    outdir = f"{BASE}/trained_models/p2_attn_unet_3sites"
    subprocess.Popen(
        ["nohup", "bash", SCRIPT],
        stdout=open(f"{outdir}/train_log.txt", "a"),
        stderr=open(f"{outdir}/train_log.err", "a"),
        start_new_session=True
    )
    time.sleep(5)
    result = subprocess.run(["pgrep", "-f", "train.py"], capture_output=True, text=True)
    if result.stdout.strip():
        log(f"Nouveau training lance PID={result.stdout.strip().split()[0]}")
    else:
        log("ATTENTION: training ne semble pas avoir demarre")

def main():
    log(f"Surveillance demarree — attente epoch {TARGET_EPOCH}, seuil val_iou={VAL_THRESHOLD}")

    while True:
        max_ep = get_current_max_epoch()
        log(f"Epoch max actuelle: {max_ep}")

        if max_ep >= TARGET_EPOCH:
            row = read_last_epoch_row(TARGET_EPOCH)
            if row:
                val_iou = float(row["val_main_output_panel_iou"])
                train_iou = float(row["main_output_panel_iou"])
                log(f"Epoch {TARGET_EPOCH}: val_iou={val_iou:.4f}  train_iou={train_iou:.4f}")
                if val_iou < VAL_THRESHOLD:
                    log(f"INTERVENTION: val_iou={val_iou:.4f} < {VAL_THRESHOLD} — redemarrage")
                    rewrite_script_for_intervention()
                    kill_training()
                    restart_training()
                    log("Intervention terminee.")
                else:
                    log(f"OK: val_iou={val_iou:.4f} >= {VAL_THRESHOLD} — aucune intervention")
            break

        if max_ep > 35:
            log(f"Epoch {max_ep} > 35 sans epoch 30 — surveillance terminee")
            break

        time.sleep(CHECK_INTERVAL)

    log("Surveillance terminee.")

if __name__ == "__main__":
    main()
