# Rapport de surveillance — 2026-05-23

## État : DÉMARRAGE — BLOCAGE ÉPOQUE 1 (17e cycle)
- Époque : 0/50 (aucune époque complète enregistrée dans `training_log.csv`)
- Meilleure val IoU panneaux : N/A
- Meilleure val loss : N/A
- Tendance (5 dernières époques) : N/A (aucune donnée)
- ETA estimée : inconnue — époque 1 toujours non complétée

Entraînement démarré, en attente de données.

## Contexte du run GPU (données du log actuel)
- GPU : NVIDIA RTX A4500 Laptop (13.7 GB VRAM) — `cuda_malloc_async`
- Données chargées : `Orthomosaic_Patisen.tif` (18 695×16 883 px)
- Masque : 34 445 277 px panneaux / 315 627 685 total (**10.9%**)
- Tuiles : 3 687 brutes → 5 886 oversamplées (train), 921 (val)
- Modèle : Fast SCNN v2 — 1 901 450 params (7.25 MB)
- Hyperparamètres actifs au dernier run : `lr=0.0001`, `panel_weight=15.0`, **`batch_size=16`** (correctif batch=8 dans le dépôt mais non appliqué sur la machine)

## Diagnostic cycle 17

### Situation vs cycle 16
**Aucun progrès détecté** — fichiers identiques au cycle 16 :
- `training_log.csv` : **0 octet** (0 époque complète)
- `train_log.txt` (347 lignes) : se termine à `[Train] epochs=50  batch=16  steps/epoch~367` puis `Epoch 1/50` — **batch=16 confirme que la machine n'a pas encore appliqué le correctif batch=8**
- `train_log.txt.err` : `python3: can't open file '/home/solar/train.py': [Errno 2] No such file or directory` (artefact d'une tentative antérieure à chemin erroné)

### Racine du blocage
Le run actuel tourne avec `batch=16`, causant vraisemblablement un OOM silencieux (RTX A4500 13.7 GB, activations ~9.6 GB pour batch=16 + overhead). L'époque 1 ne se termine jamais, `training_log.csv` reste vide.

**Le dépôt contient les correctifs depuis le cycle 11 (`batch_size 16→8` + `SCRIPT_DIR`) — la machine WSL2 n'a pas encore exécuté `git pull && bash run_gpu_wsl.sh`.**

### Calcul mémoire GPU
| batch_size | Mémoire activations (estimée) | Statut |
|---|---|---|
| 16 | ~9.6 GB | RISQUE OOM sur 13.7 GB VRAM |
| **8** | ~4.8 GB | SÛRE — marge 9 GB |
| 4 | ~2.4 GB | SÛRE si OOM persiste avec batch=8 |

## Historique (10 dernières époques enregistrées)
| Époque | val_loss | val_panel_iou |
|--------|----------|---------------|
| —      | —        | —             |

*Aucune époque complète depuis 17 cycles de surveillance consécutifs.*

---

## Recommandations

### A. Données Malicounda

**Recommandation : NE PAS intégrer Malicounda maintenant — attendre ≥ 10 époques Patisen-GPU fonctionnelles.**

Raisons :
1. **Blocage non résolu** : lancer un entraînement multi-site (9.4 GB) sans baseline stable aggrave le risque de crash
2. **Aucune val_iou connue** : impossible de mesurer l'apport de Malicounda sans référence Patisen
3. **Résolution hétérogène** : 1.7 cm/px (Malicounda) vs 3 cm/px (Patisen). Tuile 512 px = 8.7 m×8.7 m à Malicounda vs 15.4 m×15.4 m à Patisen. Les panneaux apparaissent ×1.76 plus grands — prévoir `tile_size=256` ou normalisation d'échelle pour Malicounda

**Commande multi-site (à lancer APRÈS ≥ 10 époques Patisen-GPU fonctionnelles) :**
```bash
python3 train.py \
  --ortho Data/Orthomosaic_Patisen.tif Malicounda/ortho.tif \
  --shp Data/Panneaux_Patisen.shp Malicounda/Lim_panneaux.shp \
  --tile_size 512 --stride 256 --batch_size 8 --epochs 50 \
  --panel_oversample 4 --max_tiles_per_site 5000 \
  --output_dir trained_models/patisen_malicounda_gpu
```

### B. Stratégie pour atteindre IoU >= 0.85

Décision conditionnelle à appliquer dès l'époque 10 :

| Condition à l'époque 10 | Action recommandée |
|---|---|
| val IoU > 0.70 | Continuer Fast SCNN 50 époques ; envisager multi-site Malicounda ensuite |
| 0.50 ≤ val IoU ≤ 0.70 | Ajouter `ReduceLROnPlateau(patience=5, factor=0.5)` ; réévaluer à époque 20 |
| val IoU < 0.50 | **Escalader vers U-Net + ResNet50 ImageNet** multi-site (commande ci-dessous) |
| val IoU stagne ≥ 5 époques | Réduire LR × 0.5 ou escalader vers U-Net |

**Commande U-Net de secours :**
```bash
python3 train.py \
  --model unet_resnet50 \
  --ortho Data/Orthomosaic_Patisen.tif Malicounda/ortho.tif \
  --shp Data/Panneaux_Patisen.shp Malicounda/Lim_panneaux.shp \
  --tile_size 512 --stride 256 --batch_size 4 --epochs 50 \
  --panel_oversample 4 --max_tiles_per_site 5000 \
  --output_dir trained_models/unet_resnet50_multisite
```

### C. Hyperparamètres

| Paramètre | Valeur actuelle | Recommandation |
|---|---|---|
| `batch_size` | **16** (en cours) → **8** (correctif à appliquer) | Appliquer le correctif batch=8 immédiatement |
| `panel_weight` | 15.0 | Maintenir. Augmenter à **20–25** si val IoU < 0.40 après époque 10 |
| `panel_oversample` | 4 | OK. Augmenter à **6–8** si IoU stagne après époque 15 |
| `tile_size` | 512 px (15.4 m×15.4 m à Patisen) | Correct pour Patisen. Envisager 256 px pour Malicounda (1.7 cm/px → 8.7 m×8.7 m) |
| `lr` | 0.0001 | Correct. Ajouter `ReduceLROnPlateau(patience=5)` si plateau détecté |
| `stride` | 256 (overlap 50%) | Correct |

---

## Décision

**17e cycle — EN ATTENTE DE RELANCE — correctifs batch=8 + SCRIPT_DIR confirmés dans le dépôt depuis cycle 11.**

**Action requise (unique) sur la machine WSL2 :**
```bash
cd /home/solar/digitalize-panels-solar
git pull origin main
nohup bash run_gpu_wsl.sh \
  > trained_models/patisen_gpu/train_log.txt \
  2> trained_models/patisen_gpu/train_log.txt.err &
echo "PID=$!"
# Vérifier absence d'OOM après 3 min
sleep 180 && cat trained_models/patisen_gpu/train_log.txt.err
# Vérifier époque 1 complète après 30 min
sleep 1800 && head -3 trained_models/patisen_gpu/training_log.csv
```

Si `.err` contient `OOM` ou `ResourceExhausted` → réduire à `batch_size=4` dans `run_gpu_wsl.sh`.

---

### Historique des cycles de surveillance

| Cycle | Date | État |
|-------|------|------|
| 1–8   | — | 0 époque, démarrage/crash |
| 9  | — | CORRECTIF `run_gpu_wsl.sh` chemin dynamique (`SCRIPT_DIR`) |
| 10 | — | En attente relance — `train.py` atteint, époque 1 incomplète |
| 11 | — | CORRECTIF `batch_size=16→8` dans `run_gpu_wsl.sh` |
| 12 | — | EN ATTENTE RELANCE — batch=8 confirmé, 0 époque |
| 13 | — | EN ATTENTE RELANCE — `train.py` confirmé dans dépôt |
| 14 | 2026-05-22 | EN ATTENTE RELANCE — situation inchangée |
| 15 | 2026-05-22 | EN ATTENTE RELANCE — 15e cycle sans progression |
| 16 | 2026-05-23 | EN ATTENTE RELANCE — batch=16 confirmé dans log (correctif non appliqué) |
| **17** | **2026-05-23** | **EN ATTENTE RELANCE — situation identique au cycle 16, correctifs toujours non appliqués sur WSL2** |
