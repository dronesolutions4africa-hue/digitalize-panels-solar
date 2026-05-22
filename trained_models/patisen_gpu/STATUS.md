# Rapport de surveillance — 2026-05-22

## État : DÉMARRAGE — BLOCAGE ÉPOQUE 1 (13e cycle)
- Époque : 0/50 (aucune époque complète enregistrée dans `training_log.csv`)
- Meilleure val IoU panneaux : N/A
- Meilleure val loss : N/A
- Tendance (5 dernières époques) : N/A (aucune donnée)
- ETA estimée : inconnue — époque 1 toujours non complétée

## Contexte du run GPU (données du log actuel)
- GPU : NVIDIA RTX A4500 Laptop (13.7 GB VRAM) — `cuda_malloc_async`
- Données chargées : `Orthomosaic_Patisen.tif` (18 695×16 883 px, chargé en 22.5s)
- Masque : 34 445 277 px panneaux / 315 627 685 total (**10.9%**)
- Tuiles : 3 687 brutes → 5 886 oversamplées (train), 921 (val)
- Modèle : Fast SCNN v2 — 1 901 450 params (7.25 MB)
- Hyperparamètres log actuel : `lr=0.0001`, `panel_weight=15.0`, `batch=16` (log PRÉ-fix)
- **Script `run_gpu_wsl.sh` corrigé** : `batch_size=8` + `SCRIPT_DIR` depuis cycle 11 ✓

## Diagnostic cycle 13

### Nouveauté vs cycle 12
Aucun progrès dans `training_log.csv` — fichier toujours à **0 octet**.
`train_log.txt` (347 lignes) se termine identiquement à `Epoch 1/50` — inchangé.

### Confirmation : `train.py` présent dans le dépôt git
**Bonne nouvelle** : `train.py` est bien présent à la racine du dépôt (confirmé ce cycle).
L'erreur `python3: can't open file '/home/solar/train.py'` dans `train_log.txt.err`
est un **artefact d'un ancien run** (avant le fix SCRIPT_DIR du cycle 9) où le script
était appelé depuis `/home/solar/` sans `cd` vers le répertoire du projet.

Le `run_gpu_wsl.sh` actuel règle les deux problèmes :
- `SCRIPT_DIR` → `cd "$SCRIPT_DIR"` → `python3 train.py` trouve bien `train.py` ✓
- `batch_size=8` (au lieu de 16) → VRAM ~4.8 GB (largement sous 13.7 GB) ✓

### Racine du blocage persistant
**La machine WSL2 n'a pas encore exécuté `git pull && bash run_gpu_wsl.sh`** depuis le cycle 11.
Les deux correctifs sont dans le dépôt mais pas encore appliqués sur la machine de production.

### Calcul d'impact batch=8 vs batch=16
| Paramètre | batch=16 (crashé) | batch=8 (correctif) |
|---|---|---|
| Steps/époque | ~368 | ~736 |
| Mémoire GPU (activations) | ~9.6 GB | ~4.8 GB |
| Temps/époque estimé | ~3 min | ~5–7 min |
| ETA 50 époques | ~2.5h | ~4–6h |

Avec batch=8, large marge sur 13.7 GB VRAM → aucun risque OOM attendu.

## Historique (10 dernières époques enregistrées)
| Époque | val_loss | val_panel_iou |
|--------|----------|---------------|
| —      | —        | —             |

*Aucune époque complète depuis 13 cycles de surveillance consécutifs.*

---

## Recommandations

### A. Données Malicounda

**Recommandation : NE PAS intégrer Malicounda maintenant — attendre ≥ 10 époques Patisen-GPU.**

Raisons :
1. **Blocage non résolu** : lancer un entraînement multi-site (9.4 GB) sans baseline stable aggrave le risque de crash
2. **Aucune val_iou connue** : impossible de mesurer l'apport de Malicounda sans référence Patisen
3. **Résolution hétérogène** : 1.7 cm/px (Malicounda) vs 3 cm/px (Patisen). Tuile 512 px = 8.7 m×8.7 m à Malicounda vs 15.4 m×15.4 m à Patisen. Les panneaux apparaissent ×1.76 plus grands — un `tile_size=256` ou une normalisation d'échelle sera nécessaire pour éviter le biais d'échelle

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

**AVERTISSEMENT** : si val_iou plafonne à ~19–20% comme lors du run `patisen/` (200 époques),
ce n'est pas un problème de capacité modèle mais un **problème de split de données**
(tuiles adjacentes partagées entre train et val → fuite spatiale). Vérifier que `train.py`
utilise un split spatial par blocs géographiques disjoints, pas un split aléatoire par tuile.

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
| `batch_size` | **8** (corrigé cycle 11) | OK. Si OOM persiste → réduire à **4** |
| `panel_weight` | 15.0 | Maintenir. Augmenter à **20–25** si val IoU < 0.40 après époque 10 |
| `panel_oversample` | 4 | OK (4.0 panel tiles/batch avec batch=8). Augmenter à **6–8** si IoU stagne |
| `tile_size` | 512 px | 15.4 m×15.4 m à Patisen. Correct. Envisager 256 px pour Malicounda (1.7 cm/px) |
| `lr` | 0.0001 | Correct. Ajouter `ReduceLROnPlateau(patience=5)` si plateau détecté |
| `stride` | 256 (overlap 50%) | Correct. |

**Action prioritaire sur WSL2 (CRITIQUE — 13e cycle sans démarrage) :**
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

Si le `.err` contient `OOM` ou `ResourceExhausted` → réduire à `batch_size=4` dans `run_gpu_wsl.sh`.

---

## Décision

**13e cycle — EN ATTENTE DE RELANCE — correctifs batch=8 + SCRIPT_DIR confirmés dans le dépôt.**

Diagnostic consolidé :
- `training_log.csv` : **0 octet** — 0 époque complète (13e cycle consécutif) ✗
- `train_log.txt` : modèle chargé, tuiles générées, `Epoch 1/50` amorcée, jamais complétée ✗
- `train_log.txt.err` : erreur `/home/solar/train.py` = **artefact pré-fix (cycle 9)** — non bloquante pour le run actuel ✓
- `train.py` : **présent dans le dépôt git** (confirmé cycle 13) ✓
- `run_gpu_wsl.sh` : `SCRIPT_DIR` + `batch_size=8` → prêt à l'emploi ✓

**Action requise (unique) sur la machine WSL2 :**
```
git pull origin main && nohup bash run_gpu_wsl.sh > ... &
```

---

### Historique des cycles de surveillance

| Cycle | État |
|-------|------|
| 1  | training started — 0 époque |
| 2  | 0 époque, démarrage en cours |
| 3  | 0 époque, alerte délai 24h |
| 4  | 0 époque, crash confirmé |
| 5  | 0 époque, intervention requise |
| 6  | 0 époque — BLOCAGE TOTAL |
| 7  | 0 époque — INTERVENTION URGENTE |
| 8  | 0 époque — 8e cycle |
| 9  | CORRECTIF `run_gpu_wsl.sh` — chemin dynamique (`SCRIPT_DIR`) |
| 10 | En attente relance — `train.py` atteint, époque 1 incomplète |
| 11 | CORRECTIF `batch_size=16→8` — relance requise |
| 12 | EN ATTENTE RELANCE — batch=8 confirmé, 0 époque |
| **13** | **EN ATTENTE RELANCE — `train.py` confirmé dans dépôt, correctifs prêts** |
