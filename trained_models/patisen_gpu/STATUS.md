# Rapport de surveillance — 2026-05-22

## État : DÉMARRAGE — BLOCAGE ÉPOQUE 1 (11e cycle)
- Époque : 0/50 (aucune époque complète enregistrée dans `training_log.csv`)
- Meilleure val IoU panneaux : N/A
- Meilleure val loss : N/A
- Tendance (5 dernières époques) : N/A (aucune donnée)
- ETA estimée : inconnue — époque 1 non complétée

## Contexte du run GPU (données du log)
- GPU : NVIDIA RTX A4500 Laptop (13.7 GB VRAM) — `cuda_malloc_async`
- Données chargées : `Orthomosaic_Patisen.tif` (18 695×16 883 px, chargé en 22.5s)
- Masque : 34 445 277 px panneaux / 315 627 685 total (**10.9%**)
- Tuiles : 3 687 brutes → 5 886 oversamplées (train), 921 (val)
- Panneaux/batch : **8.0 / 16** (`panel_oversample=4`)
- Modèle : Fast SCNN v2 — 1 901 450 params (7.25 MB)
- Hyperparamètres : `lr=0.0001`, `panel_weight=15.0`, `batch=16→8 (corrigé)`, `steps/epoch≈367`

## Diagnostic cycle 11

### Avancée vs cycle 10
Aucun progrès dans `training_log.csv` — fichier toujours à **0 octet**.
`train_log.txt` (347 lignes) se termine identiquement à `Epoch 1/50`.
Le correctif `SCRIPT_DIR` du cycle 9 (commit `93d63f5`) est bien présent dans `run_gpu_wsl.sh`.

### Cause probable du crash epoch 1 — batch_size=16
`run_gpu_wsl.sh` avait **batch_size=16** (non conforme à la consigne batch_size=8).
Avec 512×512 tuiles en fp32 et batch=16 :
- Activations Fast SCNN branch pooling (576 canaux, 128×128) : **~600 MB × 16 = ~9.6 GB**
- Gradients + état Adam (×3 les paramètres) ≈ **100 MB**
- Risk OOM réel même avec 13.7 GB VRAM si les activations ne sont pas libérées entre couches

**Correctif appliqué ce cycle (commit 11)** : `batch_size=16 → 8` dans `run_gpu_wsl.sh`.

### Alerte critique — overfitting pathologique run précédent (`patisen/`, 200 époques)

| Métrique | Époque 0 | Époque 10 | Époque 50 | Époque 199 |
|---|---|---|---|---|
| `train_panel_iou` | 0.725 | ~0.875 | ~0.900 | 0.902 |
| `val_panel_iou` | 0.189 | **0.195** | **0.197** | **0.197** |
| `val_loss` | 1.716 | ~1.02 | ~0.80 | 0.785 |

**La val_panel_iou ne dépasse JAMAIS 19.7% en 200 époques** malgré train_iou à 90.2%.
Ce comportement n'est PAS un simple overfitting — c'est probablement :
1. **Split spatial manquant** : les tuiles train/val sont tirées aléatoirement sur l'image entière.
   Les tuiles adjacentes partagent des pixels → fuite de données en train → val artificiel
2. **Biais de classe en validation** : si les tuiles val sont majoritairement sans panneaux,
   l'IoU panneaux est structurellement bas
3. **Méthode de métrique** : possible bug de calcul IoU sur la classe minority en val

Le **run GPU actuel** (`patisen_gpu/`) utilise un `train.py` reécrit avec `panel_weight=15` et
`panel_oversample=4` — ces mécanismes devraient partiellement contrer le biais, **mais si le split
reste aléatoire**, le plafond val_iou risque d'être similaire.

## Historique (10 dernières époques enregistrées)
| Époque | val_loss | val_panel_iou |
|--------|----------|---------------|
| —      | —        | —             |

*Aucune époque complète dans `training_log.csv` depuis le cycle 1 (11 cycles de surveillance).*

---

## Recommandations

### A. Données Malicounda

**Recommandation : attendre un run Patisen-GPU fonctionnel (≥10 époques) avant d'intégrer Malicounda.**

Raisons :
1. **Blocage technique non résolu** : si l'epoch 1 crashe, lancer un second site (9.4 GB + GPU) aggravera la situation
2. **Baseline nécessaire** : sans val_iou connue sur Patisen seul, impossible de mesurer l'apport de Malicounda
3. **Résolution différente** : 1.7 cm/px vs 3 cm/px → tuile 512 px = 8.7 m×8.7 m à Malicounda contre 15.4 m×15.4 m à Patisen ; le modèle voit des panneaux de taille apparente ×1.76 plus grande — nécessite une normalisation dédiée ou un tile_size adapté (ex: 256 px pour égaliser l'emprise réelle)

**Commande multi-site (à lancer APRÈS ≥10 époques Patisen-GPU fonctionnelles) :**
```bash
python3 train.py \
  --ortho Data/Orthomosaic_Patisen.tif Malicounda/ortho.tif \
  --shp Data/Panneaux_Patisen.shp Malicounda/Lim_panneaux.shp \
  --tile_size 512 --stride 256 --batch_size 8 --epochs 50 \
  --panel_oversample 4 --max_tiles_per_site 5000 \
  --output_dir trained_models/patisen_malicounda_gpu
```

### B. Stratégie pour atteindre IoU >= 0.85

Décision conditionnelle dès que l'entraînement démarre :

| Condition à l'époque 10 | Action recommandée |
|---|---|
| val IoU > 0.70 | Continuer Fast SCNN 50 époques ; envisager multi-site Malicounda ensuite |
| 0.50 ≤ val IoU ≤ 0.70 | Ajouter `ReduceLROnPlateau(patience=5, factor=0.5)` ; réévaluer à l'époque 20 |
| val IoU < 0.50 | **Escalader vers U-Net + ResNet50 ImageNet** multi-site (voir commande ci-dessous) |
| val IoU stagne ≥ 5 époques | Réduire LR × 0.5 ou escalader vers U-Net |

**AVERTISSEMENT** : si val_iou plafonne à ~19–20% comme lors du run `patisen/`, ce n'est pas
un problème de capacité modèle mais un **problème de split de données**. Dans ce cas :
- Vérifier que `train.py` utilise un split **spatial** (blocs géographiques disjoints) plutôt qu'aléatoire
- Vérifier la distribution class-balance dans les tuiles val (% pixels panneaux)

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
| `batch_size` | ~~16~~ → **8** | **CORRIGÉ ce cycle** — 16 était probablement cause du crash OOM epoch 1 |
| `panel_weight` | 15.0 | Maintenir. Augmenter à **20–25** si val IoU < 0.40 après époque 10 |
| `panel_oversample` | 4 | OK — 8.0 panneaux/batch. Augmenter à **6–8** si IoU stagne |
| `tile_size` | 512 px | 15.4 m×15.4 m à Patisen (3 cm/px). Adapté. Envisager 256 px à Malicounda (1.7 cm/px) |
| `lr` | 0.0001 | Correct. Ajouter `ReduceLROnPlateau(patience=5)` si plateau détecté |
| `stride` | 256 (overlap 50%) | Correct. |

**Action prioritaire** : relancer `run_gpu_wsl.sh` (batch corrigé) avec **stderr capturé** :
```bash
cd /chemin/vers/digitalize-panels-solar
git pull origin main  # récupérer batch_size=8
bash run_gpu_wsl.sh \
  > trained_models/patisen_gpu/train_log.txt \
  2> trained_models/patisen_gpu/train_log.txt.err &
echo "PID=$!"
# Vérifier après 5 min
sleep 300 && tail -5 trained_models/patisen_gpu/train_log.txt && cat trained_models/patisen_gpu/train_log.txt.err
```

Si le nouveau `.err` contient `OOM` ou `ResourceExhausted` → réduire à `batch_size=4`.

---

## Décision

**11e cycle — CORRECTIF batch_size=8 APPLIQUÉ — EN ATTENTE DE RELANCE.**

Actions réalisées ce cycle :
- `run_gpu_wsl.sh` : `batch_size=16 → 8` ✓ (commit 11)
- `training_log.csv` : **0 octet** — époque 1 toujours non complétée ✗
- Alerte overfitting run précédent documentée ✓

Action requise sur la machine WSL2 :
1. `git pull origin main` pour récupérer `batch_size=8`
2. `bash run_gpu_wsl.sh > trained_models/patisen_gpu/train_log.txt 2> trained_models/patisen_gpu/train_log.txt.err &`
3. Vérifier après **5 min** : `cat trained_models/patisen_gpu/train_log.txt.err` (OOM ?) + `tail trained_models/patisen_gpu/train_log.txt`
4. Vérifier après **30 min** : `head -2 trained_models/patisen_gpu/training_log.csv` (époque 1 complétée ?)

---

### Historique des cycles de surveillance

| Cycle | Commit | État |
|-------|--------|------|
| 1 | `7a172db` | training started — 0 époque |
| 2 | `80289e4` | 0 époque, démarrage en cours |
| 3 | `f4ba46d` | 0 époque, alerte délai 24h |
| 4 | `2513929` | 0 époque, crash confirmé |
| 5 | `b36bd1c` | 0 époque, intervention requise |
| 6 | `b96c500` | 0 époque — BLOCAGE TOTAL |
| 7 | `f0620dc` | 0 époque — INTERVENTION URGENTE |
| 8 | `647d602` | 0 époque — 8e cycle |
| 9 | `93d63f5` | CORRECTIF `run_gpu_wsl.sh` — chemin dynamique |
| 10 | `c43d16f` | En attente relance — `train.py` atteint, époque 1 incomplète |
| **11** | **ce commit** | **CORRECTIF batch_size=16→8 — relance requise** |
