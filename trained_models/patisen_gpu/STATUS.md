# Rapport de surveillance — 2026-05-21

## État : DÉMARRAGE — EN ATTENTE DE RELANCE (10e cycle)
- Époque : 0/50 (aucune époque complète enregistrée)
- Meilleure val IoU panneaux : N/A
- Meilleure val loss : N/A
- Tendance (5 dernières époques) : N/A (aucune donnée)
- ETA estimée : inconnue — aucune époque complétée

## Contexte du run (données du log)
- GPU : NVIDIA RTX A4500 Laptop (13.7 GB VRAM) — `cuda_malloc_async`
- Données chargées : `Orthomosaic_Patisen.tif` (18 695×16 883 px, chargé en 22.5s)
- Masque : 34 445 277 px panneaux / 315 627 685 total (**10.9%**)
- Tuiles : 3 687 brutes → 5 886 oversamplées (train), 921 (val)
- Panneaux/batch : **8.0 / 16** (`panel_oversample=4` actif)
- Modèle : Fast SCNN v2 — 1 901 450 params (7.25 MB)
- Hyperparamètres : `lr=0.0001`, `panel_weight=15.0`, `batch=16`, `steps/epoch≈367`

## Diagnostic du cycle 10

### Avancée constatée depuis le cycle 9
Le correctif `run_gpu_wsl.sh` (chemin dynamique `SCRIPT_DIR`) a été commité et pushé au cycle 9
(commit `93d63f5`). Le contenu actuel de `train_log.txt` confirme que `train.py` a bien été
atteint lors d'une exécution : GPU détecté, ortho chargée, architecture modèle affichée, et la
ligne `Epoch 1/50` est présente. C'est un progrès par rapport aux cycles 1–8.

### Blocage actuel
`training_log.csv` reste à **0 octet** — l'époque 1 n'a pas été menée à terme.
`train_log.txt.err` contient encore l'ancienne erreur :
```
python3: can't open file '/home/solar/train.py': [Errno 2] No such file or directory
```
Cela indique que la version corrigée du script (`SCRIPT_DIR`) n'a **pas encore été relancée**
sur la machine WSL2 — ou qu'une version non patchée a été utilisée en parallèle.

### Hypothèse — plantage en cours d'époque 1
Si le script corrigé a bien été relancé, un second problème peut provoquer l'arrêt en cours
d'époque 1 sans écrire dans le CSV :
- OOM GPU (batch=16 sur 5 886 tuiles × 512 px peut saturer 13.7 GB en fp32)
- Erreur Python non capturée lors du premier `model.fit` step
- Conflit CUDA / TF version sous WSL2

## Action immédiate — Checklist en 3 étapes

```bash
# ÉTAPE 1 — Puller le correctif (depuis la machine WSL2)
cd /chemin/vers/digitalize-panels-solar
git pull origin main

# ÉTAPE 2 — Relancer avec logs séparés
bash run_gpu_wsl.sh \
  > trained_models/patisen_gpu/train_log.txt \
  2> trained_models/patisen_gpu/train_log.txt.err &
echo "PID=$!"

# ÉTAPE 3 — Vérifier après 120 secondes
sleep 120 && echo "--- stdout ---" && tail -20 trained_models/patisen_gpu/train_log.txt
sleep 120 && echo "--- stderr ---" && cat trained_models/patisen_gpu/train_log.txt.err
sleep 120 && echo "--- CSV ---" && cat trained_models/patisen_gpu/training_log.csv
```

Si `train_log.txt.err` contient `OOM` ou `ResourceExhausted` → réduire `batch_size` de 16 à 8 :
```bash
# Version batch réduit (si OOM)
python3 train.py \
  --model fast_scnn_2 \
  --ortho Data/Orthomosaic_Patisen.tif \
  --shp Data/Panneaux_Patisen.shp \
  --tile_size 512 --stride 256 --batch_size 8 --epochs 50 \
  --panel_oversample 4 \
  --output_dir trained_models/patisen_gpu
```

## Historique (10 dernières époques)
| Époque | val_loss | val_panel_iou |
|--------|----------|---------------|
| —      | —        | —             |

*Aucune époque complète dans `training_log.csv` (fichier vide, 0 octets) depuis le cycle 1.*

## Recommandations

### A. Données Malicounda
Malicounda (86 280 panneaux annotés, 1.7 cm/px, 9.4 GB) représente un apport massif en volume
annoté. **L'intégration multi-site reste recommandée APRÈS un run Patisen-seul fonctionnel** :

1. **Baseline Patisen d'abord** : résolution 3 cm vs 1.7 cm = deux distributions d'entrée
   distinctes. Établir la baseline Patisen seul permet de quantifier l'apport de Malicounda.
2. **Densité panneau différente** : à 1.7 cm/px, un tile 512 px = 8.7 m × 8.7 m, soit ×1.76
   plus de pixels par panneau qu'à Patisen. Le modèle doit d'abord converger sur un site.
3. **Mémoire** : limiter via `--max_tiles_per_site 5000` pour éviter OOM.

**Commande multi-site (après fin run Patisen) :**
```bash
python3 train.py \
  --ortho Data/Orthomosaic_Patisen.tif Malicounda/ortho.tif \
  --shp Data/Panneaux_Patisen.shp Malicounda/Lim_panneaux.shp \
  --tile_size 512 --stride 256 --batch_size 8 --epochs 50 \
  --panel_oversample 4 --max_tiles_per_site 5000 \
  --output_dir trained_models/patisen_malicounda_gpu
```

### B. Stratégie pour atteindre IoU >= 0.85

Décision conditionnelle à l'époque 10 (dès que l'entraînement démarre) :

| Condition après époque 10 | Action recommandée |
|---|---|
| val IoU > 0.70 | Continuer Fast SCNN ; lancer multi-site Patisen+Malicounda ensuite |
| 0.50 ≤ val IoU ≤ 0.70 | Ajouter `ReduceLROnPlateau(patience=5, factor=0.5)` ; réévaluer à l'époque 20 |
| val IoU < 0.50 | Escalader directement vers **U-Net + ResNet50 ImageNet** multi-site |
| Stagnation ≥ 5 époques | Réduire LR × 0.5 ou escalader vers U-Net |

**Commande U-Net de secours (si IoU < 0.50 à l'époque 10) :**
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
| `panel_weight` | 15.0 | Maintenir. Augmenter à **20–25** si val IoU < 0.40 après époque 10 |
| `panel_oversample` | 4 | OK — 8.0 panneaux/batch. Augmenter à **6–8** si IoU stagne |
| `batch_size` | 16 (Patisen seul) | Réduire à **8** si erreur OOM au démarrage de l'époque 1 |
| `tile_size` | 512 px | Adapté aux deux sites (8.7 m × 8.7 m à Malicounda). Pas de changement |
| `lr` | 0.0001 | Correct. Ajouter `ReduceLROnPlateau(patience=5)` si plateau détecté |
| `stride` | 256 (overlap 50%) | Correct. Réduire à `stride=384` si temps/époque trop long |

## Décision

**10e cycle de surveillance — EN ATTENTE DE RELANCE.**

- `run_gpu_wsl.sh` : correctif chemin dynamique appliqué au cycle 9 (commit `93d63f5`) ✓
- `train_log.txt` : GPU détecté + ortho chargée + architecture affichée + `Epoch 1/50` lancée ✓
- `training_log.csv` : **0 octet** — époque 1 non complétée ✗
- `train_log.txt.err` : erreur ancienne encore présente — script corrigé non relancé ✗

**Action requise sur la machine WSL2 :**
1. `git pull origin main` pour récupérer `run_gpu_wsl.sh` corrigé
2. `bash run_gpu_wsl.sh > ... 2> ... &`
3. Vérifier après 120s que `training_log.csv` contient la première ligne de métriques

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
| 9 | `93d63f5` | CORRECTIF `run_gpu_wsl.sh` — chemin dynamique appliqué |
| **10** | **ce commit** | **En attente de relance — train.py atteint mais époque 1 incomplète** |
