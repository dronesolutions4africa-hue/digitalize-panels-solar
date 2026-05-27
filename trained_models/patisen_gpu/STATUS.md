# Rapport de surveillance — 2026-05-27

## État : DÉMARRAGE — BLOCAGE ÉPOQUE 1 (23e cycle)
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

## Diagnostic cycle 23

### Situation vs cycle 22
**Aucun progrès détecté — situation strictement identique au cycle 22 :**
- `training_log.csv` : **0 octet** (0 époque complète) — inchangé depuis le cycle 1
- `train_log.txt` (347 lignes) : se termine à `[Train] epochs=50  batch=16  steps/epoch~367` puis `Epoch 1/50` — batch=16 confirme que la machine n'a pas encore appliqué le correctif batch=8
- `train_log.txt.err` : `python3: can't open file '/home/solar/train.py': [Errno 2] No such file or directory` (artefact d'une tentative antérieure à chemin erroné)

### ALERTE CRITIQUE — Analyse run historique `patisen` (200 époques)
Le répertoire `trained_models/patisen/training_log.csv` révèle un problème de fond :

| Époque | train_panel_iou | val_panel_iou | val_loss |
|--------|-----------------|---------------|----------|
| 0      | 0.7251          | 0.1893        | 1.7164   |
| 1      | 0.8125          | 0.1951        | 1.1170   |
| 185    | 0.9023          | 0.1974        | 0.7873   |
| 190    | 0.9064          | 0.1974        | 0.7908   |
| 195    | 0.9051          | 0.1974        | 0.7849   |
| 199    | 0.9024          | 0.1974        | 0.7845   |

**Conclusion** : Fast SCNN avec Patisen seul atteint 0.90 en train mais plafonne à **0.197 en validation** après 200 époques. Surapprentissage massif. Le modèle mémorise le site Patisen sans généraliser.

Implication directe : même si le run GPU se débloque avec batch=8, répéter le schéma mono-site Patisen ne permettra **jamais** d'atteindre val IoU >= 0.85. **L'intégration de Malicounda est indispensable, pas optionnelle.**

### Racine du blocage GPU actuel
Le run `patisen_gpu` tourne avec `batch=16`, causant vraisemblablement un OOM silencieux (RTX A4500 13.7 GB, activations ~9.6 GB pour batch=16 + overhead). L'époque 1 ne se termine jamais.

**Le dépôt contient le correctif depuis le cycle 11 (`batch_size 16→8` + `SCRIPT_DIR`) — confirmé dans `run_gpu_wsl.sh`. La machine WSL2 n'a pas encore exécuté `git pull && bash run_gpu_wsl.sh`.**

### Calcul mémoire GPU
| batch_size | Mémoire activations (estimée) | Statut |
|---|---|---|
| 16 | ~9.6 GB | RISQUE OOM sur 13.7 GB VRAM |
| **8** | ~4.8 GB | SÛRE — marge 9 GB |
| 4 | ~2.4 GB | SÛRE si OOM persiste avec batch=8 |

## Historique (10 dernières époques enregistrées — run GPU)
| Époque | val_loss | val_panel_iou |
|--------|----------|---------------|
| —      | —        | —             |

*Aucune époque complète depuis 23 cycles de surveillance consécutifs.*

---

## Recommandations

### A. Données Malicounda

**Recommandation PRIORITAIRE : Lancer directement le multi-site Patisen + Malicounda dès que le correctif batch=8 est appliqué.**

La donnée historique (`patisen`, 200 époques, val IoU = 0.197) est sans équivoque : Fast SCNN sur Patisen seul ne peut pas atteindre 0.85. Malicounda (86 280 panneaux annotés, 1.7 cm/px) est le levier essentiel pour la généralisation.

**Points de vigilance avant intégration :**
- Résolution hétérogène : 1.7 cm/px (Malicounda) vs 3 cm/px (Patisen). Tuile 512 px = 8.7 m×8.7 m à Malicounda vs 15.4 m×15.4 m à Patisen. Les panneaux y apparaissent ×1.76 plus grands en pixels.
- Recommandé : utiliser `--tile_size 256` pour Malicounda OU normaliser l'échelle avec un resize à 3 cm/px avant tuilage
- `--max_tiles_per_site 5000` équilibre le poids de chaque site dans les batches

**Commande multi-site recommandée (à lancer après git pull) :**
```bash
python3 train.py \
  --ortho Data/Orthomosaic_Patisen.tif Malicounda/ortho.tif \
  --shp Data/Panneaux_Patisen.shp Malicounda/Lim_panneaux.shp \
  --tile_size 512 --stride 256 --batch_size 8 --epochs 50 \
  --panel_oversample 6 --panel_weight 20 --max_tiles_per_site 5000 \
  --output_dir trained_models/patisen_malicounda_gpu
```

### B. Stratégie pour atteindre IoU >= 0.85

**Contexte historique enrichi** : le run `patisen` (200 époques, Fast SCNN, mono-site) montre val IoU stagnant à 0.197. Ce n'est pas un problème de LR ou d'hyperparamètres — c'est un manque de diversité des données.

Arbre de décision à appliquer dès époque 10 du run multi-site :

| Condition à l'époque 10 (multi-site) | Action recommandée |
|---|---|
| val IoU > 0.70 | Continuer Fast SCNN 50 époques — objectif 0.85 atteignable |
| 0.50 ≤ val IoU ≤ 0.70 | Ajouter `ReduceLROnPlateau(patience=5, factor=0.5)` ; réévaluer à époque 20 |
| val IoU < 0.50 | **Escalader immédiatement vers U-Net + ResNet50 ImageNet multi-site** |
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
| `batch_size` | **16** (en cours) → **8** (correctif à appliquer) | **Urgent** : appliquer le correctif batch=8 via git pull |
| `panel_weight` | 15.0 | Augmenter à **20–25** — val IoU historique de 0.197 confirme que 15 est insuffisant pour compenser le déséquilibre 10.9% panneaux |
| `panel_oversample` | 4 | Augmenter à **6–8** dès le run multi-site pour garantir des panneaux dans chaque batch |
| `tile_size` | 512 px (15.4 m×15.4 m à Patisen) | Correct pour Patisen. Envisager `tile_size=256` pour Malicounda (1.7 cm/px → 4.3 m×4.3 m, mieux adapté à la taille des panneaux) |
| `lr` | 0.0001 | Correct. Ajouter `ReduceLROnPlateau(patience=5, factor=0.5)` dès le prochain run |
| `stride` | 256 (overlap 50%) | Correct |

---

## Décision

**23e cycle — EN ATTENTE DE RELANCE — situation inchangée depuis le cycle 22.**

**Conclusion stratégique** : ne pas relancer un run Patisen-seul. Appliquer batch=8 ET lancer directement le run multi-site (Patisen + Malicounda).

**Action requise (unique) sur la machine WSL2 :**
```bash
cd /home/solar/digitalize-panels-solar
git pull origin main

# Option A — Patisen seul (pour débloquer rapidement, valider batch=8)
nohup bash run_gpu_wsl.sh \
  > trained_models/patisen_gpu/train_log.txt \
  2> trained_models/patisen_gpu/train_log.txt.err &
echo "PID=$!"

# Option B — Multi-site immédiat (recommandé vu val IoU historique 0.197)
nohup python3 train.py \
  --ortho Data/Orthomosaic_Patisen.tif Malicounda/ortho.tif \
  --shp Data/Panneaux_Patisen.shp Malicounda/Lim_panneaux.shp \
  --tile_size 512 --stride 256 --batch_size 8 --epochs 50 \
  --panel_oversample 6 --panel_weight 20 --max_tiles_per_site 5000 \
  --output_dir trained_models/patisen_malicounda_gpu \
  > trained_models/patisen_malicounda_gpu/train_log.txt \
  2> trained_models/patisen_malicounda_gpu/train_log.txt.err &
echo "PID=$!"

# Vérifier absence d'OOM après 3 min
sleep 180 && cat trained_models/patisen_gpu/train_log.txt.err
# Vérifier époque 1 complète après 30 min
sleep 1800 && head -3 trained_models/patisen_gpu/training_log.csv
```

Si `.err` contient `OOM` ou `ResourceExhausted` → réduire à `batch_size=4`.

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
| 17 | 2026-05-23 | EN ATTENTE RELANCE — situation identique au cycle 16 |
| 18 | 2026-05-23 | EN ATTENTE RELANCE — ALERTE : val IoU historique 0.197/200 éps — multi-site indispensable |
| 19 | 2026-05-26 | EN ATTENTE RELANCE — situation inchangée — correctif batch=8 non appliqué sur WSL2 |
| 20 | 2026-05-26 | EN ATTENTE RELANCE — situation inchangée — correctif batch=8 non appliqué sur WSL2 |
| 21 | 2026-05-26 | EN ATTENTE RELANCE — situation inchangée — correctif batch=8 non appliqué sur WSL2 |
| 22 | 2026-05-26 | EN ATTENTE RELANCE — situation inchangée — correctif batch=8 non appliqué sur WSL2 |
| **23** | **2026-05-27** | **EN ATTENTE RELANCE — situation inchangée — correctif batch=8 non appliqué sur WSL2** |
