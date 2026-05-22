# Rapport de surveillance — 2026-05-22

## État : DÉMARRAGE — BLOCAGE ÉPOQUE 1 (12e cycle)
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
- Hyperparamètres log actuel : `lr=0.0001`, `panel_weight=15.0`, `batch=16` (run AVANT fix)
- **Script `run_gpu_wsl.sh` corrigé** : `batch_size=8` depuis cycle 11 ✓

## Diagnostic cycle 12

### Avancée vs cycle 11
Aucun progrès dans `training_log.csv` — fichier toujours à **0 octet**.
`train_log.txt` (347 lignes) se termine identiquement à `Epoch 1/50`.

### Analyse du blocage
Le `train_log.txt` actuel indique `batch=16` et `steps/epoch~367`, confirmant que le log
correspond au run **AVANT** le correctif `batch_size=8` du cycle 11.
Le script `run_gpu_wsl.sh` contient maintenant `--batch_size 8` (vérifié ce cycle).

**La machine WSL2 n'a pas encore relancé `run_gpu_wsl.sh` après le `git pull` du cycle 11.**

### Calcul d'impact batch=8 vs batch=16
| Paramètre | batch=16 (crashé) | batch=8 (correctif) |
|---|---|---|
| Steps/époque | ~368 | ~736 |
| Panel tiles/batch | 8.0 | **4.0** |
| Mémoire GPU (activations) | ~9.6 GB | ~4.8 GB |
| Temps/époque estimé | ~3 min | ~5–7 min |
| ETA 50 époques | ~2.5h | ~4–6h |

Avec batch=8, les activations restent sous ~5 GB → large marge sur 13.7 GB VRAM.

### Alerte persistante — overfitting run précédent (`patisen/`, 200 époques)
La val_panel_iou n'a jamais dépassé **19.7%** en 200 époques (train_iou : 90.2%).
Cause probable : **split spatial manquant** (tuiles train/val adjacentes → fuite de données).
Le run `patisen_gpu/` intègre `panel_weight=15` et `panel_oversample=4` pour corriger le
biais de classe, mais la question du split spatial reste critique à surveiller.

## Historique (10 dernières époques enregistrées)
| Époque | val_loss | val_panel_iou |
|--------|----------|---------------|
| —      | —        | —             |

*Aucune époque complète depuis 12 cycles de surveillance consécutifs.*

---

## Recommandations

### A. Données Malicounda

**Recommandation : NE PAS intégrer Malicounda maintenant — attendre ≥ 10 époques Patisen-GPU.**

Raisons :
1. **Blocage non résolu** : relancer une session multi-site (9.4 GB) sans baseline stable aggrave le risque
2. **Aucune val_iou connue** : impossible de mesurer l'apport de Malicounda sans référence Patisen
3. **Résolution hétérogène** : 1.7 cm/px (Malicounda) vs 3 cm/px (Patisen) → tuile 512 px = 8.7 m×8.7 m vs 15.4 m×15.4 m. Les panneaux apparaissent ×1.76 plus grands à Malicounda. Un `tile_size=256` ou une normalisation d'échelle sera nécessaire

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

**AVERTISSEMENT** : si val_iou plafonne à ~19–20% comme lors du run `patisen/`, ce n'est
pas un problème de capacité modèle mais un **problème de split de données** (tuiles adjacentes
partagées entre train et val). Vérifier que `train.py` utilise un split spatial par blocs
géographiques disjoints, pas un split aléatoire.

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
| `panel_oversample` | 4 | OK (4.0 panel/batch avec batch=8). Augmenter à **6–8** si IoU stagne |
| `tile_size` | 512 px | 15.4 m×15.4 m à Patisen. Correct. Envisager 256 px pour Malicounda (1.7 cm/px) |
| `lr` | 0.0001 | Correct. Ajouter `ReduceLROnPlateau(patience=5)` si plateau détecté |
| `stride` | 256 (overlap 50%) | Correct. |

**Action prioritaire sur WSL2** :
```bash
cd /home/solar/digitalize-panels-solar
git pull origin main
bash run_gpu_wsl.sh \
  > trained_models/patisen_gpu/train_log.txt \
  2> trained_models/patisen_gpu/train_log.txt.err &
echo "PID=$!"
# Vérifier après 5 min
sleep 300 && tail -5 trained_models/patisen_gpu/train_log.txt
cat trained_models/patisen_gpu/train_log.txt.err
```

Si le `.err` contient `OOM` ou `ResourceExhausted` → réduire à `batch_size=4`.

---

## Décision

**12e cycle — EN ATTENTE DE RELANCE — batch_size=8 confirmé dans `run_gpu_wsl.sh`.**

Actions réalisées ce cycle :
- `training_log.csv` : **0 octet** — époque 1 toujours non complétée ✗
- `run_gpu_wsl.sh` : `batch_size=8` confirmé (fix cycle 11 actif) ✓
- Log actuel (`batch=16`) confirme crash pré-fix ✓
- Alerte split spatial maintenue ✓

Action requise sur la machine WSL2 :
1. `git pull origin main` pour récupérer le correctif `batch_size=8`
2. Relancer `run_gpu_wsl.sh` avec redirection stdout + stderr
3. Vérifier après **5 min** : `cat trained_models/patisen_gpu/train_log.txt.err` (OOM ?)
4. Vérifier après **30 min** : `head -2 trained_models/patisen_gpu/training_log.csv` (époque 1 ?)

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
| **12** | **EN ATTENTE RELANCE — batch=8 confirmé, 0 époque (ce cycle)** |
