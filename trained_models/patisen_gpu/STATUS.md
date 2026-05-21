# Rapport de surveillance — 2026-05-21

## État : DÉMARRAGE — CRASH PERSISTANT (8e cycle sans progression)
- Époque : 0/50 (aucune époque complète enregistrée)
- Meilleure val IoU panneaux : N/A
- Meilleure val loss : N/A
- Tendance (5 dernières époques) : N/A (aucune donnée)
- ETA estimée : inconnue — entraînement crashe avant toute époque

## Contexte du run (données du log)
- GPU : NVIDIA RTX A4500 Laptop (13.7 GB VRAM) — `cuda_malloc_async`
- Données chargées : `Orthomosaic_Patisen.tif` (18 695×16 883 px, chargé en 22.5s)
- Masque : 34 445 277 px panneaux / 315 627 685 total (**10.9%**)
- Tuiles : 3 687 brutes → 5 886 oversamplées (train), 921 (val)
- Panneaux/batch : **8.0 / 16** (`panel_oversample=4` actif)
- Modèle : Fast SCNN v2 — 1 901 450 params (7.25 MB)
- Hyperparamètres : `lr=0.0001`, `panel_weight=15.0`, `batch=16`, `steps/epoch≈367`

## ALERTE CRITIQUE — Crash confirmé (8 cycles consécutifs)

```
python3: can't open file '/home/solar/train.py': [Errno 2] No such file or directory
```

| Cycle | Commit | État |
|-------|--------|------|
| 1 | `7a172db` | training started — 0 époque |
| 2 | `80289e4` | 0 époque, démarrage en cours |
| 3 | `f4ba46d` | 0 époque, alerte délai 24h |
| 4 | `2513929` | 0 époque, crash confirmé |
| 5 | `b36bd1c` | 0 époque, intervention requise |
| 6 | `b96c500` | 0 époque — BLOCAGE TOTAL |
| 7 | `f0620dc` | 0 époque — INTERVENTION URGENTE |
| **8** | **ce commit** | **0 époque — 8e CYCLE : CORRIGER LE CHEMIN train.py** |

**Cause racine confirmée :** `train.py` est appelé depuis `/home/solar/` mais il réside dans
le répertoire du projet cloné. Le `train_log.txt` (347 lignes) documente une initialisation
complète du data pipeline et de l'architecture modèle, mais s'arrête à `Epoch 1/50` sans
aucun step d'entraînement complété. Le `training_log.csv` reste vide (0 octets).

## Historique (10 dernières époques)
| Époque | val_loss | val_panel_iou |
|--------|----------|---------------|
| —      | —        | —             |

*Aucune époque complète dans `training_log.csv` (fichier vide, 0 octets) depuis le cycle 1.*

## Recommandations

### A. Action immédiate — Corriger le chemin et relancer (PRIORITÉ ABSOLUE)

**Cause racine :** `train.py` n'existe pas dans `/home/solar/`. Il existe dans la racine du
projet cloné (`digitalize-panels-solar/`). Le script de lancement (probablement `run_gpu_wsl.sh`
ou une tâche planifiée) pointe vers le mauvais répertoire.

```bash
# 1. Vérifier le script de lancement actuel
cat /home/solar/run_gpu_wsl.sh 2>/dev/null || echo "Script introuvable dans /home/solar/"

# 2. Trouver où se trouve train.py
find /home /root /opt -name "train.py" 2>/dev/null

# 3. Se placer dans le bon répertoire et relancer
cd /chemin/vers/digitalize-panels-solar   # adapter selon find ci-dessus
ls train.py   # doit répondre "train.py"

# 4. Relancer avec redirection propre
nohup python3 train.py \
  --ortho Data/Orthomosaic_Patisen.tif \
  --shp Data/Panneaux_Patisen.shp \
  --tile_size 512 --stride 256 --batch_size 16 --epochs 50 \
  --panel_oversample 4 \
  --output_dir trained_models/patisen_gpu \
  > trained_models/patisen_gpu/train_log.txt \
  2> trained_models/patisen_gpu/train_log.txt.err &

# 5. Vérifier après 60 secondes
sleep 60 && tail -5 trained_models/patisen_gpu/train_log.txt
sleep 60 && tail -5 trained_models/patisen_gpu/training_log.csv
```

**Alternative : corriger `run_gpu_wsl.sh` si c'est lui qui lance le processus :**
```bash
# Remplacer la ligne de lancement dans le script
# AVANT : python3 /home/solar/train.py ...
# APRÈS : python3 $(dirname "$0")/train.py ...
# OU :    cd /chemin/absolu/vers/projet && python3 train.py ...
```

### B. Données Malicounda

Malicounda (86 280 panneaux annotés, 1.7 cm/px, 9.4 GB) représente un apport massif.
**L'intégration multi-site est recommandée APRÈS un run Patisen-seul fonctionnel**, pour ces raisons :

1. **Baseline d'abord** : le run Patisen-seul fournit une référence propre avant d'introduire
   deux distributions d'entrée distinctes (résolution 3 cm vs 1.7 cm).
2. **Domaine différent** : à 1.7 cm/px, un tile 512 px couvre 8.7 m × 8.7 m — un panneau
   occupe proportionnellement ×1.76 plus de pixels qu'à Patisen. Le modèle doit d'abord
   converger sur Patisen avant d'absorber ce décalage de distribution.
3. **Mémoire** : 9.4 GB d'image source → limiter via `--max_tiles_per_site 5000`.

**Commande recommandée pour l'entraînement multi-site (après fix + fin run Patisen) :**
```bash
python3 train.py \
  --ortho Data/Orthomosaic_Patisen.tif Malicounda/ortho.tif \
  --shp Data/Panneaux_Patisen.shp Malicounda/Lim_panneaux.shp \
  --tile_size 512 --stride 256 --batch_size 8 --epochs 50 \
  --panel_oversample 4 --max_tiles_per_site 5000 \
  --output_dir trained_models/patisen_malicounda_gpu
```

### C. Stratégie pour atteindre IoU >= 0.85

Décision conditionnelle à l'époque 10 (après relance réussie) :

| Condition après époque 10 | Action recommandée |
|---|---|
| val IoU > 0.70 | Continuer Fast SCNN ; lancer multi-site Patisen+Malicounda ensuite |
| 0.50 ≤ val IoU ≤ 0.70 | Appliquer `ReduceLROnPlateau(patience=5, factor=0.5)` ; réévaluer à l'époque 20 |
| val IoU < 0.50 | Escalader directement vers **U-Net + ResNet50 ImageNet** multi-site |
| Stagnation ≥ 5 époques | Réduire LR × 0.5 ou escalader vers U-Net |

**Commande U-Net de secours (si escalade déclenchée après époque 10) :**
```bash
python3 train.py \
  --model unet_resnet50 \
  --ortho Data/Orthomosaic_Patisen.tif Malicounda/ortho.tif \
  --shp Data/Panneaux_Patisen.shp Malicounda/Lim_panneaux.shp \
  --tile_size 512 --stride 256 --batch_size 4 --epochs 50 \
  --panel_oversample 4 --max_tiles_per_site 5000 \
  --output_dir trained_models/unet_resnet50_multisite
```

### D. Hyperparamètres (à appliquer lors de la relance)

| Paramètre | Valeur actuelle | Recommandation |
|---|---|---|
| `panel_weight` | 15.0 | Maintenir. Augmenter à **20–25** si val IoU < 0.40 après époque 10 |
| `panel_oversample` | 4 | OK — 8.0 panneaux/batch sur 16. Augmenter à **6–8** si IoU stagne |
| `tile_size` | 512 px | Adapté aux deux sites (8.7 m × 8.7 m). Pas de changement nécessaire |
| `batch_size` | 16 (Patisen seul) | Réduire à **8** pour le multi-site (pression mémoire Malicounda) |
| `lr` | 0.0001 | Correct pour Fast SCNN. Ajouter `ReduceLROnPlateau(patience=5)` si plateau |
| `stride` | 256 (overlap 50%) | Correct. Réduire overlap à 25% (`stride=384`) si temps/époque trop long |

## Décision

**CRASH PERSISTANT — 8 cycles de monitoring consécutifs sans aucune époque complète.**

- `training_log.csv` : **0 octets** (vide depuis le début)
- `train_log.txt.err` : `No such file or directory` pour `train.py` depuis `/home/solar/`
- **Aucune décision d'escalade possible** sans données de validation
- **Seuil de décision** : époque 10 pour statuer sur Fast SCNN vs U-Net ResNet50
- **Action humaine requise IMMÉDIATEMENT** : corriger le chemin (`/home/solar/` → répertoire réel du projet) et relancer `train.py`
