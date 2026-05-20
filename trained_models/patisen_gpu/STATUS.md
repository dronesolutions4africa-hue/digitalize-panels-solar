# Rapport de surveillance — 2026-05-20

## État : DÉMARRAGE — EN ATTENTE DE DONNÉES (⚠️ CRASH PROBABLE)
- Époque : 0/50 (aucune époque complète enregistrée)
- Meilleure val IoU panneaux : N/A
- Meilleure val loss : N/A
- Tendance (5 dernières époques) : N/A
- ETA estimée : inconnue (première époque non terminée)

## Contexte de démarrage (données du run actuel)
- GPU : NVIDIA RTX A4500 Laptop (13.7 GB VRAM) — `cuda_malloc_async`
- Données chargées : Orthomosaic_Patisen.tif (18 695×16 883 px)
- Masque : 34 445 277 px panneaux / 315 627 685 total (**10.9%**)
- Tuiles : 3 687 brutes → 5 886 oversamplées (train), 921 (val)
- Panneaux/batch : **8.0 / 16** (panel_oversample=4 actif)
- Modèle : Fast SCNN v2 — 1 901 450 params (7.25 MB)
- Hyperparamètres : lr=0.0001, panel_weight=15.0, batch=16, steps/epoch≈367
- Statut log : `training_log.csv` vide — `train_log.txt` s'arrête à `Epoch 1/50`

## ⚠️ ALERTE — Crash probable confirmé par `train_log.txt.err`
```
python3: can't open file '/home/solar/train.py': [Errno 2] No such file or directory
```
**Interprétation :** le script a été invoqué depuis `/home/solar/` mais `train.py`
n'existe pas à cet emplacement. Le processus a crashé dès le lancement (ou à la relance).
Le `train_log.txt` (347 lignes, architecture + data loading + `Epoch 1/50`) correspond
à un run antérieur ou à une initialisation partielle qui s'est arrêtée avant toute époque.

3 cycles de monitoring consécutifs (commits 7a172db → 80289e4 → f4ba46d)
sans progression confirment qu'aucune époque ne s'est complétée depuis le 2026-05-19.

## Historique (10 dernières époques)
| Époque | val_loss | val_panel_iou |
|--------|----------|---------------|
| —      | —        | —             |

*Aucune époque complète dans `training_log.csv`.*

## Recommandations

### A. Données Malicounda
Malicounda (86 280 panneaux annotés, 1.7 cm/px, 9.4 GB) représente un apport massif.
**L'intégration multi-site est recommandée APRÈS un run Patisen-seul fonctionnel**, pour ces raisons :

1. **Baseline d'abord** : le run Patisen-seul fournit une référence propre avant d'introduire
   deux distributions d'entrée distinctes.
2. **Domaine différent** : résolution 1.7 cm/px vs ~3 cm/px — un panneau occupe
   proportionnellement plus de pixels à Malicounda (×1.76). Le modèle doit d'abord
   converger sur Patisen.
3. **Poids mémoire** : 9.4 GB d'image source → tiles 512×512 → limiter via
   `--max_tiles_per_site 5000` pour ne pas saturer la RAM/VRAM.
4. **Tile size** : à 1.7 cm/px, un tile 512 px couvre 8.7 m × 8.7 m — cohérent
   avec la taille réelle des panneaux. Aucun changement de `tile_size` requis.

**Commande recommandée pour l'entraînement multi-site (après fix + fin run Patisen) :**
```bash
python3 train.py \
  --ortho Data/Orthomosaic_Patisen.tif Malicounda/ortho.tif \
  --shp Data/Panneaux_Patisen.shp Malicounda/Lim_panneaux.shp \
  --tile_size 512 --stride 256 --batch_size 8 --epochs 50 \
  --panel_oversample 4 --max_tiles_per_site 5000 \
  --output_dir trained_models/patisen_malicounda_gpu
```

### B. Stratégie pour atteindre IoU >= 0.85
Aucune métrique disponible. Décision conditionnelle à l'époque 10 :

| Condition après époque 10 | Action recommandée |
|---|---|
| val IoU > 0.70 | Continuer Fast SCNN ; lancer multi-site Patisen+Malicounda ensuite |
| 0.50 ≤ val IoU ≤ 0.70 | Appliquer `ReduceLROnPlateau(patience=5, factor=0.5)` ; réévaluer à l'époque 20 |
| val IoU < 0.50 | Escalader directement vers **U-Net + ResNet50 ImageNet** multi-site |
| Stagnation ≥ 5 époques | Réduire LR × 0.5 ou escalader vers U-Net |

**Commande U-Net de secours (si escalade déclenchée) :**
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
| `panel_oversample` | 4 | OK — 8.0 panneaux/batch sur 16. Augmenter à **6–8** si IoU stagne |
| `tile_size` | 512 px | Adapté aux deux sites. Pas de changement nécessaire |
| `batch_size` | 16 (Patisen seul) | Réduire à **8** pour le multi-site (pression mémoire Malicounda) |
| `lr` | 0.0001 | Correct pour Fast SCNN. Ajouter `ReduceLROnPlateau(patience=5, factor=0.5)` si plateau |
| `stride` | 256 (overlap 50%) | Correct. Augmenter à 384 (overlap 25%) si temps/époque trop long en multi-site |

### D. Action immédiate requise — Corriger le chemin et relancer

**Cause racine identifiée :** `train.py` introuvable dans `/home/solar/`.
Lancer depuis le répertoire racine du projet :

```bash
# 1. Vérifier qu'aucun processus n'est déjà actif
ps aux | grep train.py

# 2. Se placer dans le répertoire du projet (adapter selon installation)
cd /chemin/vers/digitalize-panels-solar

# 3. Vérifier que train.py est présent
ls train.py

# 4. Relancer avec séparation stdout/stderr
nohup python3 train.py \
  --ortho Data/Orthomosaic_Patisen.tif \
  --shp Data/Panneaux_Patisen.shp \
  --tile_size 512 --stride 256 --batch_size 16 --epochs 50 \
  --panel_oversample 4 \
  --output_dir trained_models/patisen_gpu \
  > trained_models/patisen_gpu/train_log.txt \
  2> trained_models/patisen_gpu/train_log.txt.err &

# 5. Suivre la progression
tail -f trained_models/patisen_gpu/train_log.txt
```

## Décision
**Entraînement démarré (2026-05-19) — crash probable confirmé au 2026-05-20.**

- `training_log.csv` : vide après >24 h → anormal
- `train_log.txt.err` : `No such file or directory` pour `train.py` depuis `/home/solar/`
- **Action requise** : relancer `train.py` depuis le répertoire correct (voir section D ci-dessus)
- **Aucune décision d'escalade possible** sans données de validation
- **Seuil de décision** : époque 10 pour statuer sur Fast SCNN vs U-Net ResNet50
