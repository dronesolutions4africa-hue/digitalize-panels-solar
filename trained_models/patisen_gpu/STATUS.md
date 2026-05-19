# Rapport de surveillance — 2026-05-19

## État : DÉMARRAGE
- Époque : 0/50 (Époque 1 en cours d'exécution)
- Meilleure val IoU panneaux : N/A (aucune époque complète)
- Meilleure val loss : N/A
- Tendance (5 dernières époques) : N/A
- ETA estimée : inconnue (première époque non terminée)

## Contexte de démarrage
- GPU détecté : 1 device `/physical_device:GPU:0`
- Données chargées : Orthomosaic_Patisen.tif (18 695×16 883 px) en 22.5 s
- Masque : 34 445 277 px panneaux / 315 627 685 total (**10.9%**)
- Tuiles : 3 687 brutes → 5 886 oversamplées (train), 921 (val)
- Panneaux/batch attendus : **8.0 / 16** (panel_oversample=4 actif)
- Modèle : Fast SCNN v2 — 1 901 450 params (7.25 MB)
- Hyperparamètres : lr=0.0001, panel_weight=15.0, batch=16, steps/epoch≈367
- Note : tentative précédente échouée (`/home/solar/train.py` introuvable) ; le run actuel a démarré correctement

## Historique (10 dernières époques)
| Époque | val_loss | val_panel_iou |
|--------|----------|---------------|
| —      | —        | —             |

*Aucune époque complète enregistrée dans training_log.csv. Données disponibles dès la fin de l'époque 1.*

## Recommandations

### A. Données Malicounda
Malicounda (86 280 panneaux annotés, résolution 1.7 cm/px, image 9.4 GB) représente un apport massif de données. Cependant, **l'intégration multi-site est recommandée après le run Patisen seul**, pour les raisons suivantes :

1. **Baseline d'abord** : le run actuel Patisen-seul fournira une référence claire (IoU, loss) avant d'introduire des variables supplémentaires.
2. **Domaine différent** : résolution 1.7 cm/px vs ~3 cm/px pour Patisen — le modèle doit d'abord apprendre les caractéristiques locales avant de gérer deux distributions d'entrée différentes.
3. **Coût mémoire** : avec 9.4 GB d'image source + tiles 512×512, le multi-site nécessite `--max_tiles_per_site 5000` pour ne pas saturer la RAM/VRAM.

**Commande recommandée pour l'entraînement multi-site (après run Patisen) :**
```bash
python3 train.py \
  --ortho Data/Orthomosaic_Patisen.tif Malicounda/ortho.tif \
  --shp Data/Panneaux_Patisen.shp Malicounda/Lim_panneaux.shp \
  --tile_size 512 --stride 256 --batch_size 8 --epochs 50 \
  --panel_oversample 4 --max_tiles_per_site 5000 \
  --output_dir trained_models/patisen_malicounda_gpu
```

> À résolution 1.7 cm/px, un tile 512×512 px couvre **8.7 m × 8.7 m** — adapté à la taille typique d'un panneau solaire. Aucun changement de `tile_size` requis pour Malicounda.

### B. Stratégie pour atteindre IoU >= 0.85

Aucune métrique disponible à ce stade. La décision sera prise après les 10 premières époques :

| Condition après époque 10 | Action recommandée |
|---|---|
| val IoU > 0.70 | Continuer Fast SCNN ; intégrer Malicounda pour le run multi-site |
| 0.50 ≤ val IoU ≤ 0.70 | Évaluer tendance + appliquer ReduceLROnPlateau ; décider à époque 20 |
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
| `panel_weight` | 15.0 | Maintenir pour l'instant. Augmenter à **20-25** si val IoU < 0.40 après époque 10 |
| `panel_oversample` | 4 | OK — 8.0 panneaux/batch sur 16 est satisfaisant. Augmenter à **6-8** si val IoU stagne malgré panel_weight élevé |
| `tile_size` | 512 px | Adapté à Patisen (~3 cm/px → 15.4m × 15.4m) et Malicounda (1.7 cm/px → 8.7m × 8.7m). Pas de changement nécessaire |
| `batch_size` | 16 (actuel) / 8 (prévu multi-site) | Réduire à 8 pour le multi-site (mémoire Malicounda) |
| `lr` | 0.0001 | Correct pour Fast SCNN. Appliquer ReduceLROnPlateau(patience=5, factor=0.5) si plateau détecté |

## Décision
**Entraînement démarré — en attente de la première époque complète.**

Prochain contrôle : dès que `training_log.csv` contient au moins 1 ligne (époque 1 terminée).
Décision d'escalade possible à partir de l'époque 10.
