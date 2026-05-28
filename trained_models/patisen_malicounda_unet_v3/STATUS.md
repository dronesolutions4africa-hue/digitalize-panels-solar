# Surveillance U-Net v3 — 2026-05-28

## État : DÉMARRAGE — EN ATTENTE (aucune époque complète)
- Époque : 0/100
- Meilleure val IoU panneaux : N/A
- Meilleure val loss : N/A
- Tendance (5 dernières) : N/A (aucune donnée)
- ETA estimée : inconnue — époque 1 non encore complétée
- vs v2 baseline : en attente de données (v2 : 0.7140 @ époque 61/67)

## Historique (15 dernières époques)
| Époque | val_loss | val_panel_iou | LR |
|--------|----------|---------------|----|
| —      | —        | —             | —  |

*Aucune époque complète enregistrée : `training_log.csv` et `train_log.txt` absents du répertoire `trained_models/patisen_malicounda_unet_v3/`.*

## Analyse

### État du run v3 (contrôle du 2026-05-28)
- Deuxième vérification consécutive (après 2026-05-27) — **aucun nouveau fichier** dans le répertoire v3.
- Ni `training_log.csv` ni `train_log.txt` ne sont présents → l'entraînement **n'a pas démarré** sur la machine locale, ou les fichiers ne sont pas encore committés/poussés vers le dépôt.
- Le répertoire v3 ne contient que ce fichier `STATUS.md`.

### Configuration v3 (rappel)
| Paramètre            | Valeur v3              | v2 (référence)         |
|----------------------|------------------------|------------------------|
| Modèle               | U-Net ResNet50         | U-Net ResNet50         |
| Sites                | Patisen + Malicounda   | Patisen + Malicounda   |
| max_tiles_per_site   | **0 (illimité)**       | 15 000 (plateau)       |
| Époques              | 100                    | 67 (early stop)        |
| panel_weight         | 20                     | 10                     |
| panel_oversample     | 8                      | 4                      |
| batch_size           | 4                      | 4                      |
| VRAM cap             | 85 % = 13 926 MB       | 85 %                   |

### Référence v2 (terminé)
- Meilleure val_panel_iou : **0.7140** à l'époque 61 (lr=2.5e-6)
- Run arrêté à l'époque 67 — plateau attribuable à `max_tiles_per_site=15000`
- Dernières 5 époques v2 (63–67) : val_panel_iou oscillant entre 0.7118 et 0.7130 → **PLATEAU**
- v3 vise **≥ 0.85** grâce aux tuiles illimitées, panel_weight=20, panel_oversample=8

## Décision

**L'entraînement v3 n'a toujours pas démarré sur la machine locale (WSL2).**

### Actions requises (machine locale)
1. **Vérifier que le script de lancement v3 est configuré** :
   ```
   --max_tiles_per_site 0
   --panel_weight 20
   --panel_oversample 8
   --epochs 100
   --output_dir trained_models/patisen_malicounda_unet_v3/
   ```
2. **Vérifier GPU disponible** : `nvidia-smi` → VRAM libre ≥ 2 000 MB pour batch_size=4
3. **Vérifier la disponibilité des données** :
   - Orthomosaïque Malicounda présente ?
   - Masque annotations Malicounda présent ?
4. **S'assurer que l'autopush est actif** : le script doit committer `training_log.csv` et `train_log.txt` après chaque époque (ou par blocs).

### Si blocage au démarrage
- **OOM (Out of Memory)** : réduire `batch_size` de 4 à 2
- **Tuiles illimitées trop lentes** : utiliser `--max_tiles_per_site 50000` en première approche pour valider la pipeline, puis relever à 0
- **Données manquantes** : vérifier `data/malicounda/` et les chemins dans le script de config

### Prochaine surveillance
Relancer ce monitoring dès que l'entraînement est démarré et qu'au moins 1 époque est complétée.
Le CSV devrait apparaître dans `trained_models/patisen_malicounda_unet_v3/training_log.csv` après la première époque.
