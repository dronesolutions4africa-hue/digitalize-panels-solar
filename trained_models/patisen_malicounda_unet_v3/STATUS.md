# Surveillance U-Net v3 — 2026-05-27

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

*Aucune époque complète enregistrée dans `training_log.csv` (fichier absent ou vide).*

## Analyse

### État du run v3
- Le répertoire `trained_models/patisen_malicounda_unet_v3/` **n'existait pas** dans le dépôt au moment de cette vérification (2026-05-27).
- Aucun fichier `training_log.csv` ni `train_log.txt` présent → l'entraînement **n'a pas encore démarré**, ou la première époque n'est pas terminée et aucun push n'a eu lieu.

### Configuration v3 (rappel)
| Paramètre            | Valeur v3          | v2 (référence)     |
|----------------------|--------------------|--------------------|
| Modèle               | U-Net ResNet50     | U-Net ResNet50     |
| Sites                | Patisen + Malicounda | Patisen + Malicounda |
| max_tiles_per_site   | **0 (illimité)**   | 15 000 (plateau)   |
| Époques              | 100                | 67 (early stop)    |
| panel_weight         | 20                 | 10                 |
| panel_oversample     | 8                  | 4                  |
| batch_size           | 4                  | 4                  |
| VRAM cap             | 85% (13 926 MB)    | 85%                |

### Comparaison v2 (terminé)
- v2 meilleure val_panel_iou : **0.7140** (époque 61, lr=2.5e-6)
- v2 arrêté à 67 époques — plateau probable dû à la limite `max_tiles=15000`
- v3 vise **val_panel_iou ≥ 0.85** grâce aux tuiles illimitées et au panel_weight=20

## Décision

**Attendre le démarrage de l'entraînement v3 sur la machine locale (WSL2).**

### Vérifications à faire sur la machine locale
1. Confirmer que le script de lancement v3 est bien configuré (`--max_tiles_per_site 0`, `--panel_weight 20`, `--epochs 100`)
2. Vérifier que le dossier de sortie pointe vers `trained_models/patisen_malicounda_unet_v3/`
3. Vérifier GPU disponible : `nvidia-smi` → VRAM libre ≥ 1 500 MB pour batch_size=4
4. S'assurer que le script autopush est actif pour envoyer les logs toutes les N époques

### Si le démarrage est bloqué
- Vérifier OOM : si batch_size=4 ne passe pas → réduire à batch_size=2
- Vérifier chemins des données Malicounda : orthomosaïque + masque présents ?
- Consulter le diagnostic v2 : `trained_models/patisen_malicounda_unet_v2/training_log.csv`

### Prochaine surveillance
Relancer ce script de monitoring dès que la machine locale a démarré l'entraînement. La première époque devrait fournir un `training_log.csv` avec une ligne de données.
