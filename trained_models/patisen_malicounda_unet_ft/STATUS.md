# Surveillance fine-tuning — 2026-06-01

## État : DÉMARRAGE
- Époque : 0/50
- Meilleure val IoU : N/A (aucune époque complète)
- Baseline v2 : 0.714 (époque 61)
- Tendance (5 dernières) : N/A
- ETA : indéterminé — en attente du premier checkpoint

## Historique (10 dernières époques)
| Époque | val_loss | val_panel_iou | LR |
|--------|----------|---------------|----|
| —      | —        | —             | —  |

## Analyse
Le répertoire `patisen_malicounda_unet_ft` vient d'être créé mais aucun fichier `training_log.csv` n'est encore présent. L'entraînement n'a pas encore démarré ou l'époque 1 est en cours.

**Stratégie rappel :**
- Point de départ : poids U-Net v2 (val_panel_iou = 0.714)
- lr = 1e-6 (10× plus bas que v2 : 1e-5)
- panel_oversample = 4 (vs 8 en v2 pour réduire le surapprentissage)
- max_tiles_per_site = 20 000
- freeze_encoder_epochs = 0
- Durée prévue : 50 époques

**Historique des runs :**
| Run          | val_panel_iou | Remarque                         |
|--------------|---------------|----------------------------------|
| Fast SCNN    | 0.197         | Baseline léger                   |
| U-Net v1     | 0.519         | Premier U-Net                    |
| U-Net v2     | 0.714         | Meilleur modèle, époque 61       |
| U-Net v3     | 0.413         | Arrêté — surapprentissage sévère |
| U-Net ft     | en cours…     | Fine-tuning depuis v2            |

## Décision
**Attendre** — aucune action corrective n'est possible avant que le premier epoch soit terminé. Prochain contrôle dans ~30 min via autopush.

Si après 5 époques val_panel_iou < 0.700 (régression vs baseline v2), envisager :
1. Vérifier que les poids v2 ont bien été chargés (checkpoint path)
2. Augmenter légèrement lr à 2e-6 si la descente de loss est trop lente

Si après 15 époques val_panel_iou < 0.740, envisager :
1. Continuer avec lr = 5e-7 (réduction supplémentaire)
2. Ajouter dropout=0.3 dans le décodeur pour régularisation
