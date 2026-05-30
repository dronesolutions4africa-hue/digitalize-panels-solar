# Surveillance U-Net v3 — 2026-05-30

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

### État du run v3 (contrôle du 2026-05-30 — **14ème vérification consécutive**)
- **Aucun nouveau fichier** dans le répertoire v3 depuis le premier monitoring (2026-05-27 15:12 UTC).
- Ni `training_log.csv` ni `train_log.txt` ne sont présents → l'entraînement **n'a pas démarré** sur la machine locale, ou les fichiers ne sont pas committés/poussés.
- Historique git v3 : 14 commits de monitoring `epoch 0/100 val_iou=N/A` consécutifs sans progression :
  - `375fa69` — 2026-05-30 10:10 UTC (13ème)
  - `65c82b4` — 2026-05-30 05:12 UTC (12ème)
  - `f82bca8` — 2026-05-30 00:10 UTC (11ème)
  - `a1e1a18` — 2026-05-29 20:11 UTC (10ème)
  - `b92fd24` — 2026-05-29 10:12 UTC (9ème)
  - `86d443c` — 2026-05-29 05:10 UTC (8ème)
  - `53e5e44` — 2026-05-29 00:11 UTC (7ème)
  - `ddeec63` — 2026-05-28 20:10 UTC (6ème)
  - `29866cc` — 2026-05-28 15:12 UTC (5ème)
  - `56954c6` — 2026-05-28 10:11 UTC (4ème)
  - `ba42d9f` — 2026-05-28 05:10 UTC (3ème)
  - `b199bc6` — 2026-05-28 00:15 UTC (2ème)
  - `85c5c9f` — 2026-05-27 15:12 UTC (1er v3)
- **Durée totale sans données : ~72 heures (exactement 3 jours depuis la 1ère vérification 2026-05-27 15:12 UTC)**
- **Dernière vérification précédente (13ème) : 2026-05-30 10:10 UTC → ~5 heures sans changement**

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
- Dernières 5 époques v2 (63–67) : val_panel_iou entre 0.7118 et 0.7129 → **PLATEAU**
- v3 vise **≥ 0.85** grâce aux tuiles illimitées, panel_weight=20, panel_oversample=8

### Données v2 complètes (CSV — époques 15–67)
| Époque | val_loss | val_panel_iou | LR       |
|--------|----------|---------------|----------|
| 15     | 0.5208   | 0.5949        | 1.00e-05 |
| 20     | 0.4235   | 0.6596        | 1.00e-05 |
| 27     | 0.4124   | 0.6837        | 1.00e-05 |
| 32     | 0.4332   | 0.6900        | 1.00e-05 |
| 40     | 0.4160   | 0.6961        | 1.00e-05 |
| 43     | 0.4523   | 0.7004        | 5.00e-06 |
| 47     | 0.4456   | 0.7060        | 5.00e-06 |
| 52     | 0.4602   | 0.7075        | 5.00e-06 |
| 57     | 0.4616   | 0.7084        | 5.00e-06 |
| 58     | 0.4851   | 0.7064        | 2.50e-06 |
| **61** | **0.4618** | **0.7140** | **2.50e-06** |
| 63     | 0.4617   | 0.7126        | 2.50e-06 |
| 65     | 0.4793   | 0.7129        | 2.50e-06 |
| 66     | 0.4622   | 0.7122        | 2.50e-06 |
| 67     | 0.4667   | 0.7129        | 2.50e-06 |

## Décision

**L'entraînement v3 n'a toujours pas démarré (14ème contrôle consécutif — exactement 72 heures / 3 jours depuis la 1ère vérification).**

### Actions requises — ALERTE CRITIQUE MAXIMALE (3ème journée complète)

1. **Vérifier immédiatement l'état du processus Python** :
   ```bash
   ps aux | grep python
   nvidia-smi  # GPU occupé ?
   ```

2. **Vérifier que le script v3 est lancé avec les bons paramètres** :
   ```
   --max_tiles_per_site 0
   --panel_weight 20
   --panel_oversample 8
   --epochs 100
   --output_dir trained_models/patisen_malicounda_unet_v3/
   ```

3. **Vérifier les données Malicounda** :
   ```bash
   ls data/malicounda/
   # orthomosaïque + masque annotations présents ?
   ```

4. **Vérifier VRAM libre** :
   ```bash
   nvidia-smi --query-gpu=memory.free --format=csv
   # Doit afficher ≥ 2000 MiB
   ```

5. **S'assurer que l'autopush est actif** : le script doit committer `training_log.csv` + `train_log.txt` après chaque époque.

6. **Alternative d'urgence si `max_tiles_per_site 0` bloque au chargement** :
   - Démarrer avec `--max_tiles_per_site 50000` pour obtenir rapidement des métriques
   - Relancer illimité si la mémoire le permet

### Causes probables de blocage (14ème alerte — CRITIQUE ABSOLU)
| Cause                    | Diagnostic                              | Solution                              |
|--------------------------|----------------------------------------|---------------------------------------|
| OOM batch_size=4         | `nvidia-smi` crash ou OOM dans logs    | Réduire batch_size à 2                |
| Tuiles illimitées trop lentes | Chargement données > 30 min       | Utiliser `--max_tiles_per_site 50000` |
| Données Malicounda absentes | FileNotFoundError au démarrage      | Vérifier chemins `data/malicounda/`   |
| Process silencieusement crashé | Pas de PID Python actif         | Relancer le script manuellement       |
| Script non lancé         | Aucun `ps aux \| grep train`            | Lancer le script de training v3       |
| Autopush non configuré   | Fichiers générés localement mais non poussés | Vérifier le hook git post-epoch |

### Prochaine surveillance
Relancer ce monitoring dès qu'au moins 1 époque est complétée.
Le CSV devrait apparaître dans `trained_models/patisen_malicounda_unet_v3/training_log.csv` après la première époque.

> **⚠ ALERTE CRITIQUE ABSOLUE : 14 vérifications consécutives (72h = 3 jours complets) sans aucune progression — intervention manuelle sur la machine WSL2 URGENTE. L'entraînement v3 n'a probablement jamais démarré.**
