# autopush_results.ps1 — pousse les résultats d'entraînement toutes les 30 min
# Lancer avec : powershell -WindowStyle Hidden -File autopush_results.ps1
# Pré-requis : git remote origin configuré avec PAT (fait une seule fois) :
#   git remote set-url origin https://<PAT>@github.com/dronesolutions4africa-hue/digitalize-panels-solar.git

$projectDir = "c:\Users\user\Downloads\solar-panels-detection-master\solar-panels-detection-master"
$intervalSec = 1800  # 30 minutes

while ($true) {
    Set-Location $projectDir

    # 1. Pousse les résultats locaux vers GitHub
    $changed = git status --porcelain trained_models/ 2>$null
    if ($changed) {
        git -c user.email="ndiouryoussouph@gmail.com" -c user.name="dronesolutions4africa-hue" `
            add trained_models/patisen_malicounda_unet_ft/training_log.csv `
                 trained_models/patisen_malicounda_unet_ft/training_curves.png `
                 trained_models/patisen_malicounda_unet_ft/STATUS.md `
                 trained_models/patisen_malicounda_unet_v4/training_log.csv `
                 trained_models/patisen_malicounda_unet_v4/training_curves.png `
                 trained_models/p2_attn_unet_3sites/training_log.csv `
                 trained_models/p2_attn_unet_3sites/training_curves.png 2>$null
        $ts = Get-Date -Format "yyyy-MM-dd HH:mm"
        git -c user.email="ndiouryoussouph@gmail.com" -c user.name="dronesolutions4africa-hue" `
            commit -m "p2 attn unet gpu training update $ts" 2>$null
        git push origin main 2>$null
        Add-Content "$projectDir\autopush.log" "$ts  pushed OK"
    }

    # 2. Récupère les décisions du cloud (STATUS.md, scripts mis à jour)
    git pull origin main --quiet 2>$null

    # 3. Redémarrage automatique si l'agent cloud a créé RESTART_NEEDED.flag
    $flagFile = "$projectDir\RESTART_NEEDED.flag"
    if (Test-Path $flagFile) {
        $ts = Get-Date -Format "yyyy-MM-dd HH:mm"
        Add-Content "$projectDir\autopush.log" "$ts  RESTART_NEEDED.flag détecté — redémarrage auto"

        # Tuer le training actuel dans WSL2
        wsl -d Ubuntu-22.04 -e bash -c "pkill -f train.py; sleep 3" 2>$null

        # Supprimer le flag AVANT de relancer
        Remove-Item $flagFile -Force
        git -c user.email="ndiouryoussouph@gmail.com" -c user.name="dronesolutions4africa-hue" `
            add RESTART_NEEDED.flag 2>$null
        git -c user.email="ndiouryoussouph@gmail.com" -c user.name="dronesolutions4africa-hue" `
            commit -m "auto-restart: remove RESTART_NEEDED.flag $ts" 2>$null
        git push origin main 2>$null

        # Relancer l'entraînement avec le script mis à jour par l'agent
        wsl -d Ubuntu-22.04 -e bash -c "setsid /home/solar/run_p2_gpu.sh &" 2>$null
        Add-Content "$projectDir\autopush.log" "$ts  Entraînement relancé avec nouveaux paramètres"
    }

    Start-Sleep -Seconds $intervalSec
}
