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
                 trained_models/patisen_malicounda_unet_ft/STATUS.md 2>$null
        $ts = Get-Date -Format "yyyy-MM-dd HH:mm"
        git -c user.email="ndiouryoussouph@gmail.com" -c user.name="dronesolutions4africa-hue" `
            commit -m "ft training results update $ts" 2>$null
        git push origin main 2>$null
        Add-Content "$projectDir\autopush.log" "$ts  pushed OK"
    }

    # 2. Récupère les décisions du cloud (STATUS.md)
    git pull origin main --quiet 2>$null

    Start-Sleep -Seconds $intervalSec
}
