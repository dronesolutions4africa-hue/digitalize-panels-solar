# autopush_results.ps1 — pousse les résultats d'entraînement toutes les 30 min
# Lancer avec : powershell -WindowStyle Hidden -File autopush_results.ps1

$projectDir = "c:\Users\user\Downloads\solar-panels-detection-master\solar-panels-detection-master"
$intervalSec = 1800  # 30 minutes

while ($true) {
    Set-Location $projectDir

    # 1. Pousse les résultats locaux vers GitHub
    $changed = git status --porcelain trained_models/ 2>$null
    if ($changed) {
        git -c user.email="ndiouryoussouph@gmail.com" -c user.name="dronesolutions4africa-hue" `
            add trained_models/patisen_gpu/training_log.csv `
                 trained_models/patisen_gpu/train_log.txt `
                 trained_models/patisen_gpu/training_curves.png `
                 trained_models/patisen_gpu/best_model.weights.h5 `
                 trained_models/patisen_unet/ 2>$null
        $ts = Get-Date -Format "yyyy-MM-dd HH:mm"
        git -c user.email="ndiouryoussouph@gmail.com" -c user.name="dronesolutions4africa-hue" `
            commit -m "training results update $ts" 2>$null
        git push origin main 2>$null
        Add-Content "$projectDir\autopush.log" "$ts  pushed OK"
    }

    # 2. Récupère les décisions du cloud (STATUS.md, TRIGGER_UNET.flag)
    git pull origin main --quiet 2>$null

    # 3. Détecte le signal d'escalade U-Net posé par l'agent cloud
    $flagPath = "$projectDir\trained_models\TRIGGER_UNET.flag"
    if (Test-Path $flagPath) {
        $ts = Get-Date -Format "yyyy-MM-dd HH:mm"
        Add-Content "$projectDir\autopush.log" "$ts  TRIGGER U-Net détecté — lancement automatique"

        # Mise à jour du script WSL2 pour U-Net
        $unetScript = @"
#!/bin/bash
export PATH=/home/solar/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
SITE=/home/solar/.local/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=`$SITE/cublas/lib:`$SITE/cuda_cupti/lib:`$SITE/cuda_nvrtc/lib:`$SITE/cuda_runtime/lib:`$SITE/cudnn/lib:`$SITE/cufft/lib:`$SITE/curand/lib:`$SITE/cusolver/lib:`$SITE/cusparse/lib:`$SITE/nccl/lib:`$SITE/nvjitlink/lib
export TF_ENABLE_ONEDNN_OPTS=0
export PYTHONIOENCODING=utf-8
cd /mnt/c/Users/user/Downloads/solar-panels-detection-master/solar-panels-detection-master
python3 train.py --model unet_resnet50 --ortho Data/Orthomosaic_Patisen.tif --shp Data/Panneaux_Patisen.shp --tile_size 512 --stride 256 --batch_size 8 --epochs 50 --freeze_encoder_epochs 20 --output_dir trained_models/patisen_unet
"@
        $unetScript | wsl -d Ubuntu-22.04 -- tee /home/solar/run_unet_gpu.sh
        wsl -d Ubuntu-22.04 -- chmod +x /home/solar/run_unet_gpu.sh

        $logU = "$projectDir\trained_models\patisen_unet\train_log.txt"
        New-Item -ItemType Directory -Force "$projectDir\trained_models\patisen_unet" | Out-Null
        Start-Process -FilePath "wsl" `
            -ArgumentList "-d", "Ubuntu-22.04", "--", "/bin/bash", "/home/solar/run_unet_gpu.sh" `
            -RedirectStandardOutput $logU `
            -RedirectStandardError "$logU.err" `
            -WindowStyle Hidden

        Remove-Item $flagPath  # Consomme le flag
        Add-Content "$projectDir\autopush.log" "$ts  U-Net lancé"
    }

    Start-Sleep -Seconds $intervalSec
}
