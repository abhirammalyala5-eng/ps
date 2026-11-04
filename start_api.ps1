$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonCandidates = @(
    (Join-Path $ProjectRoot ".venv_run\Scripts\python.exe"),
    (Join-Path $ProjectRoot ".venv\Scripts\python.exe"),
    (Join-Path $ProjectRoot "vision\.venv\Scripts\python.exe")
)
$Python = $PythonCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
$Weights = Join-Path $ProjectRoot "vision\models\FoodSeg103-DeepLabV3Plus-MobileNet\best_model.pth"
$Labels = Join-Path $ProjectRoot "vision\models\FoodSeg103-DeepLabV3Plus-MobileNet\labels_config.json"

if (-not $Python) {
    Write-Error "No API Python environment was found. Set one up using the Windows instructions in vision\README.md, then run this script again."
}
if (-not (Test-Path -LiteralPath $Weights) -or -not (Test-Path -LiteralPath $Labels)) {
    Write-Error "FoodSeg103 model files are missing from vision\models\FoodSeg103-DeepLabV3Plus-MobileNet. Re-download or restore the model files before starting the API."
}

Set-Location $ProjectRoot
Write-Host "Starting Food Vision API at http://127.0.0.1:8000/docs"
Write-Host "Ollama must be running with qwen3-vl:2b for POST /vision requests."
& $Python -m uvicorn vision.app:app --host 127.0.0.1 --port 8000
exit $LASTEXITCODE
