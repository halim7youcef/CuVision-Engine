# ============================================================
#  CuVision-Engine | Segmentation
#  compile.ps1  — NVCC build script for Windows PowerShell
#
#  Requirements:
#    - CUDA Toolkit (nvcc in PATH)
#    - cuDNN  headers + import libs accessible via CUDA_PATH
#    - cuBLAS / cuRAND (ship with CUDA Toolkit)
#
#  Usage:
#    cd segmentation
#    .\compile.ps1
# ============================================================

Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  CuVision-Engine | Segmentation — Build" -ForegroundColor Cyan
Write-Host "  Network : Attention U-Net (ResNet encoder + ASPP)" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

# Resolve cuDNN paths from CUDA_PATH environment variable (standard installer location)
$cudaPath = $env:CUDA_PATH
if (-not $cudaPath) {
    Write-Host "[WARN] CUDA_PATH not set — assuming nvcc is in PATH and cuDNN is beside CUDA." -ForegroundColor Yellow
    $cudaPath = ""
}

$IncludeFlags = ""
$LibFlags     = ""
if ($cudaPath) {
    $IncludeFlags = "-I`"$cudaPath\include`""
    $LibFlags     = "-L`"$cudaPath\lib\x64`""
}

$OutputExe = "seg_unet.exe"

Write-Host "Compiling main.cu → $OutputExe ..." -ForegroundColor Gray

$cmd = "nvcc main.cu " +
       "$IncludeFlags $LibFlags " +
       "-O3 -use_fast_math -lineinfo " +
       "--generate-code arch=compute_75,code=sm_75 " +   # Turing (RTX 20xx)
       "--generate-code arch=compute_86,code=sm_86 " +   # Ampere (RTX 30xx)
       "--generate-code arch=compute_89,code=sm_89 " +   # Ada (RTX 40xx)
       "-lcudnn -lcublas -lcurand " +
       "-o $OutputExe"

Write-Host "  $cmd" -ForegroundColor DarkGray
Write-Host ""

Invoke-Expression $cmd

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "[SUCCESS] Build complete!" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Next steps:" -ForegroundColor White
    Write-Host "    1. Prepare dataset  : cd dataset && python prepare_dataset.py && cd .." -ForegroundColor Gray
    Write-Host "    2. Train the model  : .\$OutputExe" -ForegroundColor Gray
    Write-Host "    3. Checkpoints saved every 5 epochs automatically." -ForegroundColor Gray
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "[ERROR] Build failed (exit code $LASTEXITCODE)." -ForegroundColor Red
    Write-Host ""
    Write-Host "  Common fixes:" -ForegroundColor Yellow
    Write-Host "    - Ensure 'nvcc' is in your PATH  (CUDA Toolkit install)." -ForegroundColor Yellow
    Write-Host "    - Ensure cuDNN is installed and headers are under %CUDA_PATH%\include." -ForegroundColor Yellow
    Write-Host "    - Adjust arch flags (compute_75/86/89) to match your GPU generation." -ForegroundColor Yellow
    Write-Host "    - Run this script from the segmentation\ directory." -ForegroundColor Yellow
    Write-Host ""
    exit 1
}
