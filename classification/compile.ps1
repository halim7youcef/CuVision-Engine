# Compilation script for Windows using nvcc
# Ensure CUDA and cuDNN are installed and the paths are in your environment variables.

Write-Host "Compiling Deep Neural Network..." -ForegroundColor Cyan

nvcc main.cu -O3 -use_fast_math -lineinfo -lcudnn -lcublas -o dnn_classifier.exe

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[SUCCESS] Compilation completed!" -ForegroundColor Green
    Write-Host "Run the training using: .\dnn_classifier.exe" -ForegroundColor Gray
}
else {
    Write-Host "`n[ERROR] Compilation failed." -ForegroundColor Red
    Write-Host "Please verify that 'nvcc' is in your PATH and cuDNN/cuBLAS libraries are accessible." -ForegroundColor Yellow
}
