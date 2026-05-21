param(
    [string]$ModelDir = $(if ($env:CMML_MODEL_DIR) { $env:CMML_MODEL_DIR } else { "D:\cmml_models" })
)

$modelRoot = Join-Path $ModelDir "scimilarity"
$target = Join-Path $modelRoot "annotation_model_v1.tar.gz"
$extractDir = $modelRoot
$url = "https://zenodo.org/api/records/8240464/files/annotation_model_v1.tar.gz/content"
$expected = 6788245756
$log = Join-Path $modelRoot "download.log"
$errLog = Join-Path $modelRoot "download.err.log"

New-Item -ItemType Directory -Path $modelRoot -Force | Out-Null

while ($true) {
    $length = if (Test-Path $target) { (Get-Item $target).Length } else { 0 }
    if ($length -ge $expected) {
        break
    }

    & curl.exe -L -C - --retry 20 --retry-delay 10 --output $target $url 1>> $log 2>> $errLog
    if ($LASTEXITCODE -ne 0) {
        Add-Content -Path $errLog -Value "curl exited with code $LASTEXITCODE at $(Get-Date -Format o)"
    }
    Start-Sleep -Seconds 10
}

& tar.exe -xzf $target -C $extractDir
