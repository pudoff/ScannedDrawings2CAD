param(
    [int]$Port = 8000,
    [int]$DbPort = 55445
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    throw 'Create the environment first: py -3.14 setup_repository.py --runtime-archive <archive.zip>. See QUICKSTART.md.'
}
& .\.venv\Scripts\python.exe run_service.py --port $Port --db-port $DbPort --open
if ($LASTEXITCODE -ne 0) { throw 'Service stopped with an error. Inspect the logs and QUICKSTART.md.' }
