param(
    [string]$RuntimeArchive,
    [string]$RuntimeUrl
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$setupArgs = @('-3.14', '-B', 'setup_repository.py')
if ($RuntimeArchive) { $setupArgs += @('--runtime-archive', $RuntimeArchive) }
if ($RuntimeUrl) { $setupArgs += @('--runtime-url', $RuntimeUrl) }
& py @setupArgs
if ($LASTEXITCODE -ne 0) { throw 'Setup failed. Inspect the message above and QUICKSTART.md.' }
