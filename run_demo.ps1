$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { $python = (Get-Command python -ErrorAction Stop).Source }
$env:PYTHONPATH = Join-Path $root "src"
$env:KNOWLEDGER_MODE = "offline"
$env:KNOWLEDGER_HOST = "127.0.0.1"
$env:KNOWLEDGER_PORT = "8000"
$env:KNOWLEDGER_WORKSPACE = Join-Path $root "knowledger_data"
Write-Host "KnowLedger Web: http://127.0.0.1:8000/"
Write-Host "按 Ctrl+C 停止服务。"
& $python -m uvicorn knowledger.api:app --host 127.0.0.1 --port 8000
