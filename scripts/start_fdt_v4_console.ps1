param(
    [int]$Port = 7861
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Server = Join-Path $Root "apps\fdt_v4_console\server.py"
$RunState = Join-Path $Root "runs\fdt_v4_console"
New-Item -ItemType Directory -Force -Path $RunState | Out-Null
$Stdout = Join-Path $RunState "console.stdout.log"
$Stderr = Join-Path $RunState "console.stderr.log"
$Process = Start-Process -FilePath "python" -ArgumentList @($Server, "--port", $Port) -WorkingDirectory $Root -WindowStyle Hidden -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -PassThru
Set-Content -LiteralPath (Join-Path $RunState "console.pid") -Value $Process.Id -Encoding Ascii
Start-Sleep -Milliseconds 900
Start-Process "http://127.0.0.1:$Port/"
