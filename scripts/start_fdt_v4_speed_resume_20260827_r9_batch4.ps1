$ErrorActionPreference = 'Stop'

$Root = 'C:\Users\User\Documents\Codex\2026-06-26\dk\outputs\FDT_RLM'
$Run = Join-Path $Root 'runs\fdt_v4_main_424m_curriculum_speed_20260827_r9_batch4'
$Python = 'C:\Users\User\AppData\Local\Programs\Python\Python310\python.exe'
$Trainer = Join-Path $Root 'scripts\train_fdt_v4_curriculum_speed_observable.py'
$Config = Join-Path $Root 'configs\fdt_v4_main_426m_speed_r2_batch4.yaml'
$Recovery = Join-Path $Run 'latest_recovery.pt'
$Stdout = Join-Path $Run 'stdout.log'
$Stderr = Join-Path $Run 'stderr.log'

foreach ($Path in @($Root, $Run, $Python, $Trainer, $Config, $Recovery)) {
    if (-not ([System.IO.Path]::GetFullPath($Path).StartsWith('C:\', [System.StringComparison]::OrdinalIgnoreCase))) {
        throw "C-only contract violated: $Path"
    }
}
if (-not (Test-Path -LiteralPath $Recovery)) {
    throw "Verified recovery checkpoint is missing: $Recovery"
}
$Expected = @{
    $Trainer = '938EF8553A3CA9CE3CF54F3A1EBABA7D3E84152F0123FC3510AFA007D0BB8B9E'
    $Config = '69E8C585B062054D9B73506E68B99663CB8A2A5D8113C06589327C4ECB31DC71'
    $Recovery = '37F51CBC7CA409D1487EB9C73944323CCF43E2F06C92D9E3C947775C6E8745C2'
}
foreach ($Entry in $Expected.GetEnumerator()) {
    $Actual = (Get-FileHash -LiteralPath $Entry.Key -Algorithm SHA256).Hash
    if ($Actual -ne $Entry.Value) {
        throw "Pinned SHA-256 mismatch: $($Entry.Key)"
    }
}
if (Test-Path -LiteralPath (Join-Path $Run 'STOP_REQUESTED')) {
    Remove-Item -LiteralPath (Join-Path $Run 'STOP_REQUESTED')
}

$Arguments = @(
    $Trainer,
    '--config', $Config,
    '--output-dir', $Run,
    '--device', 'cuda',
    '--allow-gpu',
    '--resume', $Recovery
)
$Process = Start-Process -FilePath $Python -ArgumentList $Arguments -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -WindowStyle Hidden -PassThru
Set-Content -LiteralPath (Join-Path $Run 'train.pid') -Value $Process.Id -Encoding ascii
@{
    pid = $Process.Id
    stdout = $Stdout
    stderr = $Stderr
    training_log = (Join-Path $Run 'training_log.jsonl')
    recovery = $Recovery
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Run 'active_logs.json') -Encoding utf8
$Process.Id
