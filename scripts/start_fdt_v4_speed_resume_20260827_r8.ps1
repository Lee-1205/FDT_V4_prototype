$ErrorActionPreference = 'Stop'

$Root = 'C:\Users\User\Documents\Codex\2026-06-26\dk\outputs\FDT_RLM'
$Run = Join-Path $Root 'runs\fdt_v4_main_424m_curriculum_speed_20260827_r8_fast_backend'
$Python = 'C:\Users\User\AppData\Local\Programs\Python\Python310\python.exe'
$Trainer = Join-Path $Root 'scripts\train_fdt_v4_curriculum_speed.py'
$Config = Join-Path $Root 'configs\fdt_v4_main_426m_speed_r1.yaml'
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
