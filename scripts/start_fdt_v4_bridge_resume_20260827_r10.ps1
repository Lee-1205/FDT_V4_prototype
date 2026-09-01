$ErrorActionPreference = 'Stop'

$Root = 'C:\Users\User\Documents\Codex\2026-06-26\dk\outputs\FDT_RLM'
$Run = Join-Path $Root 'runs\fdt_v4_main_424m_curriculum_bridge_20260827_r10'
$Python = 'C:\Users\User\AppData\Local\Programs\Python\Python310\python.exe'
$Trainer = Join-Path $Root 'scripts\train_fdt_v4_curriculum_bridge.py'
$Config = Join-Path $Root 'configs\fdt_v4_main_426m_speed_r3_bridge.yaml'
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
    $Trainer = '6DF140EF2674CD48A3528941CB56AC893E849FE8AC322E1510C4D5B8CF5985E6'
    $Config = 'E80BA6D964F389D50ED16C400B07B2C645278477D50A39BC83047458FCBF5181'
    $Recovery = 'BC30F328CB886E698C27C1848681C4E496D521CAE656374F231A7703A81A8721'
}
foreach ($Entry in $Expected.GetEnumerator()) {
    if ((Get-FileHash -LiteralPath $Entry.Key -Algorithm SHA256).Hash -ne $Entry.Value) {
        throw "Pinned SHA-256 mismatch: $($Entry.Key)"
    }
}
if (Test-Path -LiteralPath (Join-Path $Run 'STOP_REQUESTED')) {
    Remove-Item -LiteralPath (Join-Path $Run 'STOP_REQUESTED')
}
$Arguments = @(
    $Trainer, '--config', $Config, '--output-dir', $Run,
    '--device', 'cuda', '--allow-gpu', '--resume', $Recovery
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
