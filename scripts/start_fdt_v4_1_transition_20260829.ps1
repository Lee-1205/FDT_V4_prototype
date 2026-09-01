$ErrorActionPreference = 'Stop'

$Root = 'C:\Users\User\Documents\Codex\2026-06-26\dk\outputs\FDT_RLM'
$Run = Join-Path $Root 'runs\fdt_v4_1_424m_transition_20260829_r1'
$Python = 'C:\Users\User\AppData\Local\Programs\Python\Python310\python.exe'
$Trainer = Join-Path $Root 'scripts\train_fdt_v4_curriculum_bridge.py'
$Config = Join-Path $Root 'configs\fdt_v4_1_424m_transition_20m.yaml'
$Parent = Join-Path $Root 'runs\fdt_v3_capability_completion_v20_balanced_scale_t1_20260816_r2\latest.pt'
$BehaviorAudit = Join-Path $Root 'artifacts\fdt_v4_1_repair_20260829\behavior_preservation_fp32_r3_final.json'

foreach ($Path in @($Root, $Run, $Python, $Trainer, $Config, $Parent, $BehaviorAudit)) {
    if (-not ([System.IO.Path]::GetFullPath($Path).StartsWith('C:\', [System.StringComparison]::OrdinalIgnoreCase))) {
        throw "C-only contract violated: $Path"
    }
}
$Expected = @{
    $Trainer = '5C4BD910ADFD9EE7CCD709613FDA7049787BE464C5F261B8E5D0755C607F5E72'
    $Config = 'BE10A2957653FA67FB383F3AAAED65EFFDE5FBC246AB6714B8090456834EB5D9'
    $Parent = '7EE3F88D319928DD2D3F2542290F55FFCD036DCBB32A8AB22437C511E5890179'
    $BehaviorAudit = '1857F5ED4B62E984B7254125168F0760C107DC2C6BC51C1E072E261C021E9448'
}
foreach ($Entry in $Expected.GetEnumerator()) {
    if (-not (Test-Path -LiteralPath $Entry.Key)) {
        throw "Pinned input is missing: $($Entry.Key)"
    }
    if ((Get-FileHash -LiteralPath $Entry.Key -Algorithm SHA256).Hash -ne $Entry.Value) {
        throw "Pinned SHA-256 mismatch: $($Entry.Key)"
    }
}
$Audit = Get-Content -LiteralPath $BehaviorAudit -Raw | ConvertFrom-Json
if ($Audit.status -ne 'PASS' -or $Audit.decision -ne 'ALLOW_BEHAVIOR_PRESERVING_TRANSITION_PILOT') {
    throw 'Behavior-preservation audit did not authorize the transition pilot'
}
if ($Audit.config.sha256 -ne $Expected[$Config]) {
    throw 'Behavior-preservation audit does not match the pinned config'
}
if (Test-Path -LiteralPath $Run) {
    if ((Get-ChildItem -LiteralPath $Run -Force | Measure-Object).Count -gt 0) {
        throw "Fresh output path is already non-empty: $Run"
    }
} else {
    New-Item -ItemType Directory -Path $Run | Out-Null
}
$FreeGiB = (Get-PSDrive -Name C).Free / 1GB
if ($FreeGiB -lt 20.0) {
    throw "C: free space is below the 20 GiB launch floor: $FreeGiB"
}
$Existing = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python' -and $_.CommandLine -match 'train_fdt_v4_curriculum_bridge.py'
}
if ($Existing) {
    throw 'Another FDT v4 bridge trainer is already running'
}

$Stdout = Join-Path $Run 'stdout.log'
$Stderr = Join-Path $Run 'stderr.log'
$env:CUDA_MODULE_LOADING = 'LAZY'
$Arguments = @(
    $Trainer, '--config', $Config, '--output-dir', $Run,
    '--device', 'cuda', '--allow-gpu'
)
$Process = Start-Process -FilePath $Python -ArgumentList $Arguments `
    -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr `
    -WindowStyle Hidden -PassThru
Set-Content -LiteralPath (Join-Path $Run 'train.pid') -Value $Process.Id -Encoding ascii
@{
    pid = $Process.Id
    stdout = $Stdout
    stderr = $Stderr
    training_log = (Join-Path $Run 'training_log.jsonl')
    config = $Config
    config_sha256 = $Expected[$Config]
    parent = $Parent
    parent_sha256 = $Expected[$Parent]
    behavior_audit = $BehaviorAudit
    behavior_audit_sha256 = $Expected[$BehaviorAudit]
    launched_at = (Get-Date -Format o)
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Run 'active_logs.json') -Encoding utf8
$Process.Id
