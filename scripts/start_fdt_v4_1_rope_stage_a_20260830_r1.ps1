$ErrorActionPreference = 'Stop'

$Root = 'C:\Users\User\Documents\Codex\2026-06-26\dk\outputs\FDT_RLM'
$Run = Join-Path $Root 'runs\fdt_v4_1_424m_rope_stage_a_20260830_r1'
$Python = 'C:\Users\User\AppData\Local\Programs\Python\Python310\python.exe'
$Trainer = Join-Path $Root 'scripts\train_fdt_v4_curriculum_bridge.py'
$ModelSource = Join-Path $Root 'src\fdt_rlm\models\fdt_v4.py'
$CausalSource = Join-Path $Root 'src\fdt_rlm\models\causal_lm.py'
$ConfigSource = Join-Path $Root 'src\fdt_rlm\config.py'
$Config = Join-Path $Root 'configs\fdt_v4_1_424m_rope_output_blend_stage_a_20m.yaml'
$Parent = Join-Path $Root 'runs\fdt_v3_capability_completion_v20_balanced_scale_t1_20260816_r2\latest.pt'
$BehaviorAudit = Join-Path $Root 'artifacts\fdt_v4_1_repair_20260829\stage_a_alpha_zero_behavior_audit.json'
$ControlAblation = Join-Path $Root 'artifacts\fdt_v4_1_repair_20260829\transition_r2_step1250_control_ablation.json'
$PathProbe = Join-Path $Root 'artifacts\fdt_v4_1_repair_20260829\output_blend_path_static_probe.json'
$FocusedTests = Join-Path $Root 'artifacts\fdt_v4_1_repair_20260829\stage_a_focused_tests.xml'

foreach ($Path in @($Root, $Run, $Python, $Trainer, $ModelSource, $CausalSource, $ConfigSource, $Config, $Parent, $BehaviorAudit, $ControlAblation, $PathProbe, $FocusedTests)) {
    if (-not ([System.IO.Path]::GetFullPath($Path).StartsWith('C:\', [System.StringComparison]::OrdinalIgnoreCase))) {
        throw "C-only contract violated: $Path"
    }
}
$Expected = @{
    $Trainer = '0A533F4E4AA106992F54BD0DC0F8C0836144B965F1161FE0CEA1A3ECEF5E13C8'
    $ModelSource = 'B7B0C241D1E83D660F1FED57F08734E2B718ADD0F3CD4088C869BEC425CC4973'
    $CausalSource = 'E2A3F879EECB286FE1A49DFC98E26D909AB91F6C8E13878193F26CCE79923CD3'
    $ConfigSource = '043B262F4B9B9EF3EFE6EEAE905149F2515C5329703E9103322CD4A07A9A2811'
    $Config = '3F0B1373BA4A93CEA371F4CD1EEF7930B0BEDCC03CF3E55FB7768926266BFA80'
    $Parent = '7EE3F88D319928DD2D3F2542290F55FFCD036DCBB32A8AB22437C511E5890179'
    $BehaviorAudit = '0D1748B3CDE3A6703EEF8CEF2EDF8535252A612F020284CEAFD36B2D5ADE5259'
    $ControlAblation = '7AE38C9ADFEE4E8DF6B8B1A9E324F06E821E6EFB5935B10F1C64815A5FBE1A41'
    $PathProbe = '10CCFB30D94CDB7A3F680C424F697C20921050FDE4EAC3C7EF2DEFC3090A9BD1'
    $FocusedTests = '1F9D954A9649A1F0CBF5D8F4071B629E922B842642A926CF446C1C82A875473D'
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
    throw 'Alpha-zero behavior audit did not authorize stage A'
}
if ($Audit.config.sha256 -ne $Expected[$Config]) {
    throw 'Behavior audit config hash mismatch'
}
$Ablation = Get-Content -LiteralPath $ControlAblation -Raw | ConvertFrom-Json
if ($Ablation.decomposition.weight_update_delta_at_legacy_controls -ge 0.0) {
    throw 'The stopped run did not preserve or improve legacy-control validation'
}
$Probe = Get-Content -LiteralPath $PathProbe -Raw | ConvertFrom-Json
$Quarter = $Probe.candidate | Where-Object { $_.name -eq 'rope_025' }
if ($null -eq $Quarter -or $Quarter.validation_loss -gt $Probe.parent_validation_loss * 1.05) {
    throw 'Output-blend alpha 0.25 exceeds the predeclared 5% static regression gate'
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
    trainer_sha256 = $Expected[$Trainer]
    model_source_sha256 = $Expected[$ModelSource]
    parent = $Parent
    parent_sha256 = $Expected[$Parent]
    behavior_audit = $BehaviorAudit
    behavior_audit_sha256 = $Expected[$BehaviorAudit]
    control_ablation = $ControlAblation
    control_ablation_sha256 = $Expected[$ControlAblation]
    path_probe = $PathProbe
    path_probe_sha256 = $Expected[$PathProbe]
    focused_tests = $FocusedTests
    focused_tests_sha256 = $Expected[$FocusedTests]
    launched_at = (Get-Date -Format o)
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Run 'active_logs.json') -Encoding utf8
$Process.Id
