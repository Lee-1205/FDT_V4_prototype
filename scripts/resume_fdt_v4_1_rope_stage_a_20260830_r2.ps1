$ErrorActionPreference = 'Stop'

$Root = 'C:\Users\User\Documents\Codex\2026-06-26\dk\outputs\FDT_RLM'
$Run = Join-Path $Root 'runs\fdt_v4_1_424m_rope_stage_a_20260830_r2_pinned_validation_resume'
$Python = 'C:\Users\User\AppData\Local\Programs\Python\Python310\python.exe'
$Trainer = Join-Path $Root 'scripts\train_fdt_v4_curriculum_bridge.py'
$ModelSource = Join-Path $Root 'src\fdt_rlm\models\fdt_v4.py'
$CausalSource = Join-Path $Root 'src\fdt_rlm\models\causal_lm.py'
$ConfigSource = Join-Path $Root 'src\fdt_rlm\config.py'
$Config = Join-Path $Root 'configs\fdt_v4_1_424m_rope_output_blend_stage_a_20m.yaml'
$Recovery = Join-Path $Run 'latest_recovery.pt'
$Migration = Join-Path $Run 'validation_seed_migration.json'
$BehaviorAudit = Join-Path $Root 'artifacts\fdt_v4_1_repair_20260829\stage_a_alpha_zero_behavior_audit_pinned_seed.json'
$FixedRecheck = Join-Path $Root 'artifacts\fdt_v4_1_repair_20260829\stage_a_r1_step250_fixed_validation_recheck.json'
$PathProbe = Join-Path $Root 'artifacts\fdt_v4_1_repair_20260829\output_blend_path_static_probe_pinned_seed.json'
$FocusedTests = Join-Path $Root 'artifacts\fdt_v4_1_repair_20260829\pinned_seed_focused_tests.xml'

$Expected = @{
    $Trainer = 'BF9C62105B0CF59297EEF9C90956ABA082443DBD848AD5A0D6F22A327F6F895F'
    $ModelSource = 'B7B0C241D1E83D660F1FED57F08734E2B718ADD0F3CD4088C869BEC425CC4973'
    $CausalSource = 'E2A3F879EECB286FE1A49DFC98E26D909AB91F6C8E13878193F26CCE79923CD3'
    $ConfigSource = '043B262F4B9B9EF3EFE6EEAE905149F2515C5329703E9103322CD4A07A9A2811'
    $Config = '8A6CCDD995D24551473095FEC45EE761E71C8DF5D52368FDF22CC43B1956C657'
    $Recovery = 'D889445AA4F71F6BE22BE5A0C0C9D56CC7E37374C46AC37296A392A114DFD0D8'
    $BehaviorAudit = '0AE3FC691ABD6D9F4E72883C401088711950E7E3B14F96936118684C0E4E6C02'
    $FixedRecheck = 'F995C7C4A288DC5F3296501ACE6FFACF36A64914A1F3AA9810DA206E3475A0F1'
    $PathProbe = '242EEC851DF22EE970279B01EFADFDF383D1F7FE6F70873D7ADFEC8AFC7650DC'
    $FocusedTests = '2D98B915DBBAC86A43566183C0ABEFACBC9A1C8538601D4998EE44D0E4CC6351'
}

foreach ($Entry in $Expected.GetEnumerator()) {
    if (-not (Test-Path -LiteralPath $Entry.Key)) {
        throw "Pinned input is missing: $($Entry.Key)"
    }
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $Entry.Key).Hash -ne $Entry.Value) {
        throw "Pinned SHA-256 mismatch: $($Entry.Key)"
    }
}
$MigrationState = Get-Content -LiteralPath $Migration -Raw | ConvertFrom-Json
if ($MigrationState.status -ne 'PASS' -or $MigrationState.stage_status -ne 'PAUSED' -or
    $MigrationState.checkpoint_sha256 -ne $Expected[$Recovery] -or
    $MigrationState.pinned_validation_seed -ne 20267830 -or
    $MigrationState.temp_residue.Count -ne 0) {
    throw 'Pinned-validation checkpoint migration did not pass verification'
}
$Existing = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python' -and $_.CommandLine -match 'train_fdt_v4_curriculum_bridge.py'
}
if ($Existing) {
    throw 'Another FDT v4 bridge trainer is already running'
}
$FreeGiB = (Get-PSDrive -Name C).Free / 1GB
if ($FreeGiB -lt 20.0) {
    throw "C: free space is below the 20 GiB launch floor: $FreeGiB"
}

$Timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$Stdout = Join-Path $Run "stdout_resume_$Timestamp.log"
$Stderr = Join-Path $Run "stderr_resume_$Timestamp.log"
$env:CUDA_MODULE_LOADING = 'LAZY'
$Arguments = @(
    $Trainer, '--config', $Config, '--output-dir', $Run,
    '--resume', $Recovery, '--device', 'cuda', '--allow-gpu'
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
    checkpoint = $Recovery
    checkpoint_sha256 = $Expected[$Recovery]
    migration = $Migration
    behavior_audit_sha256 = $Expected[$BehaviorAudit]
    fixed_recheck_sha256 = $Expected[$FixedRecheck]
    path_probe_sha256 = $Expected[$PathProbe]
    focused_tests_sha256 = $Expected[$FocusedTests]
    launched_at = (Get-Date -Format o)
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Run 'active_logs.json') -Encoding utf8
$Process.Id
