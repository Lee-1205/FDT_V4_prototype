$ErrorActionPreference = 'Stop'

$Root = 'C:\Users\User\Documents\Codex\2026-06-26\dk\outputs\FDT_RLM'
$Python = 'C:\Users\User\AppData\Local\Programs\Python\Python310\python.exe'
$Config = Join-Path $Root 'configs\fdt_v4_main_426m_speed_r1.yaml'
$Trainer = Join-Path $Root 'scripts\train_fdt_v4_curriculum_speed.py'
$Supervisor = Join-Path $Root 'scripts\luna_fdt_v4_speed_supervisor.py'
$Run = Join-Path $Root 'runs\fdt_v4_main_424m_curriculum_speed_20260827_r2'
$State = Join-Path $Root 'runs\fdt_v4_main_424m_speed_supervisor_20260827_r2'
$Recovery = Join-Path $Run 'latest_recovery.pt'

$Expected = @{
    $Config = '358DE85C696AC0D263791A7200D1D6E75FBBF70090365BFB963893D226772E14'
    $Trainer = '656DD0468D255AF75F3E5BC20F05E05FD12BE7CCBBAD6F5BF1E0A7285A8CB988'
    $Supervisor = '9293ADD9369FB253AC526E4D491728DF86F3B490FFB2F50E6E513F750F4557A2'
    $Recovery = '6A1419B02ACA51AE4DE4C683B2F8A6AAF3F8C4CB9B979811CB127FD49C9123FF'
}
foreach ($Path in $Expected.Keys) {
    $Actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash
    if ($Actual -ne $Expected[$Path]) {
        throw "Pinned SHA-256 mismatch: $Path"
    }
}
if (Test-Path -LiteralPath $State) {
    throw "Supervisor state path is not fresh: $State"
}
New-Item -ItemType Directory -Path $State | Out-Null
$env:CUDA_MODULE_LOADING = 'LAZY'
$Arguments = @(
    $Supervisor,
    '--config', $Config,
    '--run-dir', $Run,
    '--state-dir', $State,
    '--device', 'cuda',
    '--allow-gpu',
    '--resume-paused',
    '--poll-seconds', '10',
    '--restart-delay', '5',
    '--max-restarts', '0'
)
$Process = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $Root -WindowStyle Hidden -PassThru
$Launch = [ordered]@{
    status = 'LAUNCHED'
    supervisor_pid = $Process.Id
    run_dir = $Run
    state_dir = $State
    recovery_sha256 = $Expected[$Recovery]
    config_sha256 = $Expected[$Config]
    trainer_sha256 = $Expected[$Trainer]
    supervisor_sha256 = $Expected[$Supervisor]
    runtime = 'short512_batch2_no_checkpoint_long8k16k_checkpointed'
}
$Temporary = Join-Path $State 'launch.json.tmp'
$Final = Join-Path $State 'launch.json'
$Launch | ConvertTo-Json | Set-Content -LiteralPath $Temporary -Encoding UTF8
Move-Item -LiteralPath $Temporary -Destination $Final
$Launch | ConvertTo-Json -Compress
