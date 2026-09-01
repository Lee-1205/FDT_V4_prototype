$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$outputDir = Join-Path $root "artifacts\fdt_v4_capability_trajectory_20260829"
if (Test-Path -LiteralPath $outputDir) {
    throw "Trajectory audit output already exists: $outputDir"
}
New-Item -ItemType Directory -Path $outputDir | Out-Null

$evaluator = Join-Path $root "scripts\evaluate_fdt_v4.py"
$tokenizer = Join-Path $root "tokenizers\fdt_v3_bpe_24k"
$dataset = Join-Path $root "artifacts\fdt_v4_200m_audit_20260829\fixed_generation_inputs.jsonl"
$repetition = Join-Path $root "artifacts\fdt_v4_200m_audit_20260829\fixed_repetition_inputs_100.jsonl"
$comparator = Join-Path $root "runs\fdt_v3_capability_completion_v20_balanced_scale_t1_20260816_r2\latest.pt"

$points = @(
    @{
        Label = "000000000"
        Checkpoint = Join-Path $root "runs\fdt_v4_audit_warmstart_20260823_0af5577\latest.pt"
    },
    @{
        Label = "009831728"
        Checkpoint = Join-Path $root "runs\fdt_v4_main_424m_curriculum_speed_20260827_r7_syncfix\latest.pt"
    },
    @{
        Label = "036370475"
        Checkpoint = Join-Path $root "runs\fdt_v4_main_424m_curriculum_speed_20260827_r9_batch4\latest.pt"
    },
    @{
        Label = "100005238"
        Checkpoint = Join-Path $root "runs\fdt_v4_main_424m_curriculum_bridge_20260827_r10\milestone_000100000000_tokens.pt"
    }
)

foreach ($point in $points) {
    $output = Join-Path $outputDir ("fdt_v4_{0}_official_fp32_eval.json" -f $point.Label)
    Write-Output ("START {0} {1}" -f $point.Label, (Get-Date -Format o))
    & python $evaluator `
        --checkpoint $point.Checkpoint `
        --output $output `
        --tokenizer $tokenizer `
        --dataset $dataset `
        --dataset-limit 52 `
        --comparator-checkpoint $comparator `
        --repetition-dataset $repetition `
        --bootstrap-samples 20000 `
        --device cpu
    if ($LASTEXITCODE -ne 0) {
        throw "Evaluation failed for $($point.Label) with exit code $LASTEXITCODE"
    }
    Write-Output ("COMPLETE {0} {1}" -f $point.Label, (Get-Date -Format o))
}
