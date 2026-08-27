param(
    [Parameter(Mandatory = $true)]
    [string]$AssetsDirectory,

    [string]$OutputPath = "fdt_v4_step3000_recovery.pt",

    [Parameter(Mandatory = $true)]
    [string]$ExpectedSha256
)

$ErrorActionPreference = "Stop"
$parts = Get-ChildItem -LiteralPath $AssetsDirectory -File |
    Where-Object { $_.Name -match '^fdt_v4_step3000_recovery\.pt\.part\d{3}$' } |
    Sort-Object Name

if ($parts.Count -ne 3) {
    throw "Expected exactly three recovery parts, found $($parts.Count)."
}

$resolvedOutput = [System.IO.Path]::GetFullPath($OutputPath)
$output = [System.IO.File]::Open($resolvedOutput, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write)
try {
    foreach ($part in $parts) {
        $input = [System.IO.File]::OpenRead($part.FullName)
        try {
            $input.CopyTo($output)
        }
        finally {
            $input.Dispose()
        }
    }
}
finally {
    $output.Dispose()
}

$actual = (Get-FileHash -LiteralPath $resolvedOutput -Algorithm SHA256).Hash
if ($actual -ne $ExpectedSha256.ToUpperInvariant()) {
    throw "SHA-256 mismatch. Expected $ExpectedSha256, got $actual."
}

Write-Output "Verified recovery checkpoint: $resolvedOutput"

