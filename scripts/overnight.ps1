# overnight.ps1 - Runs all remaining gpt2_small experiments in safe order.
#
# USAGE:
#   .\scripts\overnight.ps1                  # all 3 domains, skip full_finetune
#   .\scripts\overnight.ps1 -Domain news     # one domain only
#   .\scripts\overnight.ps1 -IncludeFullFT   # include full_finetune (7h/domain)
#
# SAFE TO INTERRUPT AND RESUME: already-complete runs are skipped automatically.
# Run this script again after waking up and it picks up where it left off.

param(
    [string]$Domain = "all",
    [switch]$IncludeFullFT,
    [string]$Model = "gpt2_small"
)

$ErrorActionPreference = "Stop"

# -- Methods (cheapest first) -------------------------------------------------

$METHODS = @(
    "frozen",
    "pure_paft",
    "svf",
    "bitfit",
    "lora_r8",
    "polar",
    "safe_pure_paft",
    "safe_hybrid_paft",
    "hybrid_paft",
    "lora_r64"
)

if ($IncludeFullFT) {
    $METHODS = $METHODS + @("full_finetune")
    Write-Host "[WARNING] full_finetune included - approx 7h per domain" -ForegroundColor Yellow
}

# -- Domains ------------------------------------------------------------------

if ($Domain -eq "all") {
    $DOMAINS = @("news", "biomedical", "code")
} else {
    $DOMAINS = @($Domain)
}

# -- Helpers ------------------------------------------------------------------

function Is-Complete {
    param($model, $domain, $method)
    $sentinel = "results\checkpoints\$model\$domain\$method\final\training_complete"
    return Test-Path $sentinel
}

function Run-Experiment {
    param($model, $domain, $method)

    Write-Host ""
    Write-Host ("=" * 65) -ForegroundColor Cyan
    Write-Host "  $model / $domain / $method" -ForegroundColor Cyan
    Write-Host ("=" * 65) -ForegroundColor Cyan
    Write-Host "  Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

    $t = Measure-Command {
        python scripts\run_experiment.py --model $model --domain $domain --method $method
    }

    $mins = [math]::Round($t.TotalMinutes, 1)
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  Completed in ${mins} min" -ForegroundColor Green
        return $true
    } else {
        Write-Host "  FAILED after ${mins} min (exit $LASTEXITCODE)" -ForegroundColor Red
        return $false
    }
}

# -- Main ---------------------------------------------------------------------

$total   = $DOMAINS.Count * $METHODS.Count
$counter = 0
$skipped = 0
$failed  = @()
$tStart  = Get-Date

Write-Host ""
Write-Host "PAFT Overnight Sweep" -ForegroundColor White
Write-Host "Model:   $Model"
Write-Host "Domains: $($DOMAINS -join ', ')"
Write-Host "Methods: $($METHODS.Count) per domain"
Write-Host "Total:   $total runs"
Write-Host "Started: $tStart"
Write-Host ""

foreach ($domain in $DOMAINS) {
    Write-Host ""
    Write-Host ">>> Domain: $domain <<<" -ForegroundColor Yellow

    foreach ($method in $METHODS) {
        $counter++
        $pct     = [math]::Round(100 * $counter / $total)
        $elapsed = (Get-Date) - $tStart
        $elapsedH = [math]::Round($elapsed.TotalHours, 1)

        if (Is-Complete -model $Model -domain $domain -method $method) {
            Write-Host "  [$counter/$total] SKIP (complete): $method" -ForegroundColor DarkGray
            $skipped++
            continue
        }

        Write-Host "  [$counter/$total ${pct}%] $method  (elapsed: ${elapsedH}h)"

        $ok = Run-Experiment -model $Model -domain $domain -method $method
        if (-not $ok) {
            $failed = $failed + @("$Model/$domain/$method")
            Write-Host "  Continuing after failure..." -ForegroundColor Yellow
        }

        # Brief cooldown between runs so GPU temperature drops before next job
        if (-not (Is-Complete -model $Model -domain $domain -method $method)) {
            Write-Host "  Cooling down 3 min..." -ForegroundColor DarkGray
            Start-Sleep -Seconds 180
        }
    }

    Write-Host ""
    Write-Host "  Validating $domain checkpoints..." -ForegroundColor Cyan
    python scripts\validate_saves.py --model $Model --domain $domain
}

# -- Summary ------------------------------------------------------------------

$totalElapsed = (Get-Date) - $tStart
$totalH = [math]::Round($totalElapsed.TotalHours, 1)

Write-Host ""
Write-Host ("=" * 65) -ForegroundColor White
Write-Host "SWEEP COMPLETE" -ForegroundColor White
Write-Host "  Total time:  ${totalH} hours"
Write-Host "  Skipped:     $skipped (already complete)"
Write-Host "  Failed:      $($failed.Count)"

if ($failed.Count -gt 0) {
    Write-Host "  Failed runs:" -ForegroundColor Red
    foreach ($f in $failed) {
        Write-Host "    $f" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Next: python scripts\validate_saves.py"