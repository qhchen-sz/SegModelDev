# analyze_profile.ps1 -- summarize EVDll_Segformer ev_trace.csv
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File analyze_profile.ps1 -Path D:\trace\ev_trace.csv
#
# Output:
#   1) per-stage n / mean / sd / p50 / p90 / p99 / max
#   2) per-frame table written to <input>_frames.csv
#   3) variance attribution: runmask / getmask / platform-gap share of frame span
#
# Notes:
#   - "trace.meta" rows are the diagnostic overhead itself.
#   - The first event of each frame has gapMs = GetBitMap + SetBitMap + platform delay.
#   - native* values come from the C++ SegGetLastTiming snapshot in the meta row note.

param(
    [Parameter(Mandatory = $false)]
    [string]$Path = "D:\trace\ev_trace.csv",
    [int]$Warmup = 5
)

if (-not (Test-Path $Path)) {
    Write-Error "trace file not found: $Path"
    exit 1
}

$rows = Import-Csv -Path $Path -Encoding UTF8
$events = @($rows | Where-Object {
    $_.event -ne 'meta' -and $_.event -ne 'trace.meta' -and [int]$_.frame -gt 0
})
if ($events.Count -eq 0) {
    Write-Error "no valid event rows"
    exit 1
}

$events = @($events | ForEach-Object {
    [pscustomobject]@{
        frame   = [int]$_.frame
        frameNo = [int]$_.frame
        seq     = [int]$_.seq
        event   = $_.event
        wall    = $_.wall
        dur     = [double]$_.durMs
        gap     = [double]$_.gapMs
    }
})

function Get-Stat($name, $values) {
    $v = @($values | Where-Object { $_ -ne $null -and "$_" -ne "" } | ForEach-Object { [double]$_ })
    $n = $v.Count
    if ($n -eq 0) { return $null }
    $s = @($v | Sort-Object)
    $mean = ($v | Measure-Object -Average).Average
    $var = 0.0
    foreach ($x in $v) { $var += ($x - $mean) * ($x - $mean) }
    $sd = if ($n -gt 1) { [math]::Sqrt($var / ($n - 1)) } else { 0.0 }
    $i50 = [math]::Min($n - 1, [math]::Max(0, [int][math]::Floor($n * 0.50)))
    $i90 = [math]::Min($n - 1, [math]::Max(0, [int][math]::Floor($n * 0.90)))
    $i99 = [math]::Min($n - 1, [math]::Max(0, [int][math]::Floor($n * 0.99)))
    [pscustomobject]@{
        Stage = $name
        N = $n
        Mean = [math]::Round($mean, 2)
        Sd = [math]::Round($sd, 2)
        P50 = [math]::Round($s[$i50], 2)
        P90 = [math]::Round($s[$i90], 2)
        P99 = [math]::Round($s[$i99], 2)
        Max = [math]::Round($s[$n - 1], 2)
    }
}

function Get-Var($values) {
    $v = @($values | ForEach-Object { [double]$_ })
    if ($v.Count -lt 2) { return 0.0 }
    $m = ($v | Measure-Object -Average).Average
    $s = 0.0
    foreach ($x in $v) { $s += ($x - $m) * ($x - $m) }
    return $s / ($v.Count - 1)
}

function Get-Cov($a, $b) {
    $n = [math]::Min($a.Count, $b.Count)
    if ($n -lt 2) { return 0.0 }
    $ma = 0.0
    $mb = 0.0
    for ($i = 0; $i -lt $n; $i++) { $ma += [double]$a[$i]; $mb += [double]$b[$i] }
    $ma /= $n
    $mb /= $n
    $s = 0.0
    for ($i = 0; $i -lt $n; $i++) {
        $s += ([double]$a[$i] - $ma) * ([double]$b[$i] - $mb)
    }
    return $s / ($n - 1)
}

$frameRows = New-Object System.Collections.Generic.List[object]
$groups = $events | Group-Object frameNo | Sort-Object { [int]$_.Name }
$prevGc0 = -1
$prevGc1 = -1
$prevGc2 = -1

foreach ($g in $groups) {
    if ([int]$g.Name -le $Warmup) { continue }
    $items = @($g.Group | Sort-Object { [int]$_.seq })

    $metaRow = @($rows | Where-Object {
        [int]$_.frame -eq [int]$g.Name -and $_.event -eq 'meta'
    }) | Select-Object -First 1

    $kv = @{}
    if ($metaRow -and $metaRow.note) {
        foreach ($part in ($metaRow.note -split ';')) {
            $pair = $part -split '=', 2
            if ($pair.Count -eq 2) { $kv[$pair[0]] = $pair[1] }
        }
    }

    $runMs = ($items | Where-Object { $_.event -like 'runmask.*' -or $_.event -like 'run.*' } |
              Measure-Object dur -Sum).Sum
    $getMs = ($items | Where-Object { $_.event -like 'getmask.*' -or $_.event -like 'getoverlay.*' } |
              Measure-Object dur -Sum).Sum
    $traceMs = ($items | Where-Object { $_.event -eq 'trace.meta' } |
                Measure-Object dur -Sum).Sum
    $spanMs = ($items | Measure-Object dur -Sum).Sum + ($items | Measure-Object gap -Sum).Sum
    $platformGap = [double]$items[0].gap

    $g0 = [int]$metaRow.gc0
    $g1 = [int]$metaRow.gc1
    $g2 = [int]$metaRow.gc2
    $d0 = if ($prevGc0 -ge 0 -and $g0 -ge $prevGc0) { $g0 - $prevGc0 } else { 0 }
    $d1 = if ($prevGc1 -ge 0 -and $g1 -ge $prevGc1) { $g1 - $prevGc1 } else { 0 }
    $d2 = if ($prevGc2 -ge 0 -and $g2 -ge $prevGc2) { $g2 - $prevGc2 } else { 0 }
    $prevGc0 = $g0
    $prevGc1 = $g1
    $prevGc2 = $g2

    $frameRows.Add([pscustomobject]@{
        frame         = [int]$g.Name
        wall          = $items[0].wall
        spanMs        = [math]::Round($spanMs, 3)
        runMaskMs     = [math]::Round($(if ($runMs) { $runMs } else { 0 }), 3)
        getMaskMs     = [math]::Round($(if ($getMs) { $getMs } else { 0 }), 3)
        traceMs       = [math]::Round($(if ($traceMs) { $traceMs } else { 0 }), 3)
        platformGapMs = [math]::Round($platformGap, 3)
        nativeTotal   = [double]$kv['nativeTotal']
        nativeClear   = [double]$kv['nativeClear']
        nativeWait    = [double]$kv['nativeWait']
        nativeAcc     = [double]$kv['nativeAcc']
        nativeArgmax  = [double]$kv['nativeArgmax']
        gpuInferSum   = [double]$kv['gpuInferSum']
        gpuWallSum    = [double]$kv['gpuWallSum']
        syncSum       = [double]$kv['syncSum']
        prepSum       = [double]$kv['prepSum']
        upSum         = [double]$kv['upSum']
        wakeMax       = [double]$kv['wakeMax']
        tailMax       = [double]$kv['tailMax']
        gc0           = $g0
        gc1           = $g1
        gc2           = $g2
        dgc0          = $d0
        dgc1          = $d1
        dgc2          = $d2
        srcFmt        = $metaRow.srcFmt
    })
}

if ($frameRows.Count -eq 0) {
    Write-Error "no data after skipping $Warmup warmup frames"
    exit 1
}

$traceDir = [System.IO.Path]::GetDirectoryName($Path)
$traceStem = [System.IO.Path]::GetFileNameWithoutExtension($Path)
$outFrames = if ($traceDir) { Join-Path $traceDir ($traceStem + "_frames.csv") } `
             else { $traceStem + "_frames.csv" }
$frameRows | Export-Csv -Path $outFrames -NoTypeInformation -Encoding UTF8

Write-Host ""
Write-Host "================ Stage statistics ================"
$stats = @()
$stats += Get-Stat 'runmask per frame' $frameRows.runMaskMs
$stats += Get-Stat 'getmask per frame' $frameRows.getMaskMs
$stats += Get-Stat 'platform gap'      $frameRows.platformGapMs
$stats += Get-Stat 'frame span'        $frameRows.spanMs
$stats += Get-Stat 'native total'      $frameRows.nativeTotal
$stats += Get-Stat 'native clear'      $frameRows.nativeClear
$stats += Get-Stat 'native wait'       $frameRows.nativeWait
$stats += Get-Stat 'native acc'        $frameRows.nativeAcc
$stats += Get-Stat 'native argmax'     $frameRows.nativeArgmax
$stats += Get-Stat 'gpu infer sum'     $frameRows.gpuInferSum
$stats += Get-Stat 'gpu wall sum'      $frameRows.gpuWallSum
$stats += Get-Stat 'sync sum'          $frameRows.syncSum
$stats += Get-Stat 'prep sum'          $frameRows.prepSum
$stats += Get-Stat 'up sum'            $frameRows.upSum
$stats += Get-Stat 'wake max'          $frameRows.wakeMax
$stats += Get-Stat 'tail max'          $frameRows.tailMax
$stats | Where-Object { $_ -ne $null } | Format-Table -AutoSize

Write-Host "================ Variance attribution ================"
$vt = Get-Var $frameRows.spanMs
if ($vt -gt 0) {
    $cRun = Get-Cov $frameRows.runMaskMs $frameRows.spanMs
    $cGet = Get-Cov $frameRows.getMaskMs $frameRows.spanMs
    $cGap = Get-Cov $frameRows.platformGapMs $frameRows.spanMs
    Write-Host ("runmask      : {0,6:F1} %" -f (100.0 * $cRun / $vt))
    Write-Host ("getmask      : {0,6:F1} %" -f (100.0 * $cGet / $vt))
    Write-Host ("platform gap : {0,6:F1} %" -f (100.0 * $cGap / $vt))
    Write-Host ("sum          : {0,6:F1} %" -f (100.0 * ($cRun + $cGet + $cGap) / $vt))
    Write-Host ""
}

$sortedSpan = @($frameRows.spanMs | Sort-Object)
$n = $sortedSpan.Count
$i50 = [math]::Min($n - 1, [int]($n * 0.50))
$i90 = [math]::Min($n - 1, [int]($n * 0.90))
$i99 = [math]::Min($n - 1, [int]($n * 0.99))
Write-Host ("frame span: p50={0:F1} p90={1:F1} p99={2:F1} max={3:F1}" -f `
    $sortedSpan[$i50], $sortedSpan[$i90], $sortedSpan[$i99], $sortedSpan[$n - 1])
Write-Host ""
Write-Host ("per-frame table: " + $outFrames)
