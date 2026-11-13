param(
    [Parameter(Mandatory=$true)][string]$DeadlineUtc,
    [Parameter(Mandatory=$true)][string]$StopFile,
    [Parameter(Mandatory=$true)][string]$StatusPath
)
Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; public static class McseJobPower { [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags); }'
$jobDeadline=[DateTimeOffset]::Parse($DeadlineUtc)
$jobRecord=[ordered]@{ pid=$PID; started_utc=[DateTimeOffset]::UtcNow.ToString('o'); deadline_utc=$jobDeadline.ToString('o'); status='starting'; power_plan_modified=$false }
try {
    $jobRequest=[McseJobPower]::SetThreadExecutionState([uint32]2147483649)
    if ($jobRequest -eq 0) { throw 'Could not request temporary system-awake state' }
    $jobRecord.status='requested'; $jobRecord | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding utf8
    while ([DateTimeOffset]::UtcNow -lt $jobDeadline -and -not (Test-Path -LiteralPath $StopFile)) { Start-Sleep -Seconds 5 }
} finally {
    [void][McseJobPower]::SetThreadExecutionState([uint32]2147483648)
    $jobRecord.status='released'; $jobRecord.released_utc=[DateTimeOffset]::UtcNow.ToString('o')
    $jobRecord | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding utf8
}
