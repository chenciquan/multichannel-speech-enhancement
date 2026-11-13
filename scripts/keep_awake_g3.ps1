param([Parameter(Mandatory=$true)][string]$DeadlineUtc)
Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; public static class G3Power { [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags); }'
try {
    $g3Deadline=[DateTimeOffset]::Parse($DeadlineUtc)
    [void][G3Power]::SetThreadExecutionState([uint32]2147483649)
    while ([DateTimeOffset]::UtcNow -lt $g3Deadline) { Start-Sleep -Seconds 15 }
} finally { [void][G3Power]::SetThreadExecutionState([uint32]2147483648) }
