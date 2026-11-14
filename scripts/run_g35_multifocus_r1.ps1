$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $workspace
Add-Type -Namespace G35Round1 -Name Awake -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint flags);
'@
$null = [G35Round1.Awake]::SetThreadExecutionState([uint32]2147483649)
try {
    & wsl.exe --exec /mnt/c/Users/chenciquan/Desktop/multichannel_speech_enhancement/.venv/bin/python -B scripts/g35_multifocus_r1.py run
    if ($LASTEXITCODE -ne 0) { throw "Round-one process stopped with code $LASTEXITCODE; do not restart automatically" }
}
finally {
    $null = [G35Round1.Awake]::SetThreadExecutionState([uint32]2147483648)
}
