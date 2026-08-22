[CmdletBinding()]
param(
    [ValidatePattern('^[A-Za-z0-9._-]+$')]
    [string]$HostAlias = 'torch',

    [switch]$Stop
)

# The WSL ~/.ssh/config entry for this alias must define ControlMaster,
# ControlPath, and ControlPersist. For the current torch entry, ControlPersist
# is 12h and ControlPath is ~/.ssh/cm-torch.
$ErrorActionPreference = 'Stop'

function Test-SshMaster {
    param([Parameter(Mandatory)][string]$Alias)

    $priorErrorActionPreference = $ErrorActionPreference
    try {
        # Windows PowerShell wraps native stderr (including ssh's successful
        # "Master running" status) as an ErrorRecord when it is redirected.
        $ErrorActionPreference = 'SilentlyContinue'
        & wsl.exe -- ssh -O check $Alias 2>$null
        $status = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $priorErrorActionPreference
    }
    return $status -eq 0
}

if ($Stop) {
    if (Test-SshMaster -Alias $HostAlias) {
        & wsl.exe -- ssh -O exit $HostAlias
        if ($LASTEXITCODE -ne 0) {
            throw "Could not stop the WSL SSH master for '$HostAlias'."
        }
        Write-Host "Stopped the WSL SSH master for '$HostAlias'."
    }
    else {
        Write-Host "No WSL SSH master is running for '$HostAlias'."
    }
    exit 0
}

if (Test-SshMaster -Alias $HostAlias) {
    Write-Host "WSL SSH master already running for '$HostAlias'."
    exit 0
}

Write-Host "Starting one persistent WSL SSH master for '$HostAlias'."
Write-Host 'Complete Microsoft device authentication once if prompted.'
& wsl.exe -- ssh -MNf $HostAlias
if ($LASTEXITCODE -ne 0) {
    throw "Could not start the WSL SSH master for '$HostAlias'."
}
if (-not (Test-SshMaster -Alias $HostAlias)) {
    throw "SSH returned success, but no reusable master is available for '$HostAlias'."
}

Write-Host "WSL SSH master is ready. Reuse it with: wsl -- ssh $HostAlias '<command>'"
