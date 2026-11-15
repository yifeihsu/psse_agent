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

# A master that died (a sleep, a reboot of WSL) leaves its control socket
# behind; ssh then refuses to start a new master ("ControlSocket ... already
# exists, disabling multiplexing") and runs a plain connection instead, which
# is what the earlier "SSH returned success, but no reusable master" meant.
# Remove the stale socket first; the check above already proved it is dead.
# The command holds no dollar sign: wsl.exe hands it to a shell of its own before sh sees it, and that shell
# expanded awk's `$2` to nothing, so this read "controlpath <path>", a name no test finds, and the stale socket
# stayed (2026-11-15: two sign-ins, each ending in "no reusable master").
$staleSocket = & wsl.exe -- sh -c "ssh -G $HostAlias | sed -n 's/^controlpath //p'" 2>$null
$staleSocket = ("$staleSocket").Trim()
if ($staleSocket) {
    & wsl.exe -- sh -c "test -S '$staleSocket' && rm -f '$staleSocket' && echo removed" 2>$null | ForEach-Object {
        if ($_ -eq 'removed') { Write-Host "Removed the stale control socket $staleSocket." }
    }
}

Write-Host "Starting one persistent WSL SSH master for '$HostAlias'."
Write-Host 'Complete Microsoft device authentication once if prompted.'
& wsl.exe -- ssh -MNf $HostAlias
if ($LASTEXITCODE -ne 0) {
    throw "Could not start the WSL SSH master for '$HostAlias'."
}
if (-not (Test-SshMaster -Alias $HostAlias)) {
    # ssh signed in but could not take the control socket, so it is a plain connection in the background that
    # nothing can reuse.  End it: left alone, each failed start leaves one more signed-in session behind.
    & wsl.exe -- pkill -f "^ssh -MNf $HostAlias" 2>$null
    throw "SSH returned success, but no reusable master is available for '$HostAlias'. The control socket '$staleSocket' is in the way: remove it in WSL and run this again."
}

Write-Host "WSL SSH master is ready. Reuse it with: wsl -- ssh $HostAlias '<command>'"
