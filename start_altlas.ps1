$ErrorActionPreference = 'Stop'

$logFile = "memory/launch_log.txt"

# Detect if inside *any* container (dev container or Docker)
$inContainer = $false
if ($env:ALT_IN_CONTAINER -eq '1') {
    $inContainer = $true
} elseif (Test-Path -Path '/proc/1/cgroup' -PathType Leaf -ErrorAction SilentlyContinue) {
    $content = Get-Content '/proc/1/cgroup' -ErrorAction SilentlyContinue
    if ($content -match 'docker|containerd|/docker/') { $inContainer = $true }
} elseif (Test-Path -Path '/.dockerenv' -ErrorAction SilentlyContinue) {
    $inContainer = $true
}

if ($inContainer) {
    Write-Host '[INFO] Detected container environment. Running AltLAS UI directly...'
    python altlas_ui.py @args
    exit $LASTEXITCODE
}

# Check if Docker is installed
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host '[ERROR] Docker is not installed or not in PATH. Please install Docker.'
    exit 127
}

# Log launch timestamp
if (-not (Test-Path 'memory')) { New-Item -ItemType Directory -Path 'memory' | Out-Null }
if ((Test-Path 'memory') -and (Get-Item 'memory').Attributes -notmatch 'ReadOnly') {
    $timestamp = (Get-Date -Format 'yyyy-MM-dd HH:mm:ss UTC')
    "$timestamp : Launching AltLAS with args: $args" | Add-Content -Encoding UTF8 $logFile
}

# Run Docker container
$pwdPath = (Get-Location).Path

docker run --rm -it `
    -v "$pwdPath:/app" `
    -w /app `
    -e ALT_IN_CONTAINER=1 `
    altlas-image:latest `
    python altlas_ui.py @args
