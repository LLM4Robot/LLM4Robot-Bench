param(
    [string]$TaskPython = 'D:\conda\envs\isaac51\python.exe',
    [string]$Config,
    [string]$ModelConfig,
    [switch]$Gui,
    [switch]$Debugger,
    [string]$Check = 'gpt-pick-place',
    [string]$DebugHost = '127.0.0.1',
    [ValidateRange(1, 65535)][int]$DebugPort = 5678
)

$ErrorActionPreference = 'Stop'
# Normal checks share the cross-platform Python launcher. Keep the legacy debugger below.
if (-not $Debugger) {
    $taskBootstrap = if (Test-Path -LiteralPath $TaskPython -PathType Leaf) { $TaskPython } else { 'python' }
    $taskOldPython = $env:GPT_BENCHMARK_PYTHON
    if ($PSBoundParameters.ContainsKey('TaskPython')) {
        $taskBootstrap = $TaskPython
        $env:GPT_BENCHMARK_PYTHON = $TaskPython
    } elseif (-not $env:GPT_BENCHMARK_PYTHON -and -not $env:GPT_BENCHMARK_RUNTIME_CONFIG -and
              -not (Test-Path "$PSScriptRoot/../configs/runtime/local.toml") -and
              (Test-Path -LiteralPath $TaskPython -PathType Leaf)) {
        # Preserve the old default when no modern runtime selection exists.
        $env:GPT_BENCHMARK_PYTHON = $TaskPython
    }
    $taskSavedBootstrapEnvironment = @{}
    foreach ($taskName in @('PYTHONNOUSERSITE', 'OMNI_KIT_DISABLE_TELEMETRY', 'OMNI_KIT_ACCEPT_EULA')) {
        $taskSavedBootstrapEnvironment[$taskName] = [Environment]::GetEnvironmentVariable($taskName, 'Process')
    }
    try {
        $env:PYTHONNOUSERSITE = '1'
        $env:OMNI_KIT_DISABLE_TELEMETRY = '1'
        # Preserve the original Windows launcher's acknowledgement for this
        # already configured Isaac installation, including non-debug checks.
        $env:OMNI_KIT_ACCEPT_EULA = 'YES'
        $taskArguments = @("$PSScriptRoot/launch.py", 'check', $Check)
        if ($Config) { $taskArguments += @('--config', $Config) }
        if ($ModelConfig) { $taskArguments += @('--model-config', $ModelConfig) }
        if ($Gui) { $taskArguments += '--gui' }
        & $taskBootstrap @taskArguments
        $taskCode = $LASTEXITCODE
    } finally {
        $env:GPT_BENCHMARK_PYTHON = $taskOldPython
        foreach ($taskName in $taskSavedBootstrapEnvironment.Keys) {
            [Environment]::SetEnvironmentVariable($taskName, $taskSavedBootstrapEnvironment[$taskName], 'Process')
        }
    }
    exit $taskCode
}
. "$PSScriptRoot\research_storage.ps1"
$taskProject = Split-Path -Parent $PSScriptRoot
$taskEnvironment = Split-Path -Parent $TaskPython
if (-not (Test-Path $TaskPython)) { throw "Python not found: $TaskPython" }
$taskSavedEnvironment = @{}
foreach ($taskName in @('PATH', 'CONDA_PREFIX', 'PYTHONNOUSERSITE', 'PYTHONUTF8', 'PYTHONUNBUFFERED', 'OMNI_KIT_DISABLE_TELEMETRY', 'OMNI_KIT_ACCEPT_EULA')) {
    $taskSavedEnvironment[$taskName] = [Environment]::GetEnvironmentVariable($taskName, 'Process')
}
Push-Location $taskProject
try {
    $taskPreviousStorage = Set-ResearchStorageEnvironment
    $env:PATH = "$taskEnvironment;$taskEnvironment\Scripts;$taskEnvironment\Library\bin;$env:PATH"
    $env:CONDA_PREFIX = $taskEnvironment
    $env:PYTHONNOUSERSITE = '1'
    $env:PYTHONUTF8 = '1'
    $env:PYTHONUNBUFFERED = '1'
    $env:OMNI_KIT_DISABLE_TELEMETRY = '1'
    # Existing local Isaac installation already accepted its EULA during setup.
    $env:OMNI_KIT_ACCEPT_EULA = 'YES'
    # Names, defaults and implementations belong to the Python registry.
    $taskArguments = @('-m', 'robot_tasks', 'check', $Check)
    if ($Debugger) {
        $taskArguments = @('-Xfrozen_modules=off', '-m', 'robot_tasks.integrationtest.debug_check',
                           '--listen-host', $DebugHost, '--port', "$DebugPort", '--check', $Check)
    }
    if ($PSBoundParameters.ContainsKey('Config')) { $taskArguments += @('--config', $Config) }
    if ($PSBoundParameters.ContainsKey('ModelConfig')) { $taskArguments += @('--model-config', $ModelConfig) }
    if ($Gui) { $taskArguments += '--gui' }
    & $TaskPython @taskArguments
    $taskExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
    if ($null -ne $taskPreviousStorage) { Restore-ResearchStorageEnvironment $taskPreviousStorage }
    foreach ($taskName in $taskSavedEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($taskName, $taskSavedEnvironment[$taskName], 'Process')
    }
}
exit $taskExitCode
