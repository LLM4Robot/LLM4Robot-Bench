param(
    [string]$TaskPython = 'D:\conda\envs\isaac51\python.exe'
)

$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\research_storage.ps1"
if (-not (Test-Path $TaskPython)) {
    throw "Python not found: $TaskPython"
}
$taskOldUtf8 = $env:PYTHONUTF8
$env:PYTHONUTF8 = '1'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $taskPreviousStorage = Set-ResearchStorageEnvironment
    & $TaskPython -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw 'Framework tests failed' }
    & $TaskPython -m robot_tasks validate-robot --profile configs/robots/r5.toml --output reports/r5_asset_audit_windows.json
    if ($LASTEXITCODE -ne 0) { throw 'R5 asset validation failed' }
    foreach ($taskConfig in @(
        'configs/tasks/l2_01_marker_uncap.toml',
        'configs/checks/gpt_single_arm_pick_place.toml'
    )) {
        & $TaskPython -m robot_tasks inspect --config $taskConfig
        if ($LASTEXITCODE -ne 0) { throw "Invalid configuration: $taskConfig" }
        & $TaskPython -m robot_tasks dry-run --config $taskConfig
        if ($LASTEXITCODE -ne 0) { throw "Flow check failed: $taskConfig" }
    }
}
finally {
    if ($null -ne $taskPreviousStorage) { Restore-ResearchStorageEnvironment $taskPreviousStorage }
    Pop-Location
    $env:PYTHONUTF8 = $taskOldUtf8
}
