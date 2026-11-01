# Shared storage mapping for project scripts and the isaac51 Conda environment.
function Get-ResearchStorageConfig {
    $taskFile = Join-Path (Split-Path -Parent $PSScriptRoot) 'configs\runtime\windows.json'
    $taskConfig = Get-Content -LiteralPath $taskFile -Raw | ConvertFrom-Json
    if (-not [IO.Path]::IsPathRooted($taskConfig.root)) { throw 'Research storage root must be an absolute path' }
    $taskConfig.root = [IO.Path]::GetFullPath($taskConfig.root)
    return $taskConfig
}

function Set-ResearchStorageEnvironment {
    $taskConfig = Get-ResearchStorageConfig
    $taskPrevious = @{}
    foreach ($taskProperty in $taskConfig.environment.PSObject.Properties) {
        if ($taskProperty.Name -in @('HOME', 'USERPROFILE', 'APPDATA', 'LOCALAPPDATA', 'PATH', 'CODEX_HOME')) {
            throw "Identity/system environment variable is not a cache path: $($taskProperty.Name)"
        }
        $taskPath = [IO.Path]::GetFullPath((Join-Path $taskConfig.root $taskProperty.Value))
        if (-not $taskPath.StartsWith($taskConfig.root.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
            throw "Storage path escapes root: $taskPath"
        }
        New-Item -ItemType Directory -Path $taskPath -Force | Out-Null
        $taskPrevious[$taskProperty.Name] = [Environment]::GetEnvironmentVariable($taskProperty.Name, 'Process')
        [Environment]::SetEnvironmentVariable($taskProperty.Name, $taskPath, 'Process')
    }
    return $taskPrevious
}

function Restore-ResearchStorageEnvironment($Previous) {
    foreach ($taskName in $Previous.Keys) {
        [Environment]::SetEnvironmentVariable($taskName, $Previous[$taskName], 'Process')
    }
}
