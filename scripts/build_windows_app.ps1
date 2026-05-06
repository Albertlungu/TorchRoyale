Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BuildAssetsDir = Join-Path $RootDir "build_assets"
$PlatformToolsDir = Join-Path $BuildAssetsDir "platform-tools"
$VenvPython = Join-Path $RootDir "venv\Scripts\python.exe"

function Resolve-PlatformToolsDir {
    $candidates = @()

    if ($env:ANDROID_SDK_ROOT) {
        $candidates += (Join-Path $env:ANDROID_SDK_ROOT "platform-tools")
    }
    if ($env:ANDROID_HOME) {
        $candidates += (Join-Path $env:ANDROID_HOME "platform-tools")
    }
    $candidates += (Join-Path $env:LOCALAPPDATA "Android\Sdk\platform-tools")

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path (Join-Path $candidate "adb.exe"))) {
            return $candidate
        }
    }

    $adbCommand = Get-Command adb.exe -ErrorAction SilentlyContinue
    if ($adbCommand) {
        return Split-Path -Parent $adbCommand.Source
    }

    throw "Unable to locate Android platform-tools. Install adb or set ANDROID_SDK_ROOT."
}

if (-not (Test-Path $VenvPython)) {
    throw "Missing virtualenv python at $VenvPython"
}

& $VenvPython -c "import PyInstaller" | Out-Null
if ($LASTEXITCODE -ne 0) {
    & $VenvPython -m pip install pyinstaller
}

$SourcePlatformToolsDir = Resolve-PlatformToolsDir
if (Test-Path $PlatformToolsDir) {
    Remove-Item $PlatformToolsDir -Recurse -Force
}

New-Item -ItemType Directory -Force -Path $BuildAssetsDir | Out-Null
Copy-Item $SourcePlatformToolsDir $PlatformToolsDir -Recurse

Push-Location $RootDir
try {
    if (Test-Path "build") {
        Remove-Item "build" -Recurse -Force
    }
    if (Test-Path "dist") {
        Remove-Item "dist" -Recurse -Force
    }

    & $VenvPython -m PyInstaller --noconfirm TorchRoyale.windows.spec
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "Build complete:"
Write-Host "  Executable directory: $RootDir\dist\TorchRoyale"
Write-Host "  Executable: $RootDir\dist\TorchRoyale\TorchRoyale.exe"
