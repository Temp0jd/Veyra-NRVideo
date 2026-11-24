[CmdletBinding()]
param(
    [string]$Root = (Split-Path -Parent $PSScriptRoot),
    [string]$BuildDir,
    [string]$QtRoot,
    [string]$DepsRoot = 'C:\veyra-deps',
    [string]$DisplayVersion = '2.2.0-amd-sr-test',
    [int]$Jobs = 2,
    [switch]$SkipChecks
)
# One-click Windows build for the AMD SR/FG integration.
# Validates every external dependency first, then configures with Ninja and builds
# veyra_qml_ui. Nothing proprietary is copied or downloaded: the SDK trees below
# must already exist (see docs/BUILD_AMD_SR_FG_2026-11-24.md).
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Fail([string]$message) {
    Write-Host ""
    Write-Host "ERROR: $message" -ForegroundColor Red
    exit 2
}

if (-not (Test-Path -LiteralPath (Join-Path $Root 'CMakeLists.txt') -PathType Leaf)) {
    Fail "CMakeLists.txt not found under $Root (pass -Root <repo root>)"
}
if (-not $BuildDir) { $BuildDir = Join-Path $Root 'out\build\amd-sr' }

# --- Visual Studio 2022 + MSVC environment -------------------------------
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vsRoot = ''
if (Test-Path -LiteralPath $vswhere -PathType Leaf) {
    $vsRoot = (& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath) -join ''
}
if ([string]::IsNullOrWhiteSpace($vsRoot)) { Fail 'Visual Studio 2022 C++ tools not found (vswhere returned nothing)' }
$vcvars = Join-Path $vsRoot 'VC\Auxiliary\Build\vcvars64.bat'
if (-not (Test-Path -LiteralPath $vcvars -PathType Leaf)) { Fail "vcvars64.bat not found: $vcvars" }

$cmake = Join-Path $vsRoot 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
if (-not (Test-Path -LiteralPath $cmake -PathType Leaf)) {
    $found = Get-Command cmake -ErrorAction SilentlyContinue
    if ($null -eq $found) { Fail 'cmake not found (install CMake 3.24+ or the VS CMake component)' }
    $cmake = $found.Source
}

# --- dxc (Windows SDK) ---------------------------------------------------
$dxcFound = $false
$kits = Join-Path ${env:ProgramFiles(x86)} 'Windows Kits\10\bin'
if (Test-Path -LiteralPath $kits -PathType Container) {
    foreach ($kit in (Get-ChildItem -LiteralPath $kits -Directory | Sort-Object Name -Descending)) {
        if (Test-Path -LiteralPath (Join-Path $kit.FullName 'x64\dxc.exe') -PathType Leaf) { $dxcFound = $true; break }
    }
}
if (-not $dxcFound) { Fail 'dxc.exe not found under Windows Kits\10\bin\*\x64 (install the Windows 10/11 SDK)' }

# --- Qt 6.8.3 -----------------------------------------------------------
if (-not $QtRoot) {
    foreach ($candidate in @('C:\Qt\6.8.3\msvc2022_64',
                             (Join-Path $DepsRoot 'qt\6.8.3\msvc2022_64'),
                             (Join-Path $Root 'deps\qt\6.8.3\msvc2022_64'))) {
        if (Test-Path -LiteralPath (Join-Path $candidate 'lib\cmake\Qt6\Qt6Config.cmake') -PathType Leaf) { $QtRoot = $candidate; break }
    }
}
if (-not $QtRoot -or -not (Test-Path -LiteralPath (Join-Path $QtRoot 'lib\cmake\Qt6\Qt6Config.cmake') -PathType Leaf)) {
    Fail 'Qt 6.8.3 msvc2022_64 not found. Install it or pass -QtRoot "<qt>\6.8.3\msvc2022_64"'
}

# --- FFmpeg (vcpkg x64-windows tree) ------------------------------------
$ffmpegRoot = Join-Path $DepsRoot 'installed\x64-windows'
if (-not (Test-Path -LiteralPath (Join-Path $ffmpegRoot 'include\libavformat\avformat.h') -PathType Leaf)) {
    Fail "FFmpeg headers not found under $ffmpegRoot. Pass -DepsRoot or stage the vcpkg x64-windows tree."
}

$missing = New-Object System.Collections.Generic.List[string]
function Need([string]$label, [string]$relative) {
    $full = Join-Path $Root $relative
    if (-not (Test-Path -LiteralPath $full -PathType Leaf)) { $missing.Add("$label -> $full") }
}

Need 'NVIDIA DLSS SDK 310.7 header'        'third_party_local\nvidia\DLSS_SDK_310.7.0\include\nvsdk_ngx.h'
Need 'NVIDIA DLSS static shim'              'third_party_local\nvidia\DLSS_SDK_310.7.0\lib\Windows_x86_64\x64\nvsdk_ngx_s.lib'
Need 'NVIDIA Optical Flow SDK header'       'third_party_local\nvidia\Optical_Flow_SDK_5.0.7\NvOFInterface\nvOpticalFlowCommon.h'
Need 'nv-codec-headers 13.0'                'third_party_local\nvidia\nv-codec-headers-13.0\include\ffnvcodec\nvEncodeAPI.h'
Need 'AMD FidelityFX 2.3.0 FSR API header'  'third_party_local\amd\FidelityFX-SDK-2.3.0\Kits\FidelityFX\framegeneration\include\dx12\ffx_api_framegeneration_dx12.h'
Need 'AMD AMF SDK 1.5.3 FRC header'         'third_party_local\amd\AMF-SDK-1.5.3\include\components\FRC.h'

if ($missing.Count -gt 0 -and -not $SkipChecks) {
    Write-Host 'Missing dependency files:' -ForegroundColor Yellow
    foreach ($item in $missing) { Write-Host "  - $item" }
    Write-Host ''
    Write-Host 'Stage them under third_party_local/ (git-ignored). See docs/BUILD_AMD_SR_FG_2026-11-24.md section 2.'
    exit 2
}

$amfHq = Test-Path -LiteralPath (Join-Path $Root 'third_party_local\amd\AMF-SDK-1.5.3\include\components\HQScaler.h') -PathType Leaf
$amfVq = Test-Path -LiteralPath (Join-Path $Root 'third_party_local\amd\AMF-SDK-1.5.3\include\components\VQEnhancer.h') -PathType Leaf
if (-not $amfHq) { Write-Host 'WARN: HQScaler.h missing; SR id 26 (AMD AMF Video SR) will not compile in.' -ForegroundColor Yellow }
if (-not $amfVq) { Write-Host 'WARN: VQEnhancer.h missing; SR id 27 (AMD AMF quality enhance) will not compile in.' -ForegroundColor Yellow }

# libdovi is ON by default once FFmpeg is present. Use it when staged, otherwise disable
# explicitly so configure does not abort on a machine without the prefix.
$libdoviRoot = Join-Path $DepsRoot 'libdovi'
$libdoviArgs = @()
if (Test-Path -LiteralPath (Join-Path $libdoviRoot 'include\libdovi\rpu_parser.h') -PathType Leaf) {
    $libdoviArgs = @('-DVEYRA_LIBDOVI_ROOT=' + $libdoviRoot)
} else {
    Write-Host 'WARN: libdovi not staged; building with VEYRA_ENABLE_LIBDOVI=OFF (Dolby Vision RPU parsing disabled).' -ForegroundColor Yellow
    $libdoviArgs = @('-DVEYRA_ENABLE_LIBDOVI=OFF')
}

$opensslCandidates = @((Join-Path $ffmpegRoot 'lib\cmake\OpenSSL\OpenSSLConfig.cmake'),
                       (Join-Path $DepsRoot 'openssl'),
                       (Join-Path $DepsRoot 'installed\x64-windows'))
$opensslOk = $false
foreach ($candidate in $opensslCandidates) {
    if ((Test-Path -LiteralPath $candidate -PathType Leaf) -or (Test-Path -LiteralPath (Join-Path $candidate 'OpenSSLConfig.cmake') -PathType Leaf)) { $opensslOk = $true; break }
}
if (-not $opensslOk) { Write-Host 'WARN: OpenSSL CMake package not found; configure may fail at find_package(OpenSSL).' -ForegroundColor Yellow }

$prefixPath = $QtRoot + ';' + $ffmpegRoot
if (Test-Path -LiteralPath (Join-Path $DepsRoot 'openssl') -PathType Container) { $prefixPath += ';' + (Join-Path $DepsRoot 'openssl') }

# --- configure + build ---------------------------------------------------
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
$cmakeArgs = @(
    '-S', $Root, '-B', $BuildDir, '-G', 'Ninja',
    '-DCMAKE_BUILD_TYPE=Release',
    "-DCMAKE_PREFIX_PATH=$prefixPath",
    '-DVEYRA_BUILD_QML_UI=ON',
    '-DVEYRA_ENABLE_D3D12_DEBUG=OFF',
    '-DVEYRA_ENABLE_EXPERIMENTAL_DLSSNR=ON',
    ('-DVEYRA_DLSS_SDK_ROOT=' + (Join-Path $Root 'third_party_local\nvidia\DLSS_SDK_310.7.0')),
    ('-DVEYRA_NVOF_SDK_ROOT=' + (Join-Path $Root 'third_party_local\nvidia\Optical_Flow_SDK_5.0.7')),
    ('-DVEYRA_NVENC_HEADERS_ROOT=' + (Join-Path $Root 'third_party_local\nvidia\nv-codec-headers-13.0')),
    ('-DVEYRA_FIDELITYFX2_ROOT=' + (Join-Path $Root 'third_party_local\amd\FidelityFX-SDK-2.3.0\Kits\FidelityFX')),
    ('-DVEYRA_AMF_SDK_ROOT=' + (Join-Path $Root 'third_party_local\amd\AMF-SDK-1.5.3')),
    ('-DVEYRA_FIDELITYFX_ROOT=' + (Join-Path $Root 'third_party_local\amd\FidelityFX-SDK\sdk')),
    ('-DVEYRA_FFMPEG_ROOT=' + $ffmpegRoot),
    "-DVEYRA_DISPLAY_VERSION=$DisplayVersion"
) + $libdoviArgs

Write-Host 'Importing the MSVC environment from vcvars64.bat ...'
$envDump = & cmd.exe /d /c ("call `"$vcvars`" >nul && set")
if ($LASTEXITCODE -ne 0) { Fail "vcvars64.bat failed (exit $LASTEXITCODE)" }
foreach ($line in $envDump) {
    if ($line -match '^([^=]+)=(.*)$') { Set-Item -Path ("env:" + $matches[1]) -Value $matches[2] }
}

Write-Host "Configuring $BuildDir ..."
& $cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { Fail "CMake configure failed (exit $LASTEXITCODE)" }

Write-Host "Building veyra_qml_ui with $Jobs jobs ..."
& $cmake --build $BuildDir --target veyra_qml_ui --parallel $Jobs
if ($LASTEXITCODE -ne 0) { Fail "Build failed (exit $LASTEXITCODE)" }

$exe = Join-Path $BuildDir 'veyra_qml_ui.exe'
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) { Fail "Build reported success but $exe is missing" }

Write-Host ''
Write-Host "OK: $exe" -ForegroundColor Green
Write-Host 'Shaders (.dxil) are in the same build directory under shaders\; run the exe from there.'
Write-Host 'Optional runtime assets (FSR DLLs, ONNX models, AMD NR module) go under runtime_local\ or runtime\; see docs/BUILD_AMD_SR_FG_2026-11-24.md section 4.'
