<#
.SYNOPSIS
    One-shot Windows setup + build for zig-ai. Audits prerequisites, installs
    whatever is missing (winget where possible, direct download otherwise),
    wires up the repo (submodules / zigui / patches / SDL3), then builds.

.DESCRIPTION
    Assumes a fresh clone and that ONLY git is installed. Everything else is
    detected and, unless -CheckOnly is given, installed:

      * Zig 0.16.0 .................. direct download from ziglang.org -> .tools\zig
      * CMake ...................... winget  Kitware.CMake
      * Ninja ...................... winget  Ninja-build.Ninja
      * ccache (optional) .......... winget  Ccache.Ccache   (caches C/C++/nvcc; -NoCcache to skip)
      * VS 2022 Build Tools (MSVC) . winget  Microsoft.VisualStudio.2022.BuildTools
      * Vulkan SDK (GPU build) ..... winget  KhronosGroup.VulkanSDK   (skip with -NoVulkan)
      * CUDA Toolkit (NVIDIA GPU) .. winget  Nvidia.CUDA              (skip with -NoCuda)
      * SDL3 3.4.10 (mingw devel) .. direct download -> sdl3-prefix\
      * zigui v0.2.0 ............... git clone -> zigui-src\
      * git submodules + patches ... git

    Build model (Windows): the three C++ AI backends (llama.cpp / stable-diffusion.cpp
    / qwen3-tts.cpp) are built as SHARED libraries with ggml's dynamic backend
    loader (GGML_BACKEND_DL). Each ggml compute backend — CPU, Vulkan and CUDA —
    becomes a loadable module that ggml discovers at runtime and selects from
    (CUDA -> Vulkan -> CPU). This is what allows the CUDA backend to be compiled
    with the NATIVE toolchain (nvcc + MSVC) — nvcc cannot use `zig cc` as its host
    compiler — while the rest of the app is built by zig: the only thing crossing
    between them is ggml's C ABI, so the C++ runtimes never have to match.

    Consequently the deps are compiled with MSVC (cl.exe), NOT `zig cc`. The zig
    executable itself is still built by zig (x86_64-windows-gnu) and links the
    shared libraries over their C API. The script imports the MSVC (vcvars)
    environment before building so CMake finds cl.exe, nvcc and the Vulkan/CUDA
    toolchains.

.PARAMETER NoVulkan
    Skip the Vulkan SDK and the ggml Vulkan backend.

.PARAMETER NoCuda
    Skip the CUDA backend even if the CUDA Toolkit is present. The default is to
    build CUDA when a CUDA Toolkit is detected (or can be installed).

.PARAMETER CudaArch
    CMAKE_CUDA_ARCHITECTURES value. Default 'native' (build only for the GPU in
    this machine — fastest local compile). Use an explicit list for portability,
    e.g. '75;80;86;89;90;120'.

.PARAMETER CheckOnly
    Report the status of every prerequisite and exit. Installs/builds nothing.

.PARAMETER SkipBuild
    Do all the setup/installs but stop before `zig build`.

.PARAMETER Optimize
    Zig optimize mode (Debug | ReleaseSafe | ReleaseFast | ReleaseSmall).
    Default ReleaseFast (what CI uses).

.PARAMETER Run
    After a successful build, run a headless `--mcp-smoke` sanity check.

.EXAMPLE
    .\scripts\build-windows.ps1 -CheckOnly         # just tell me what's missing
.EXAMPLE
    .\scripts\build-windows.ps1                    # install missing + GPU build (Vulkan + CUDA)
.EXAMPLE
    .\scripts\build-windows.ps1 -NoCuda -Run       # Vulkan only, then smoke test
#>
[CmdletBinding()]
param(
    [switch]$NoVulkan,
    [switch]$NoCuda,
    [switch]$NoCcache,
    [string]$CudaArch = 'native',
    [switch]$CheckOnly,
    [switch]$SkipBuild,
    [ValidateSet('Debug','ReleaseSafe','ReleaseFast','ReleaseSmall')]
    [string]$Optimize = 'ReleaseFast',
    [switch]$Run
)

$ErrorActionPreference = 'Stop'

# ---- pinned versions (kept in sync with .github/workflows/release.yml) --------
$ZIG_VERSION = '0.16.0'
$SDL_REF     = 'release-3.4.10'
$SDL_VER     = $SDL_REF -replace '^release-', ''
$ZIGUI_REF   = 'v0.3.1'
$ZIGUI_REPO  = 'https://github.com/ddalcu/zigui.git'
$TARGET      = 'x86_64-windows-gnu'
$Vulkan      = -not $NoVulkan
$Cuda        = -not $NoCuda
$Ccache      = -not $NoCcache

# ---- locate the repo root (this script lives in <repo>\scripts\) -------------
$RepoRoot = $PSScriptRoot
if (-not (Test-Path (Join-Path $RepoRoot 'build.zig'))) {
    $parent = Split-Path -Parent $PSScriptRoot
    if (Test-Path (Join-Path $parent 'build.zig')) { $RepoRoot = $parent }
}
if (-not (Test-Path (Join-Path $RepoRoot 'build.zig'))) {
    throw "Could not find build.zig. Run this script from inside the zig-ai repo."
}
$ToolsDir  = Join-Path $RepoRoot '.tools'
$ZiguiDir  = Join-Path $RepoRoot 'zigui-src'
$SdlPrefix = Join-Path $RepoRoot 'sdl3-prefix'

# ---- tiny output helpers -----------------------------------------------------
function Section($t) { Write-Host ''; Write-Host "== $t ==" -ForegroundColor Cyan }
function Ok($m)       { Write-Host "  [ok]   $m"   -ForegroundColor Green }
function Info($m)     { Write-Host "  [..]   $m"   -ForegroundColor Yellow }
function Miss($m)     { Write-Host "  [MISS] $m"   -ForegroundColor Red }

$Status = New-Object System.Collections.Generic.List[object]
function Record($name, $state, $detail) {
    $Status.Add([pscustomobject]@{ Component = $name; Status = $state; Detail = $detail })
}

# Re-read PATH + VULKAN_SDK + CUDA_PATH from the registry so freshly
# winget-installed tools become visible without restarting the shell.
function Update-EnvFromRegistry {
    $m = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $u = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = (@($m, $u) | Where-Object { $_ }) -join ';'
    foreach ($v in @('VULKAN_SDK', 'CUDA_PATH')) {
        $val = [Environment]::GetEnvironmentVariable($v, 'Machine')
        if (-not $val) { $val = [Environment]::GetEnvironmentVariable($v, 'User') }
        if ($val) { Set-Item -Path "env:$v" -Value $val }
    }
}

# Find an exe on PATH, else by scanning candidate glob locations.
function Find-Exe([string]$name, [string[]]$globs) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($cmd) { return $cmd.Source }
    foreach ($g in $globs) {
        $hit = Get-ChildItem -Path $g -ErrorAction SilentlyContinue | Sort-Object FullName | Select-Object -Last 1
        if ($hit) { return $hit.FullName }
    }
    return $null
}

# Run a native exe WITHOUT letting its stderr trip $ErrorActionPreference='Stop'
# (git/cmake/ninja all write normal progress to stderr). Returns the exit code.
function Invoke-Native {
    param([Parameter(Mandatory)][string]$File, [string[]]$Arguments = @(), [switch]$Quiet)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        if ($Quiet) { & $File @Arguments 2>&1 | Out-Null }
        else { & $File @Arguments 2>&1 | ForEach-Object { Write-Host ($_.ToString()) } }
    }
    finally { $ErrorActionPreference = $prev }
    return $LASTEXITCODE
}

function Install-Winget([string]$id, [string[]]$extra) {
    if ($CheckOnly) { return }
    Info "installing $id via winget (may prompt for admin)..."
    $args = @('install', '--id', $id, '-e', '--accept-source-agreements', '--accept-package-agreements', '--disable-interactivity')
    if ($extra) { $args += $extra }
    [void](Invoke-Native 'winget' $args)
    Update-EnvFromRegistry
}

function Get-File([string]$url, [string]$out) {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $old = $ProgressPreference; $ProgressPreference = 'SilentlyContinue'
    try { Invoke-WebRequest -Uri $url -OutFile $out -UseBasicParsing } finally { $ProgressPreference = $old }
}

function Get-VulkanSdk {
    if ($env:VULKAN_SDK -and (Test-Path $env:VULKAN_SDK)) { return $env:VULKAN_SDK }
    $d = Get-ChildItem 'C:\VulkanSDK' -Directory -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -Last 1
    if ($d) { return $d.FullName }
    return $null
}

function Get-CudaToolkit {
    if ($env:CUDA_PATH -and (Test-Path (Join-Path $env:CUDA_PATH 'bin\nvcc.exe'))) { return $env:CUDA_PATH }
    $root = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'
    $d = Get-ChildItem $root -Directory -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -Last 1
    if ($d -and (Test-Path (Join-Path $d.FullName 'bin\nvcc.exe'))) { return $d.FullName }
    return $null
}

# Resolve "native" to an explicit CMAKE_CUDA_ARCHITECTURES value via nvidia-smi.
# CMake's own "native" probe is unreliable (it can report "no NVIDIA GPU was
# detected" even when one is present), so we query the compute capability
# directly (e.g. "12.0" -> "120"). Falls back to whatever was passed in.
function Resolve-CudaArch([string]$arch) {
    if ($arch -ne 'native') { return $arch }
    $smi = (Get-Command nvidia-smi -ErrorAction SilentlyContinue).Source
    if (-not $smi) { $smi = 'C:\Windows\System32\nvidia-smi.exe' }
    if (Test-Path $smi) {
        $cap = (& $smi --query-gpu=compute_cap --format=csv,noheader 2>$null | Select-Object -First 1)
        if ($cap -match '^\s*(\d+)\.(\d+)\s*$') { return "$($matches[1])$($matches[2])" }
    }
    Info "could not resolve GPU compute capability; passing 'native' to CMake"
    return 'native'
}

# Locate the MSVC vcvars batch file via vswhere (installed with every VS 2017+).
function Get-VcVarsAll {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) { return $null }
    $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null | Select-Object -First 1
    if (-not $vs) { return $null }
    $vc = Join-Path $vs 'VC\Auxiliary\Build\vcvarsall.bat'
    if (Test-Path $vc) { return $vc }
    return $null
}

# Import the MSVC build environment (INCLUDE/LIB/PATH/etc.) into this session so
# that CMake — invoked as a child process by `zig build` — finds cl.exe + nvcc.
function Import-VcVars([string]$vcvarsall) {
    Info "importing MSVC environment ($vcvarsall x64)"
    $lines = & cmd /c "`"$vcvarsall`" x64 >nul 2>&1 && set"
    foreach ($line in $lines) {
        if ($line -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
    }
}

New-Item -ItemType Directory -Force -Path $ToolsDir | Out-Null

Write-Host "zig-ai Windows build" -ForegroundColor White
Write-Host ("repo: {0}" -f $RepoRoot)
$mode = if ($CheckOnly) { 'check-only' } else { 'install + build' }
$gpu  = (@(if ($Vulkan) { 'Vulkan' }; if ($Cuda) { 'CUDA' }) -join '+')
if (-not $gpu) { $gpu = 'CPU-only' }
Write-Host ("mode: {0}, {1}" -f $mode, $gpu)

# =============================================================================
# 1. git (assumed present)
# =============================================================================
Section 'git'
$gitExe = Find-Exe 'git' @()
if ($gitExe) { Ok "git -> $gitExe"; Record 'git' 'present' $gitExe }
else {
    Miss 'git not found. Install it first: winget install Git.Git'
    Record 'git' 'MISSING' 'required'
    throw 'git is required and is the one prerequisite this script will not install.'
}

# =============================================================================
# 2. Zig 0.16.0  (not on winget at a pinned version -> direct download)
# =============================================================================
Section "Zig $ZIG_VERSION"
$ZigExe = $null
$onPath = Get-Command zig -ErrorAction SilentlyContinue | Select-Object -First 1
if ($onPath) {
    $v = (& $onPath.Source version) 2>$null
    if ($v -like '0.16*') { $ZigExe = $onPath.Source; Ok "zig $v -> $ZigExe" }
    else { Info "zig $v on PATH is not 0.16.x; will use a pinned local copy" }
}
if (-not $ZigExe) {
    $localZig = Get-ChildItem (Join-Path $ToolsDir 'zig-*') -Filter zig.exe -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($localZig) { $ZigExe = $localZig.FullName; Ok "zig (local) -> $ZigExe" }
}
if (-not $ZigExe) {
    if ($CheckOnly) { Miss "Zig $ZIG_VERSION (would download to $ToolsDir)"; Record 'zig' 'MISSING' "download $ZIG_VERSION" }
    else {
        $zigUrl = "https://ziglang.org/download/$ZIG_VERSION/zig-x86_64-windows-$ZIG_VERSION.zip"
        $zigZip = Join-Path $ToolsDir "zig-$ZIG_VERSION.zip"
        Info "downloading $zigUrl"
        Get-File $zigUrl $zigZip
        Info 'extracting...'
        Expand-Archive -Path $zigZip -DestinationPath $ToolsDir -Force
        $ZigExe = (Get-ChildItem (Join-Path $ToolsDir 'zig-*') -Filter zig.exe -Recurse | Select-Object -First 1).FullName
        Ok "zig -> $ZigExe"
    }
}
if ($ZigExe) {
    $ZigDir = Split-Path -Parent $ZigExe
    Record 'zig' 'present' $ZigExe
}

# =============================================================================
# 3. CMake / Ninja (winget)
# =============================================================================
Section 'CMake'
$CMakeExe = Find-Exe 'cmake' @('C:\Program Files\CMake\bin\cmake.exe')
if (-not $CMakeExe) {
    if ($CheckOnly) { Miss 'CMake (winget Kitware.CMake)'; Record 'cmake' 'MISSING' 'winget Kitware.CMake' }
    else { Install-Winget 'Kitware.CMake'; $CMakeExe = Find-Exe 'cmake' @('C:\Program Files\CMake\bin\cmake.exe') }
}
if ($CMakeExe) { Ok "cmake -> $CMakeExe"; Record 'cmake' 'present' $CMakeExe }

Section 'Ninja'
$ninjaGlobs = @(
    "$env:LOCALAPPDATA\Microsoft\WinGet\Links\ninja.exe",
    "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\Ninja-build.Ninja_*\ninja.exe"
)
$NinjaExe = Find-Exe 'ninja' $ninjaGlobs
if (-not $NinjaExe) {
    if ($CheckOnly) { Miss 'Ninja (winget Ninja-build.Ninja)'; Record 'ninja' 'MISSING' 'winget Ninja-build.Ninja' }
    else { Install-Winget 'Ninja-build.Ninja'; $NinjaExe = Find-Exe 'ninja' $ninjaGlobs }
}
if ($NinjaExe) { Ok "ninja -> $NinjaExe"; Record 'ninja' 'present' $NinjaExe }

# =============================================================================
# 3b. ccache (optional) — caches compiled objects so rebuilds are near-instant.
# ggml enables it automatically (GGML_CCACHE=ON) when ccache is on PATH at
# configure time, and applies it to C, C++ AND nvcc/CUDA compiles, which is the
# slowest part of a clean build. Skipped with -NoCcache; never fatal if absent.
# =============================================================================
$CcacheExe = $null
if ($Ccache) {
    Section 'ccache (optional)'
    $ccacheGlobs = @(
        "$env:LOCALAPPDATA\Microsoft\WinGet\Links\ccache.exe",
        "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\Ccache.Ccache_*\ccache-*\ccache.exe"
    )
    $CcacheExe = Find-Exe 'ccache' $ccacheGlobs
    if (-not $CcacheExe) {
        if ($CheckOnly) { Miss 'ccache (winget Ccache.Ccache) — optional, speeds up rebuilds'; Record 'ccache' 'optional' 'winget Ccache.Ccache' }
        else { Install-Winget 'Ccache.Ccache'; $CcacheExe = Find-Exe 'ccache' $ccacheGlobs }
    }
    if ($CcacheExe) { Ok "ccache -> $CcacheExe"; Record 'ccache' 'present' $CcacheExe }
    else { Record 'ccache' 'optional' 'not installed (rebuilds slower)' }
}
else { Record 'ccache' 'skipped' '-NoCcache' }

# =============================================================================
# 4. MSVC (VS 2022 Build Tools) — the C++ deps and the CUDA host compile use it
# =============================================================================
Section 'MSVC (VS 2022 Build Tools)'
$VcVars = Get-VcVarsAll
if (-not $VcVars) {
    if ($CheckOnly) { Miss 'MSVC / VC++ Build Tools (winget Microsoft.VisualStudio.2022.BuildTools)'; Record 'msvc' 'MISSING' 'VC.Tools.x86.x64 workload' }
    else {
        # The VS installer needs elevation; winget triggers a UAC prompt. The
        # --override carries the workload selection to the bootstrapper.
        Install-Winget 'Microsoft.VisualStudio.2022.BuildTools' @('--override', '--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows11SDK.22621 --includeRecommended')
        $VcVars = Get-VcVarsAll
    }
}
if ($VcVars) { Ok "vcvarsall -> $VcVars"; Record 'msvc' 'present' $VcVars }
else { Record 'msvc' 'MISSING' 'required for the C++ deps and CUDA' }

# =============================================================================
# 5. Vulkan SDK (GPU build only)
# =============================================================================
$VulkanSdk = $null
if ($Vulkan) {
    Section 'Vulkan SDK'
    $VulkanSdk = Get-VulkanSdk
    $glslc = if ($VulkanSdk) { Find-Exe 'glslc' @((Join-Path $VulkanSdk 'Bin\glslc.exe')) } else { Find-Exe 'glslc' @('C:\VulkanSDK\*\Bin\glslc.exe') }
    if (-not $VulkanSdk -or -not $glslc) {
        if ($CheckOnly) { Miss 'Vulkan SDK (winget KhronosGroup.VulkanSDK)'; Record 'vulkan-sdk' 'MISSING' 'winget KhronosGroup.VulkanSDK' }
        else {
            Install-Winget 'KhronosGroup.VulkanSDK'
            $VulkanSdk = Get-VulkanSdk
        }
    }
    if ($VulkanSdk) { Ok "Vulkan SDK -> $VulkanSdk"; Record 'vulkan-sdk' 'present' $VulkanSdk }
    else { Record 'vulkan-sdk' 'MISSING' 'GPU build needs it' }
}
else { Record 'vulkan-sdk' 'skipped' '-NoVulkan' }

# =============================================================================
# 6. CUDA Toolkit (NVIDIA GPU backend)
# =============================================================================
$CudaPath = $null
if ($Cuda) {
    Section 'CUDA Toolkit'
    $CudaPath = Get-CudaToolkit
    if (-not $CudaPath) {
        if ($CheckOnly) { Miss 'CUDA Toolkit (winget Nvidia.CUDA)'; Record 'cuda' 'MISSING' 'winget Nvidia.CUDA' }
        else {
            Info 'CUDA Toolkit not found; attempting winget install (large download)...'
            Install-Winget 'Nvidia.CUDA'
            $CudaPath = Get-CudaToolkit
        }
    }
    if ($CudaPath) {
        $nvccVer = (& (Join-Path $CudaPath 'bin\nvcc.exe') --version 2>$null | Select-String 'release' | ForEach-Object { $_.ToString().Trim() })
        Ok "CUDA -> $CudaPath  ($nvccVer)"
        Record 'cuda' 'present' $CudaPath
    }
    else {
        # Not fatal: fall back to a Vulkan/CPU build so the user still gets a binary.
        Miss 'CUDA Toolkit not available; building WITHOUT the CUDA backend (Vulkan/CPU only).'
        Record 'cuda' 'MISSING' 'building without CUDA'
        $Cuda = $false
    }
}
else { Record 'cuda' 'skipped' '-NoCuda' }

# =============================================================================
# 7. git submodules
# =============================================================================
Section 'Submodules (llama.cpp / stable-diffusion.cpp / qwen3-tts.cpp)'
$subOk = Test-Path (Join-Path $RepoRoot 'deps\llama.cpp\CMakeLists.txt')
if ($subOk) { Ok 'submodules populated'; Record 'submodules' 'present' '' }
elseif ($CheckOnly) { Miss 'submodules not checked out (git submodule update --init --recursive)'; Record 'submodules' 'MISSING' '' }
else {
    Info 'git submodule update --init --recursive (large; first run is slow)...'
    Push-Location $RepoRoot
    try { $code = Invoke-Native $gitExe @('submodule', 'update', '--init', '--recursive') } finally { Pop-Location }
    if ($code -ne 0) { throw 'submodule checkout failed' }
    Ok 'submodules checked out'; Record 'submodules' 'installed' ''
}

# =============================================================================
# 8. zigui checkout
# =============================================================================
Section "zigui $ZIGUI_REF"
if (Test-Path (Join-Path $ZiguiDir '.git')) {
    if (-not $CheckOnly) {
        Push-Location $ZiguiDir
        try {
            [void](Invoke-Native $gitExe @('fetch', '--tags', '--depth', '1', 'origin', $ZIGUI_REF) -Quiet)
            [void](Invoke-Native $gitExe @('checkout', '--quiet', $ZIGUI_REF) -Quiet)
        } finally { Pop-Location }
    }
    Ok "zigui -> $ZiguiDir"; Record 'zigui' 'present' $ZiguiDir
}
elseif ($CheckOnly) { Miss "zigui not cloned (-> $ZiguiDir)"; Record 'zigui' 'MISSING' $ZIGUI_REPO }
else {
    Info "cloning $ZIGUI_REPO @ $ZIGUI_REF"
    $code = Invoke-Native $gitExe @('clone', '--depth', '1', '--branch', $ZIGUI_REF, $ZIGUI_REPO, $ZiguiDir)
    if ($code -ne 0) { throw 'zigui clone failed' }
    Ok "zigui -> $ZiguiDir"; Record 'zigui' 'installed' $ZiguiDir
}

# =============================================================================
# 9. SDL3 (prebuilt mingw devel package)
# =============================================================================
Section "SDL3 $SDL_VER"
if (Test-Path (Join-Path $SdlPrefix 'lib\SDL3.lib')) { Ok "SDL3 staged -> $SdlPrefix"; Record 'sdl3' 'present' $SdlPrefix }
elseif ($CheckOnly) { Miss "SDL3 not staged (-> $SdlPrefix)"; Record 'sdl3' 'MISSING' "$SDL_VER mingw devel" }
else {
    $sdlZip = Join-Path $ToolsDir "SDL3-devel-$SDL_VER-mingw.zip"
    $sdlUrl = "https://github.com/libsdl-org/SDL/releases/download/$SDL_REF/SDL3-devel-$SDL_VER-mingw.zip"
    Info "downloading $sdlUrl"
    Get-File $sdlUrl $sdlZip
    $sdlTmp = Join-Path $ToolsDir 'sdl3-unzip'
    if (Test-Path $sdlTmp) { Remove-Item $sdlTmp -Recurse -Force }
    Expand-Archive -Path $sdlZip -DestinationPath $sdlTmp -Force
    $mingw = Join-Path $sdlTmp "SDL3-$SDL_VER\x86_64-w64-mingw32"
    if (-not (Test-Path $mingw)) { throw "unexpected SDL3 archive layout: $mingw not found" }
    if (Test-Path $SdlPrefix) { Remove-Item $SdlPrefix -Recurse -Force }
    Copy-Item $mingw $SdlPrefix -Recurse
    # zig probes SDL3.lib but the mingw import lib is named libSDL3.dll.a.
    Copy-Item (Join-Path $SdlPrefix 'lib\libSDL3.dll.a') (Join-Path $SdlPrefix 'lib\SDL3.lib') -Force
    Ok "SDL3 staged -> $SdlPrefix"; Record 'sdl3' 'installed' $SdlPrefix
}

# =============================================================================
# 10. submodule patches (idempotent)
# =============================================================================
Section 'Submodule patches'
function Apply-Patch([string]$subRel, [string]$patchName) {
    $patch = Join-Path $RepoRoot "deps\patches\$patchName"
    $sub   = Join-Path $RepoRoot $subRel
    if (-not (Test-Path $sub)) { Miss "$subRel missing (submodules not checked out)"; return }
    Push-Location $sub
    try {
        if ((Invoke-Native $gitExe @('apply', '--reverse', '--check', $patch) -Quiet) -eq 0) {
            Ok "$patchName already applied"; Record "patch:$patchName" 'present' ''; return
        }
        if ((Invoke-Native $gitExe @('apply', '--check', $patch) -Quiet) -ne 0) {
            Miss "$patchName does not apply cleanly (manual check needed)"; Record "patch:$patchName" 'CONFLICT' ''; return
        }
        if ($CheckOnly) { Miss "$patchName not applied"; Record "patch:$patchName" 'MISSING' ''; return }
        [void](Invoke-Native $gitExe @('apply', $patch) -Quiet)
        Ok "$patchName applied"; Record "patch:$patchName" 'installed' ''
    }
    finally { Pop-Location }
}
Apply-Patch 'deps\llama.cpp'             'llama.cpp-metal-left-pad.patch'
Apply-Patch 'deps\stable-diffusion.cpp'  'stable-diffusion.cpp-conv3d-direct.patch'
Apply-Patch 'deps\qwen3-tts.cpp'         'qwen3-tts.cpp-win-portability.patch'

# =============================================================================
# Status summary
# =============================================================================
Section 'Summary'
$Status | Format-Table -AutoSize Component, Status, Detail | Out-String | Write-Host
$missing = $Status | Where-Object { $_.Status -in @('MISSING', 'CONFLICT') }

if ($CheckOnly) {
    if ($missing) { Write-Host "Missing: $($missing.Component -join ', '). Re-run without -CheckOnly to install + build." -ForegroundColor Yellow }
    else { Write-Host 'All prerequisites present. Re-run without -CheckOnly to build.' -ForegroundColor Green }
    return
}
if ($missing) { throw "Setup incomplete: $($missing.Component -join ', '). See messages above." }
if (-not $VcVars) { throw 'MSVC is required to build the C++ deps on Windows. Install VS 2022 Build Tools (VC++ workload).' }
if ($Vulkan -and -not $VulkanSdk) { throw 'Vulkan build requested but no Vulkan SDK; use -NoVulkan to skip it.' }
if ($SkipBuild) { Write-Host 'Setup complete (-SkipBuild). Skipping zig build.' -ForegroundColor Green; return }

# =============================================================================
# 11. Build
# =============================================================================
Section "Build (zig build, $Optimize, $gpu)"

# Import the MSVC environment FIRST so CMake (a child of `zig build`) picks up
# cl.exe / nvcc / the Windows SDK. Then put our toolchain dirs on PATH so cmake,
# ninja, glslc and the CUDA bin are all resolvable.
Import-VcVars $VcVars

$buildDirs = @($ZigDir, (Split-Path -Parent $CMakeExe), (Split-Path -Parent $NinjaExe))
if ($Vulkan)    { $buildDirs += (Join-Path $VulkanSdk 'Bin') }
if ($Cuda)      { $buildDirs += (Join-Path $CudaPath 'bin') }
if ($CcacheExe) { $buildDirs += (Split-Path -Parent $CcacheExe) }
$env:Path = (($buildDirs | Where-Object { $_ }) -join ';') + ';' + $env:Path

# ggml caches its ccache probe in CMakeCache. If build-deps was configured
# before ccache existed, a one-time clean reconfigure is needed to engage it.
$cacheFile = Join-Path $RepoRoot 'build-deps\CMakeCache.txt'
if ($CcacheExe -and (Test-Path $cacheFile) -and -not (Select-String -Path $cacheFile -Pattern 'GGML_CCACHE_FOUND.*[\\/]ccache' -Quiet)) {
    Info 'ccache is installed but build-deps was configured without it.'
    Info 'Delete build-deps once to engage caching:  Remove-Item -Recurse -Force build-deps'
}

# NOTE: deliberately NOT setting CC/CXX to `zig cc`. The deps build natively with
# MSVC (and nvcc for CUDA); the zig exe links the resulting shared libraries over
# their C ABI. Let CMake auto-detect cl.exe from the vcvars environment.
if ($Vulkan) { $env:VULKAN_SDK = $VulkanSdk }
if ($Cuda)   { $env:CUDA_PATH = $CudaPath }

$zigArgs = @('build', "-Doptimize=$Optimize", "-Dtarget=$TARGET",
             "-Dzigui=$ZiguiDir", "-Dsdl3=$SdlPrefix")
if ($Vulkan) { $zigArgs += "-Dvulkan-prefix=$VulkanSdk" } else { $zigArgs += '-Dvulkan=false' }
if ($Cuda) {
    $arch = Resolve-CudaArch $CudaArch
    Info "CUDA architecture(s): $arch"
    $zigArgs += '-Dcuda=true'; $zigArgs += "-Dcuda-arch=$arch"
}

Write-Host ("> {0} {1}" -f $ZigExe, ($zigArgs -join ' ')) -ForegroundColor DarkGray
Push-Location $RepoRoot
try { $code = Invoke-Native $ZigExe $zigArgs } finally { Pop-Location }
if ($code -ne 0) { throw "zig build failed (exit $code)" }

# =============================================================================
# 12. Package: stage the exe + every shared library it needs at runtime
# =============================================================================
Section 'Package runtime libraries'
$exe    = Join-Path $RepoRoot 'zig-out\bin\zig-ai.exe'
$binDir = Split-Path -Parent $exe

# SDL3.
$dll = Join-Path $SdlPrefix 'bin\SDL3.dll'
if (Test-Path $dll) { Copy-Item $dll $binDir -Force }

# The AI shared libs + ggml backend modules (ggml.dll, ggml-base.dll,
# ggml-cpu.dll, ggml-vulkan.dll, ggml-cuda.dll, llama.dll, stable-diffusion.dll,
# qwen3_tts.dll). ggml loads the *-cpu/-vulkan/-cuda modules from this directory
# at runtime and picks the best available device.
$depBin = Join-Path $RepoRoot 'build-deps\bin'
if (Test-Path $depBin) {
    Get-ChildItem $depBin -Filter *.dll | ForEach-Object { Copy-Item $_.FullName $binDir -Force }
    Ok ("staged {0} dep DLL(s) from build-deps\bin" -f (Get-ChildItem $depBin -Filter *.dll).Count)
}

# CUDA runtime redistributables (cudart / cublas / cublasLt) so the machine
# doesn't need the full CUDA Toolkit installed to run the CUDA backend.
if ($Cuda) {
    # CUDA 12.x keeps these in bin\; CUDA 13.x moved them to bin\x64\ — check both.
    $copied = 0
    foreach ($sub in @('bin', 'bin\x64')) {
        $cudaBin = Join-Path $CudaPath $sub
        foreach ($pat in @('cudart64_*.dll', 'cublas64_*.dll', 'cublasLt64_*.dll')) {
            Get-ChildItem (Join-Path $cudaBin $pat) -ErrorAction SilentlyContinue | ForEach-Object {
                Copy-Item $_.FullName $binDir -Force; $copied++
            }
        }
    }
    Ok "staged $copied CUDA runtime DLL(s)"
}

Section 'Done'
if (Test-Path $exe) {
    $mb = [math]::Round((Get-Item $exe).Length / 1MB, 1)
    Ok "built $exe ($mb MB)"
}
Write-Host "Launch the app:  & '$exe'" -ForegroundColor Green

if ($Run) {
    Section 'Smoke test (--mcp-smoke)'
    $code = Invoke-Native $exe @('--mcp-smoke')
    Write-Host ("exit: {0}" -f $code)
}
