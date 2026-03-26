@echo off
setlocal

echo ==========================================
echo Building Frost Blender Adapter (C++ Core)
echo ==========================================

echo Setting up Visual Studio Environment...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

echo Setting up CMake path...
set "PATH=%PATH%;C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"

if not exist deps\vcpkg\vcpkg.exe (
    echo Error: Vcpkg not found in deps\vcpkg.
    echo Please run the setup steps first.
    pause
    exit /b 1
)

echo Creating build directory...
if not exist build mkdir build
cd build

echo Configuring CMake...
echo Toolchain: ..\deps\vcpkg\scripts\buildsystems\vcpkg.cmake
cmake ../blender_frost_adapter -DCMAKE_TOOLCHAIN_FILE=../deps/vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows -A x64

if errorlevel 1 (
    echo CMake configuration failed!
    pause
    exit /b 1
)

echo Building Release...
cmake --build . --config Release --parallel 8

if errorlevel 1 (
    echo Build failed!
    pause
    exit /b 1
)

echo Copying extension to addon folder...
copy Release\*.pyd ..\frost_blender_addon\blender_frost_adapter.pyd
if exist Release\frost_native.dll copy Release\frost_native.dll ..\frost_blender_addon\
if exist Release\blender_frost_adapter.dll copy Release\blender_frost_adapter.dll ..\frost_blender_addon\

echo ==========================================
echo Build Complete!
echo You can now use the addon in Blender.
echo ==========================================
pause
