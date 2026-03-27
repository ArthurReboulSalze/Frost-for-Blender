# Frost for Blender - Technical Reference

Scope: addon release `1.25.0`  
Supported Blender version: `5+`

---

## Architecture

High-level stack:

```text
Blender UI / bpy
  -> frost_blender_addon (Python)
  -> native_bridge.py (ctypes)
  -> frost_native.dll (C API / C++)
  -> CPU Thinkbox Frost core
  -> optional GPU backend
```

Main Python files:

- `frost_blender_addon/operator.py`
- `frost_blender_addon/ui.py`
- `frost_blender_addon/particle_extractor.py`
- `frost_blender_addon/native_bridge.py`

Main native files:

- `blender_frost_adapter/src/frost_interface.cpp`
- `blender_frost_adapter/src/frost_native_api.cpp`
- `blender_frost_adapter/src/gpu_backend.cpp`
- `blender_frost_adapter/src/vulkan_backend.cpp`
- `blender_frost_adapter/src/vulkan_compute_shader.cpp`
- `blender_frost_adapter/src/cuda/frost_cuda.cu`

---

## Python Layer

Responsibilities:

- extract source particles from Blender
- transform world-space source particles into the Frost object local space
- map UI properties to native parameters
- write the generated mesh back with `foreach_set`
- cache native contexts for repeated updates
- record the backend actually used for the last meshing pass

The addon now also stores lightweight status about the last run:

- backend used
- whether GPU fell back to CPU
- short fallback reason

This state is updated only when Frost actually runs, so it is effectively free at UI idle time.

---

## Native Bridge

`frost_native.dll` exposes a small C API and is loaded through `ctypes`.

This matters because Blender `5.1+` moved to a newer embedded Python, and the old `.pyd` path is much more fragile across Blender version changes.

Main exposed operations:

- create / destroy native context
- set particles
- set parameters
- generate mesh
- copy vertex / face buffers
- query GPU backend and Vulkan probe status
- query last meshing backend / fallback state

---

## CPU Path

The CPU path remains the stable reference implementation.

It still goes through the integrated Thinkbox Frost meshing core and is the most feature-complete option for:

- `Union of Spheres`
- `Metaball`
- `Zhu-Bridson`
- `Anisotropic`

---

## Vulkan Path

The current Vulkan backend is functional, but still under active development.

Today it can:

- upload particles into Vulkan-managed buffers
- plan voxel influence bounds
- build coverage and scalar-field data
- classify Marching Cubes surface cells
- emit surface triangles
- attempt a direct GPU-oriented surface path
- fall back to safer scalar-field extraction when needed

Important nuance:

- the Vulkan path is real and usable
- but it is not yet the universal best-performance path on every scene

Current testing trend:

- CPU still usually wins on low-poly scenes
- Vulkan can already win on heavier / high-poly scenes depending on the setup

---

## Current Safety Strategy

The GPU path now uses several safety layers:

- validation of raw GPU vertex/index buffers before they are returned to Blender
- rejection of invalid or clearly pathological direct GPU surface geometry
- fallback from unsafe direct Vulkan surface extraction to safer GPU-derived scalar-field extraction
- final fallback to CPU meshing when GPU extraction is not safe enough

This is why Blender stability improved even when aggressive GPU paths are being tested.

---

## Vertex Refinement

`Vertex Refinement` is still treated as CPU-only for the final surface build.

Current behavior:

- `Vertex Refinement = 0`: Frost can stay on the direct GPU surface path
- `Vertex Refinement > 0`: Frost switches the final surface build back to the CPU path

Reason:

- the current Vulkan path does not yet reproduce the CPU refinement stage safely enough for release use

So this is a current capability gap, not a UI bug.

---

## Mesh Source Limitation

For `MESH` sources, the addon still reads `obj.data` directly.

Implication:

- non-applied mesh modifiers are not automatically evaluated as Frost source geometry

---

## Build Notes

Typical Windows native build:

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" && cmake -S K:\Codex_Projects\Projet_Frost_Plugin_For_Blender_CODEX\blender_frost_adapter -B K:\Codex_Projects\Projet_Frost_Plugin_For_Blender_CODEX\build_vulkan -DCMAKE_TOOLCHAIN_FILE=K:\Codex_Projects\Projet_Frost_Plugin_For_Blender_CODEX\deps\vcpkg\scripts\buildsystems\vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows -DFROST_BUILD_PYTHON_MODULE=OFF -A x64 && cmake --build K:\Codex_Projects\Projet_Frost_Plugin_For_Blender_CODEX\build_vulkan --config Release --target frost_native --parallel 8'
```

After build:

1. copy `build_vulkan/Release/frost_native.dll` into `frost_blender_addon/`
2. restart Blender

---

## Packaging

The current public distribution model is:

- one Blender install zip
- CPU path included
- Vulkan path included
- no separate user-facing CPU-only / GPU-only addon packages

---

## Known Limitations

- Vulkan is still being tuned for stability and performance.
- `Vertex Refinement` still forces the final surface build back to CPU.
- `MESH` sources still do not use evaluated modifier-stack geometry automatically.
- The legacy CUDA code still exists in the source tree, but the public release direction is the single `CPU + Vulkan` addon package.

---

Document version: `1.7`  
Last update: `2026-03-27`
