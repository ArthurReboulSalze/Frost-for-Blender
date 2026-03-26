# Frost Particle Meshing - Technical Reference

## Scope

This document describes the current architecture of the Blender addon and the native CPU / CUDA / Vulkan backends as of addon version `1.24.1`.

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
  -> GPU backend implementation

Legacy compatibility path:

```text
Blender UI / bpy
  -> frost_blender_addon (Python)
  -> blender_frost_adapter.pyd (Pybind11 / C++)
  -> CPU Thinkbox Frost core
  -> GPU backend implementation
```
```

Main Python files:

- `frost_blender_addon/operator.py`
- `frost_blender_addon/ui.py`
- `frost_blender_addon/particle_extractor.py`

Main native files:

- `blender_frost_adapter/src/adapter.cpp`
- `blender_frost_adapter/src/frost_interface.cpp`
- `blender_frost_adapter/src/gpu_backend.cpp`
- `blender_frost_adapter/src/frost_native_api.cpp`
- `blender_frost_adapter/src/vulkan_probe.cpp`
- `blender_frost_adapter/src/vulkan_compute_probe.cpp`
- `blender_frost_adapter/src/vulkan_buffer_probe.cpp`
- `blender_frost_adapter/src/cuda/frost_cuda.cu`

---

## Python Layer

### Responsibilities

- collect source particles from Blender
- transform source positions into the Frost object's local space
- map Blender UI properties to native parameters
- write the resulting mesh back to Blender with `foreach_set`

### Current Mesh Source Limitation

`extract_from_mesh_vertices()` currently reads `obj.data` directly for `MESH` sources.

Implication:

- non-applied mesh modifiers are not evaluated automatically as Frost source geometry

This is currently a documented limitation rather than a hidden bug.

---

## Native Interface

`FrostInterface` is now a Python-agnostic C++ entry point.

It is consumed by:

- `frost_native.dll` through a plain C API
- the legacy `blender_frost_adapter.pyd` module through a thin Pybind11 wrapper

Main methods:

- `set_particles(positions, radii, velocities)`
- `set_parameter(name, value)`
- `generate_mesh()`

The CPU path dispatches to the upstream Thinkbox Frost meshing pipeline.

The GPU path now goes through a backend abstraction layer before entering the shared post-processing stage.

### Backend Abstraction

The native layer now separates:

- `FrostInterface`: shared particle ingest, parameter mapping, CPU fallback, post-processing
- `gpu_backend.cpp`: backend selection and GPU backend implementation entry point
- `frost_native_api.cpp`: stable C ABI exposed to Python through `ctypes`

Today, the compiled backend can be either:

- the legacy CUDA meshing path, when explicitly enabled
- the new `vulkan-experimental` path, when Vulkan buffer support is available in the current native build

This matters because Vulkan is now introduced as another native backend implementation instead of being hard-wired inside `FrostInterface`.

### Vulkan Runtime Probe

The native layer now also contains a lightweight Vulkan runtime probe.

Characteristics:

- no Vulkan SDK is required for this probe to compile
- the probe loads the system Vulkan runtime dynamically
- it checks whether a Vulkan instance can be created
- it enumerates physical devices to verify that a usable runtime is present

By itself, this does not mesh particles yet.

Its purpose is to validate that a Vulkan backend can be distributed with fewer external build and runtime assumptions than the old CUDA-specific path.

### Vulkan Compute Probe

The native layer now also validates a more realistic next step: whether a minimal Vulkan compute context can be created.

This probe currently goes further than the runtime probe:

- create a Vulkan instance
- enumerate physical devices
- find a compute-capable queue family
- create a logical device
- retrieve a compute queue

By itself, this still does not mesh particles yet.

But it confirms that the host machine is capable of running the first real compute stages of a Vulkan backend without relying on CUDA.

### Vulkan Storage Buffer Probe

The native layer now also verifies a first memory path that a real Vulkan meshing backend will need immediately:

- create a storage buffer
- query memory requirements
- allocate host-visible coherent memory
- bind the memory
- map and write to the buffer from the CPU

This still does not dispatch a compute shader.

But it confirms that particle upload into Vulkan-managed buffers is already viable on the current machine and runtime, and it is now directly reused by the first experimental Vulkan backend.

### Experimental Vulkan Backend

The native layer now includes an initial `vulkan-experimental` backend.

Current behavior:

- upload particle position / radius packets into Vulkan-managed buffers
- record and submit a real Vulkan command buffer
- dispatch a first Vulkan compute shader on those particle buffers
- write per-particle voxel influence bounds that match the currently selected Frost meshing method
- dispatch a second Vulkan compute shader that accumulates integer voxel coverage over the planned domain
- dispatch a third Vulkan compute shader that now derives method-aware dense scalar fields instead of applying one generic approximation to every method
- use a dedicated `Metaball` threshold field in that same scalar stage so `Metaball` no longer collapses into the generic signed-distance approximation
- use a dedicated `Plain Marching Cubes` density field in that same scalar stage so its explicit `isovalue` now affects the experimental Vulkan surface path too
- use a first dedicated `Zhu-Bridson` blended-center / blended-radius approximation instead of routing `Zhu-Bridson` through the same simple sphere signed-distance field
- that `Zhu-Bridson` path now follows the Frost CPU weighting model more closely by using a cubic compact-support kernel, raw particle radii in the blended-radius accumulator, and a uniform support radius based on the maximum particle radius during voxel planning
- use a first dedicated velocity-aware `Anisotropic` field approximation so anisotropic meshing no longer falls back to the same isotropic sphere field as the generic default
- compute velocity-aware anisotropic voxel bounds per particle in the first dispatch, so the planned domain follows stretched particles more tightly instead of only enlarging a global isotropic radius
- prune scalar-field particle evaluation against those voxel bounds, so the compute shader skips particles that cannot influence the current voxel and the CPU-side exact edge refinement uses the same bound checks
- compact active voxel indices after the coverage stage, prefill the dense scalar field once on the CPU, and dispatch the scalar-field compute stage only on covered voxels instead of all planned voxels
- build compact `active voxel -> candidate particle` tables from those bounds, so both the Vulkan scalar-field shader and the CPU-side exact edge refinement can iterate only particles that overlap each active voxel
- build those compact `active voxel -> candidate particle` tables by sparse active row ranges rather than by scanning every voxel inside each particle AABB, so setup cost stays closer to active coverage than to the full AABB volume
- cache exact scalar-field samples on shared grid corners during experimental Marching Cubes refinement, and enable that refinement only when the compact voxel-particle budget stays within a manageable range
- build compact candidate particle sets once per refined Marching Cubes edge by directly merging neighboring voxel candidate ranges, then reuse them for both cached edge-corner evaluation and midpoint bisection so exact edge refinement stays local instead of rebuilding broader sample searches on every iteration
- compact candidate Marching Cubes cells from those active voxels before CPU extraction, so the experimental surface extraction avoids scanning the full grid when the occupied region is sparse
- prioritize candidate Marching Cubes cells around negative field samples first, then fall back to coverage-derived candidates only if no interior samples are available yet
- reuse the compact active-voxel index list emitted by the Vulkan scalar-field preparation path, so candidate discovery no longer has to rescan the dense scalar grid to find those negative samples
- classify those candidate Marching Cubes cells in a dedicated Vulkan compute pass and return active cell `cubeIndex` values, so the extraction stage can start from GPU-classified surface cells instead of recomputing every cell state on the CPU
- emit Marching Cubes triangle vertices in a dedicated Vulkan compute pass directly from those GPU-classified surface cells
- rebuild a welded `trimesh3` from those Vulkan-emitted triangles on the CPU side using shared Marching Cubes edge identities, without re-running CPU Marching Cubes in the successful direct path
- bypass `trimesh3` entirely and return raw vertex/index buffers directly from the Vulkan backend when no CPU push / relax / refinement stage is requested
- disable direct GPU surface extraction whenever Frost `Vertex Refinement` is enabled, so those cases stay on the stable Frost CPU meshing path
- skip the CPU-side compact `active voxel -> candidate particle` table build on that direct raw-buffer path, and instead let the Vulkan scalar-field shader walk per-particle voxel bounds directly on the GPU
- skip output particle readback on that direct raw-buffer path when the caller does not need transformed particle packets back on the CPU
- include a first dense full-cell surface-mesh compute path for very small domains, while larger domains still default to the sparse direct path because it remains faster on the current implementation
- include a first sparse GPU candidate-cell classification pass driven directly from active voxels, while keeping it under a very conservative gate until it proves faster than the current hybrid candidate preparation path
- reuse shared descriptor pools, descriptor sets, and command buffers for both the surface passes and the hot planning passes (`particles`, `coverage`, `scalar field`), so repeated generations avoid rebuilding the main dispatch plumbing each time
- compact active voxels directly on the GPU after the coverage pass, then read back only the compact active-voxel list and aggregate stats needed by the CPU-side fallbacks, instead of rescanning the full dense coverage buffer every generation
- keep the dense scalar field resident on the GPU on the direct raw-buffer path, and only read it back later if a CPU fallback truly needs host-side scalar samples
- compact sparse surface-cell `cubeIndex` data directly on the GPU before triangle emission, so the sparse direct path no longer rescans a dense surface-cell grid on the CPU just to rebuild the active-cell list
- reuse those resident compact active-cell buffers directly for the next Vulkan triangle-emission pass, so the sparse direct path also removes the old compact-cell CPU-to-GPU upload roundtrip
- emit compact `xyz` triangle buffers instead of padded `vec4` output on the direct surface path, reducing GPU/CPU transfer pressure during direct mesh packaging
- package the direct Vulkan mesh with an adaptive packed-edge weld/index strategy, using sorting for smaller meshes and a packed-edge hash path for larger ones
- clamp the experimental `Union of Spheres` field more like the Frost CPU policy instead of reusing the broader planning radius directly
- keep density-based modes from inventing matter in voxels that lie inside coarse particle AABBs but outside the real particle support
- refine extracted edge vertices against the exact experimental field on the CPU side, so the final Marching Cubes vertex placement is less dependent on raw voxel-corner interpolation
- fall back inside that same scalar-field stage to the cheaper voxel-coverage-derived field when the scene is heavier
- keep the older CPU scalar-field extraction path as a safety fallback when the direct Vulkan surface path is unavailable or produces no usable triangles
- keep the shared Vulkan runtime/device/queue/command pool alive across repeated generations, along with the shared compute pipelines, layouts, and persistent host-visible working buffers, so the experimental backend avoids rebuilding the full context every time
- fall back to the selected stable Frost CPU mesher only if both experimental Vulkan extraction paths are unavailable or empty

The selected `meshing_method` is preserved in this path, so the experimental Vulkan backend can currently plan and extract around:

- `Union of Spheres`
- `Metaball`
- `Zhu-Bridson`
- `Anisotropic`

This means the addon now has a real functional Vulkan meshing path inside `frost_native.dll`: field evaluation, surface-cell classification, and first-pass Marching Cubes triangle emission now all happen on the GPU, with CPU packaging and fallback still retained for robustness.

### Why This Matters for Blender 5.1+

Blender `5.1` ships with a newer embedded Python than Blender `5.0.x`.

The stable `ctypes -> frost_native.dll` path avoids coupling the addon runtime to a specific Python extension ABI, which makes cross-version Blender support much more robust than the old `.pyd`-only loading model.

---

## Current GPU Pipeline

The legacy production GPU implementation is still CUDA-based, but it now sits behind a more generic backend layer.

The current GPU path no longer uses the original simple blob-density approach as the final surface model.

### 1. Particle Upload

The GPU now receives:

- particle positions
- particle radii

This is required for a Zhu-Bridson-style field because the field depends on a blended radius term, not just density.

### 2. Neighbor Search

`cuNSearch` is used to maintain the GPU-oriented neighborhood stage and keep the overall pipeline structured around local particle influence.

### 3. Zhu-Bridson Grid Accumulation

The grid accumulation stage stores:

- total weight
- blended radii accumulator
- blended offset accumulator

Conceptually, each voxel gathers weighted particle influence inside the search radius.

### 4. Field Finalization

Each voxel is converted into the scalar field using:

- blended offset magnitude
- blended radius term
- optional low-density compensation term

The GPU extraction now uses an isovalue of `0.0`.

### 5. Marching Cubes

Marching Cubes runs on the finalized field and emits raw triangle vertices.

Important recent stability changes:

- more stable edge interpolation for nearly equal field values
- no premature tiny-triangle culling inside the CUDA kernel

This matters because deleting triangles too early can create open seams between neighboring cells.

### 6. CPU-Side Weld and Post-Process

After download, the C++ layer:

- welds vertices on a tolerance grid
- rejects only triangles that collapse to duplicate welded vertex indices
- applies shared post-processing such as push / relax

The previous near-zero-area rejection after weld was removed because it could also remove triangles that were still topologically necessary.

---

## GPU UI Mapping

Current exposed GPU controls:

- `meshing_resolution_mode`
- `meshing_resolution`
- `meshing_voxel_length`
- `gpu_search_radius_scale`

Current hidden / internal GPU controls:

- `gpu_block_size`
- `gpu_surface_refinement`
- GPU low-density trimming values

### Why Block Size Is Hidden

`gpu_block_size` only affects the 1D CUDA launch size used for field accumulation / finalization.

It does not meaningfully change the meshing result itself.

Also note:

- the Marching Cubes kernel uses a fixed `8 x 8 x 8` block configuration
- the normal workflow now hardcodes the internal block size to `256`

---

## CPU Path

The CPU path still relies on the upstream Thinkbox Frost algorithms, including:

- Union of Spheres
- Metaball
- Zhu-Bridson
- Anisotropic

This remains the feature-complete path and the visual quality reference for future GPU work.

---

## Build Notes

Typical Windows native build:

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" && cmake -S K:\Codex_Projects\Projet_Frost_Plugin_For_Blender_CODEX\blender_frost_adapter -B K:\Codex_Projects\Projet_Frost_Plugin_For_Blender_CODEX\build_native -DCMAKE_TOOLCHAIN_FILE=K:\Codex_Projects\Projet_Frost_Plugin_For_Blender_CODEX\deps\vcpkg\scripts\buildsystems\vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows -DFROST_BUILD_PYTHON_MODULE=OFF -DFROST_ENABLE_CUDA=OFF -A x64 && cmake --build K:\Codex_Projects\Projet_Frost_Plugin_For_Blender_CODEX\build_native --config Release --target frost_native --parallel 8'
```

After a successful build, copy the generated `frost_native.dll` into `frost_blender_addon/`.

Blender must be restarted to load the updated native module.

---

## Validation Notes

Recent regression checks used:

- headless Blender tests
- Suzanne source meshes
- subdivided Suzanne source meshes
- topology checks for boundary edges and degenerate faces

The current CUDA seam fix was validated by reducing boundary-edge counts to `0` on these regression meshes.

---

## Known Limitations

- GPU path is still less feature-complete than the Thinkbox CPU path.
- `MESH` sources do not yet use evaluated modifier-stack geometry automatically.
- the current `vulkan-experimental` path is not a performance backend yet; it now performs a first full GPU field/classification/triangle-emission pass, but still keeps CPU-side packaging, refinement, and fallback logic for robustness
- the legacy CUDA meshing path remains the more advanced true-GPU meshing implementation in the source tree

---

## Next Good Targets

- evaluated mesh extraction for `MESH` sources
- further visual convergence between the GPU Zhu-Bridson path and Thinkbox CPU Zhu-Bridson
- stronger automated regression tests around topology and shading artifacts
- shader-based Vulkan field evaluation and meshing inside the stable `frost_native.dll` architecture

---

Document version: `1.6`
Last update: `2026-03-26`
