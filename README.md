# Frost Particle Meshing for Blender

Frost Particle Meshing is a Blender addon that converts particles, point clouds, and mesh vertices into polygonal surfaces.

Current release: `1.24.1`
Target Blender version: `5+`

Repository: [ArthurReboulSalze/Frost-for-Blender](https://github.com/ArthurReboulSalze/Frost-for-Blender)

<p align="center">
  <img src="Frost4Blender_screen.png" alt="Screenshot of the Frost Particle Meshing plugin in Blender" width="900">
</p>

<p align="center">
  Screenshot of the Frost Particle Meshing plugin running inside Blender.
</p>

The project combines two meshing backends:

- `CPU / Thinkbox Frost`: the original production-proven Frost meshing core.
- `GPU backend`: a backend abstraction that can now expose either the legacy CUDA path or an experimental Vulkan path from the same addon package.

## Current State

- CPU meshing is stable and fully functional.
- The addon now supports Blender `5.1` through a version-agnostic native bridge loaded with `ctypes`.
- Recent GPU work focused on mesh quality, closed topology, UI cleanup, reducing coupling to Blender's embedded Python version, and opening the way to Vulkan.
- A runtime-only Vulkan probe now works without requiring a Vulkan SDK in the build environment.
- A Vulkan compute probe now also verifies whether a minimal logical device and compute queue can be created on the current machine.
- A Vulkan storage-buffer probe now verifies that host-visible storage buffer allocation and mapping already work on the current machine.
- The current native build can now expose an initial `vulkan-experimental` backend that evaluates scalar fields, classifies Marching Cubes cells, and emits surface triangles directly in Vulkan, with CPU fallback only when the direct GPU path is unavailable.
- The current public release is now prepared as one Blender-installable addon package for the `CPU + Vulkan` workflow.

## Key Features

- Multiple particle sources:
  - Blender particle systems
  - Geometry Nodes point clouds
  - Mesh vertices
- Multiple CPU meshing modes:
  - Union of Spheres
  - Metaball
  - Zhu-Bridson
  - Anisotropic
- Experimental GPU acceleration path:
  - Legacy CUDA backend when explicitly built
  - Vulkan compute preprocessing before the selected Frost CPU meshing method
- Post-processing:
  - Push Distance
  - Geometric Flow passes
  - Smooth Shading
- Animation support:
  - Auto Update
  - Bake to Alembic

## GPU Notes

The current GPU path is intentionally simplified in the UI. The exposed controls are:

- `Resolution Mode`
- `Subdivisions` or `Voxel Length`
- `Search Radius Scale`

When the active backend is `vulkan-experimental`, the actual surface algorithm still comes from the regular Frost method selector:

- `Union of Spheres`
- `Metaball`
- `Zhu-Bridson`
- `Anisotropic`

The following options are intentionally not exposed anymore in the normal GPU workflow:

- `Block Size`
- `Low Density Trimming`
- `Surface Refinement`

These either had negligible user-facing impact or were producing unstable results in practice.

## Important Limitation

For `MESH` sources, the addon currently samples `obj.data` directly. That means a non-applied `Subdivision Surface` modifier is not yet used as a Frost source mesh automatically.

If you want Frost to follow a subdivided mesh source today, use one of these approaches:

1. Apply the modifier.
2. Use real subdivided geometry.

## Documentation

Project documentation lives in [documentations/README.md](documentations/README.md).

Main documents:

- [User Guide](documentations/USER_GUIDE.md)
- [Technical Reference](documentations/TECHNICAL_REFERENCE.md)
- [Changelog](documentations/CHANGELOG.md)

## Build

Typical native build flow on Windows:

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" && cmake -S K:\Codex_Projects\Projet_Frost_Plugin_For_Blender_CODEX\blender_frost_adapter -B K:\Codex_Projects\Projet_Frost_Plugin_For_Blender_CODEX\build_native -DCMAKE_TOOLCHAIN_FILE=K:\Codex_Projects\Projet_Frost_Plugin_For_Blender_CODEX\deps\vcpkg\scripts\buildsystems\vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows -DFROST_BUILD_PYTHON_MODULE=OFF -DFROST_ENABLE_CUDA=OFF -A x64 && cmake --build K:\Codex_Projects\Projet_Frost_Plugin_For_Blender_CODEX\build_native --config Release --target frost_native --parallel 8'
```

After building, copy the generated `frost_native.dll` into `frost_blender_addon/` so Blender picks up the updated native bridge on next launch.

## Repository Layout

- `frost_blender_addon/`: Blender addon Python files and bundled runtime dependencies
- `blender_frost_adapter/`: native meshing bridge, C API, and legacy pybind wrapper
- `blender_frost_adapter/src/cuda/`: CUDA implementation
- `thinkbox-frost/`: upstream Thinkbox Frost sources
- `documentations/`: user and technical docs

## Distribution Direction

The project is moving toward a single addon distribution:

- CPU meshing always available
- one optional GPU backend inside the same addon package
- no separate user-facing CPU-only / GPU-only plugin variants

Current release packaging follows that direction: one Blender zip containing the Python addon, `frost_native.dll`, and the runtime dependencies needed for the current Blender `5+` CPU + Vulkan path.

## Vulkan Direction

The native layer now includes a lightweight Vulkan runtime probe:

- no Vulkan SDK is required to build this probe
- the probe dynamically loads the system Vulkan runtime
- the addon can report whether a Vulkan runtime and physical device are available
- the addon can also report whether a minimal Vulkan compute context is ready
- the addon can also report whether a first storage-buffer allocation path is already usable

The project now has an initial `vulkan-experimental` backend:

- particle data is uploaded into Vulkan-managed buffers
- a real Vulkan command buffer dispatches a first compute shader over those particle buffers
- the compute stage now also writes per-particle voxel influence bounds for the selected Frost method
- a second compute stage can now populate an integer voxel coverage buffer over that planned domain
- a third compute stage can now derive method-aware dense scalar fields instead of one generic approximation for every mode
- the same scalar-field stage now has a dedicated `Metaball` threshold field instead of treating `Metaball` like the generic signed-distance approximation
- that stage now also has a dedicated `Plain Marching Cubes` density field and a first dedicated `Zhu-Bridson` blended field approximation
- that `Zhu-Bridson` approximation now follows the Frost CPU policy more closely by using a cubic compact-support kernel, raw particle radii for the blended radius term, and a uniform support radius based on the maximum particle radius
- that stage now also has a first dedicated velocity-aware `Anisotropic` field approximation instead of routing `Anisotropic` through the generic isotropic sphere field
- the first Vulkan planning dispatch now also computes velocity-aware anisotropic voxel bounds per particle instead of inflating the whole domain isotropically
- the scalar-field stage now skips particles whose voxel bounds do not overlap the current voxel, and the CPU-side exact edge refinement follows the same pruning logic
- active voxels are now compacted before scalar-field dispatch, so the dense field buffer is prefilled once and the compute shader only runs on covered voxels
- active voxels now also build compact `voxel -> candidate particle` tables, so the scalar-field pass and exact edge refinement can iterate only relevant particles per active voxel
- that compact `voxel -> candidate particle` build now walks sparse active rows instead of scanning every voxel covered by each particle AABB, so larger sparse domains avoid a lot of wasted CPU-side setup work before dispatch
- the experimental extraction path now caches exact corner samples during refinement and only enables that refinement when the compact voxel-particle workload stays small enough
- the exact edge-refinement pass now builds compact per-edge candidate particle sets by directly merging neighboring voxel ranges, then reuses them for both edge endpoints and midpoint bisection, which improves local vertex placement without falling back to a full particle scan per exact sample
- the experimental CPU Marching Cubes extraction now also compacts candidate cells from active voxels, so extraction can skip large empty regions instead of iterating the whole scalar-field domain
- that extraction pass now prioritizes candidate cells around negative scalar-field samples first, with a coverage-based fallback only when needed
- the extraction stage now also reuses the compact active-voxel list from the Vulkan pipeline, so candidate discovery no longer rescans the full scalar-field grid
- a new Vulkan compute stage now classifies candidate Marching Cubes cells and returns active cell `cubeIndex` values, so the experimental extraction path already starts from GPU-classified surface cells instead of recomputing every cell state on the CPU
- `Union of Spheres` now uses a more Frost-like signed-distance clamp in the experimental Vulkan field path instead of reusing the broader planning radius directly
- density-based modes now avoid inventing matter in voxels that are only inside coarse AABB coverage and not inside the true particle support
- the experimental Marching Cubes extraction now refines edge vertices against the exact field to reduce local bumps and overextended protrusions
- that same scalar-field stage automatically falls back to the cheaper voxel-coverage field on heavier scenes
- a dedicated Vulkan compute stage now emits Marching Cubes triangle vertices directly from the GPU-classified surface cells
- the direct GPU surface path now rebuilds a welded mesh from those Vulkan-emitted triangles using shared Marching Cubes edge identities before falling back to the older CPU extraction path
- when no CPU post-process is requested, the Vulkan backend can now return raw vertex/index buffers directly and bypass `trimesh3` entirely on the native hot path
- that direct raw-buffer path now also skips the CPU-side compact `active voxel -> candidate particle` setup and lets the scalar-field shader walk per-particle voxel bounds directly on the GPU, which removes another CPU bottleneck from the hot path
- the Vulkan backend now also has a first dense full-cell surface-mesh compute path for very small domains, while larger domains still stay on the faster sparse direct path for now
- the Vulkan backend also includes a first sparse GPU candidate-cell classification path driven from active voxels, but it stays behind a very conservative gate until it consistently beats the current hybrid path
- the Vulkan surface-cell classification now uses the same Marching Cubes corner ordering as the CPU path, so GPU `cubeIndex` results match CPU validation again
- the Vulkan runtime/device/queue/command-pool setup is now cached between generations, the shared compute pipelines/layouts are reused as well, and the main host-visible working buffers are now retained between updates too
- the hot Vulkan planning passes now also reuse shared descriptor pools, descriptor sets, and command buffers for particle planning, coverage, and scalar-field dispatch instead of recreating them on every generation
- the voxel-coverage stage now compacts active voxels on the GPU before scalar-field work begins, so the CPU no longer has to rescan the whole dense coverage grid just to rebuild the active-voxel list
- the direct Vulkan raw-buffer path now keeps the dense scalar field resident on the GPU until a CPU fallback explicitly needs it, so the hot path no longer reads back the full scalar field by default
- the sparse Vulkan surface path now compacts dense surface-cell `cubeIndex` data on the GPU and reuses those resident compact active-cell buffers for triangle emission, removing another CPU scan-and-reupload step from direct GPU meshing
- the direct Vulkan triangle buffers now use compact `xyz` floats instead of padded `vec4` output, reducing triangle buffer bandwidth on the hot path
- the direct Vulkan weld/index stage now switches between packed-edge sorting on smaller meshes and a packed-edge hash path on larger ones to keep CPU packaging more proportional to mesh size
- the selected Frost meshing method still drives the field behavior, but the experimental Vulkan path now reaches direct GPU triangle emission before any CPU fallback is considered

This is still an experimental path, but it now performs field evaluation, surface-cell classification, and first-pass Marching Cubes triangle emission directly in Vulkan instead of stopping at preprocessing.

## Credits

- Arthur Reboul Salze
- AWS Thinkbox Frost core
- Codex
- Antigravity project generation / iteration workflow
