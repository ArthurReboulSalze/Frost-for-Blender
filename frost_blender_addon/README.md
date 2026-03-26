# Frost Particle Meshing Addon

Blender addon package for Frost Particle Meshing.

Current addon version: `1.24.1`
Supported Blender version: `5+`

## Features

- CPU meshing via Thinkbox Frost
- Experimental GPU backend support while keeping the Frost CPU meshing methods
- Particle systems, point clouds, and mesh vertex sources
- Auto Update workflow
- Bake to Alembic
- Push / smoothing post-processing

## Notes

- The normal GPU UI now exposes only the controls that materially affect the result.
- `Block Size`, `Low Density Trimming`, and `Surface Refinement` are not part of the normal GPU workflow anymore.
- For `MESH` sources, Frost currently reads raw mesh data, not evaluated modifier-stack geometry.

## Installation

1. Open Blender `5+`
2. Install the release zip from GitHub Releases, or symlink the addon folder into Blender's addon directory
3. Enable `Frost Particle Meshing`

## Native Module

The addon now prefers `frost_native.dll`, loaded through a stable Python `ctypes` bridge.

The legacy `blender_frost_adapter.pyd` path is still supported as a fallback, but it depends on Blender's embedded Python ABI and is therefore more sensitive to Blender version changes.

If the native module is rebuilt, restart Blender so the updated binary is actually loaded.

The native layer also includes a runtime-only Vulkan probe, so the addon can detect whether a system Vulkan runtime is present without requiring a Vulkan SDK in the build environment.

It now also includes a Vulkan compute probe that checks whether a minimal logical device and compute queue can be created on the current machine.

It also includes a Vulkan storage-buffer probe that checks whether the addon can already allocate and map a first host-visible storage buffer through Vulkan.

The current native build can also expose an initial `vulkan-experimental` backend that runs real Vulkan compute shaders on particle buffers, writes per-particle voxel influence bounds for the selected Frost method, computes velocity-aware anisotropic voxel bounds per particle, populates an integer voxel coverage buffer, compacts active voxels before scalar-field dispatch, builds compact `active voxel -> candidate particle` tables for the scalar-field and exact-refinement passes, derives method-aware dense scalar fields instead of one generic approximation, skips particles whose voxel bounds cannot affect the current voxel, classifies Marching Cubes surface cells on the GPU, emits Marching Cubes triangle vertices directly in Vulkan, rebuilds a welded mesh from those GPU triangles, includes dedicated `Metaball` and `Plain Marching Cubes` field modes plus a first dedicated `Zhu-Bridson` blended approximation that now follows the Frost CPU weighting model more closely, and a first velocity-aware `Anisotropic` approximation, clamps `Union of Spheres` more like the Frost CPU policy, falls back to a cheaper coverage-derived field on heavier scenes, keeps the shared Vulkan runtime/device/queue/command pool alive across repeated generations, reuses the shared compute pipelines and layouts, retains the main host-visible working buffers between updates, and only falls back to the older CPU extraction paths when the direct experimental Vulkan surface path is unavailable.

## Distribution Model

The current public package follows the single-addon direction:

- CPU meshing always included
- GPU acceleration included when the native backend in the package supports it
- no separate user-facing CPU-only and GPU-only addon variants

This keeps installation simpler and makes future backend changes, including a potential Vulkan path, easier to distribute.

## Credits

- Arthur Reboul Salze
- AWS Thinkbox Frost core
- Codex
