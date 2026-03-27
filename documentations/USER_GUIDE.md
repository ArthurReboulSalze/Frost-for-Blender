# Frost for Blender - User Guide

Supported Blender version: `5+`  
Compatible addon release: `1.25.0+`

---

## Overview

Frost for Blender builds a mesh from:

- particle systems
- Geometry Nodes point clouds
- mesh vertices

It currently offers two execution paths:

- `CPU / Thinkbox Frost`: stable reference path
- `GPU / Vulkan`: in-progress GPU path inside the same addon package

---

## Installation

1. Download the current release zip from GitHub Releases.
2. Open Blender.
3. Go to `Edit > Preferences > Add-ons`.
4. Use `Install from Disk...`.
5. Select the release zip and enable `Frost Particle Meshing`.

If Blender shows `Core Load Error`, `frost_native.dll` or one of its runtime dependencies did not load correctly.

---

## Basic Workflow

1. Create or select an object that will become the Frost mesh.
2. Assign a `Main Source`.
3. Choose CPU or GPU mode.
4. Adjust resolution and method settings.
5. Leave `Auto Update` enabled if you want live refresh.

---

## Main Controls

### Top Bar

- `Auto Update`: rebuilds the mesh when settings change
- `GPU Acceleration`: enables the current GPU backend
- `Bake to Alembic`: exports the Frost mesh as an Alembic sequence

### Last Meshing Status

When GPU is enabled, the panel now shows the backend actually used for the last meshing pass.

Examples:

- `Vulkan Experimental`: the GPU path completed normally
- `CPU Fallback`: GPU was requested, but Frost switched back to the CPU for safety

This is useful because it avoids confusing a safe CPU fallback with a real GPU performance result.

### Particle Sources

- `Main Source`
- `Additional Sources`
- `Source`: auto-detects particle system, point cloud, or mesh vertices

---

## Meshing

### CPU Methods

- `Union of Spheres`
- `Metaball`
- `Zhu-Bridson`
- `Anisotropic`

### GPU Workflow

The current Vulkan workflow keeps the normal Frost meshing method selector active, but exposes a simpler set of GPU-facing controls:

- `Resolution Mode`
- `Subdivisions` or `Voxel Length`
- `Search Radius Scale`

### Resolution Guidance

- higher `Subdivisions` = smaller voxels = more detail
- smaller `Voxel Length` = more detail
- both also increase cost and the chance of surfacing edge-case GPU issues

### Vertex Refinement

`Vertex Refinement` is still CPU-only for the final surface build.

So today:

- `Vertex Refinement = 0` lets Frost stay on the direct GPU surface path
- `Vertex Refinement > 0` forces a CPU fallback for the final surface

This is intentional for now, because the current Vulkan path does not yet implement the same refinement stage safely enough for release use.

---

## Post Processing

- `Push Distance`
- `Smooth Shading`
- `Geometric Flow Passes`
- `Smoothing Strength`

These remain shared addon controls regardless of whether the source mesh came from the CPU or GPU path.

---

## Performance Notes

Current testing indicates:

- CPU is still usually faster on low-poly scenes
- Vulkan can already become faster on heavier / high-poly scenes depending on the setup

So the Vulkan backend is no longer just a tech demo, but it is also not yet a universal CPU replacement.

---

## Important Source Limitation

For `MESH` sources, Frost still reads raw mesh data directly.

That means a non-applied `Subdivision Surface` modifier is not automatically used as Frost source geometry yet.

If you want Frost to use the subdivided mesh today:

1. apply the modifier, or
2. use real subdivided geometry

---

## Troubleshooting

### GPU requested, but the UI shows CPU Fallback

That means Frost deliberately switched back to the CPU path for safety or compatibility.

Typical causes:

- `Vertex Refinement` is enabled
- the direct GPU surface path was rejected as unsafe
- the current scene exceeded a safety budget for the direct GPU mesh path

### Blender was updated and Frost no longer loads

Restart Blender after replacing the addon, and check that `frost_native.dll` and its bundled runtime `.dll` files are present.

### GPU artifacts still appear at very high resolution

This can still happen on edge cases while Vulkan meshing is under active development.

The current addon now tries to reject clearly unsafe direct GPU geometry before it reaches Blender, but the Vulkan path is still being refined.

---

## Animation

1. Set up the source animation.
2. Enable `Auto Update`.
3. Scrub the timeline.
4. Use `Bake to Alembic` when you want a cached output sequence.

---

Document version: `1.4`  
Last update: `2026-03-27`
