# Frost Particle Meshing - User Guide

## Overview

Frost Particle Meshing creates a mesh from:

- particle systems
- Geometry Nodes point clouds
- mesh vertices

It offers two execution paths:

- `CPU / Thinkbox Frost`: highest feature completeness and the original Frost algorithms.
- `GPU / Vulkan Experimental`: the current single-addon GPU path, while the normal Frost meshing method selector stays active.

Supported Blender version: `5+`
Compatible addon release: `1.24.1+`

---

## Installation

1. Install the current release zip from GitHub Releases, or symlink the `frost_blender_addon` folder into Blender's addon directory.
2. Open Blender.
3. Enable `Frost Particle Meshing` in Preferences > Add-ons.

If Blender shows `Core Load Error`, `frost_native.dll` or one of its bundled `.dll` dependencies failed to load.

---

## Basic Workflow

1. Create or select an object that will become the Frost mesh.
2. In the Frost panel, assign a `Main Source`.
3. Choose CPU or GPU mode.
4. Adjust the meshing resolution.
5. Enable `Auto Update` if you want live refresh while tweaking settings.

---

## Panel Overview

### Top Controls

- `Auto Update`: automatically rebuilds the Frost mesh when settings change.
- `GPU Acceleration`: switches to the current experimental GPU backend.
- `Bake to Alembic`: exports the animated Frost mesh as a frame sequence.

### Particle Sources

- `Main Source`: primary source object.
- `Additional Sources`: optional extra contributors.
- `Source`: auto-detects particle system, point cloud, or mesh vertices.

### Meshing

CPU mode exposes the Thinkbox meshing methods.

GPU mode currently exposes:

- `Resolution Mode`
- `Subdivisions` or `Voxel Length`
- `Search Radius Scale`

When the active backend is `vulkan-experimental`, the final surface algorithm still comes from the normal Frost method selector:

- `Union of Spheres`
- `Metaball`
- `Zhu-Bridson`
- `Anisotropic`

### Post Processing

- `Push Distance`
- `Smooth Shading`
- `Geometric Flow Passes`
- `Smoothing Strength` when passes are greater than `0`

---

## CPU Parameters

### Union of Spheres

Fast preview meshing.

Main control:

- `Radius Scale`

### Metaball

Smooth, blobby surface.

Main controls:

- `Radius Scale`
- `Isovalue`

### Zhu-Bridson

The original Thinkbox fluid-oriented mesher.

Main control currently exposed in the UI:

- `Blend Radius Scale`

### Anisotropic

Velocity-aware stretched surface.

Main controls:

- `Radius Scale`
- `Isovalue`
- `Max Anisotropy`
- `Min Neighbors`
- `Pos Smoothing`

---

## GPU Parameters

GPU mode is intentionally simpler than the CPU panel.

### Resolution Mode

Two choices:

- `Subdivide Max Radius`
- `Fixed Voxel Length`

### Subdivisions

Used when `Resolution Mode` is `Subdivide Max Radius`.

- higher value = smaller voxels
- smaller value = faster rebuilds

### Voxel Length

Used when `Resolution Mode` is `Fixed Voxel Length`.

- smaller value = more detail
- larger value = faster but softer result

### Search Radius Scale

Controls how far each particle influences the Zhu-Bridson field.

Typical range:

- `1.0 - 1.5`: tighter surface
- `1.5 - 2.0`: broader blending

### Vertex Refinement With GPU Mode

If `Vertex Refinement` is enabled, Frost now deliberately falls back to the CPU meshing path for the final surface build.

Reason:

- the current direct GPU surface path is not meant to combine with Frost CPU vertex-refinement iterations yet
- this fallback avoids the spikes or holes that could appear in earlier experimental GPU builds

### No Longer Exposed in GPU UI

These settings are intentionally not part of the normal GPU workflow anymore:

- `Block Size`
- `Low Density Trimming`
- `Surface Refinement`

Reason:

- `Block Size` was mostly an internal CUDA scheduling detail.
- `Low Density Trimming` was not useful in the current GPU workflow.
- `Surface Refinement` produced unstable results compared to the benefit.

---

## Important Source Note for Mesh Objects

If your source is a regular `MESH`, Frost currently reads the raw mesh data.

That means a non-applied `Subdivision Surface` modifier is not automatically used as Frost source geometry yet.

If you want Frost to use the subdivided version today:

1. apply the modifier, or
2. use actual subdivided geometry

---

## Post Processing

### Push Distance

Offsets the generated mesh along its normals.

- positive = inflate
- negative = deflate

### Geometric Flow Passes

Applies smoothing after meshing.

- `0` = disabled
- more passes = smoother surface

### Smooth Shading

Applies smooth shading to the generated Blender mesh.

---

## Animation Workflow

1. Set up your source animation.
2. Enable `Auto Update`.
3. Scrub the timeline.
4. Use `Bake to Alembic` when you want a cached exported sequence.

The Alembic workflow exports frame by frame so the Frost mesh is rebuilt correctly on each frame.

---

## Troubleshooting

### The mesh does not update

Check:

- a valid source object is assigned
- the source actually has particles / points / vertices
- the Frost object is visible in the viewport

### Core Load Error

Check:

- `frost_native.dll` exists in the addon folder
- required bundled `.dll` files are present
- Blender was restarted after updating the native module

### GPU cracks or unwelded-looking seams

This was addressed in `1.23.0`, improved again in `1.24.0`, and reinforced in `1.24.1`.

If you still see it:

1. restart Blender so the updated native module is actually loaded
2. retest on the same scene

### GPU mode does not seem to use my Subdivision Surface source

This is expected for now when the source type is `MESH` and the subdivision is only a modifier.

### GPU performance is poor

Try:

- increasing `Voxel Length`, or
- lowering `Subdivisions`, or
- reducing `Search Radius Scale`

---

## Version Info

Document version: `1.2`
Last update: `2026-03-26`
Compatible addon version: `1.24.1+`
