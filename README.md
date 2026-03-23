# Frost Particle Meshing for Blender

Frost Particle Meshing is a Blender addon that converts particles, point clouds, and mesh vertices into polygonal surfaces.

Current release: `1.23.0`
Target Blender version: `5+`

<p align="center">
  <img src="Frost4Blender_screen.png" alt="Screenshot of the Frost Particle Meshing plugin in Blender" width="900">
</p>

<p align="center">
  Screenshot of the Frost Particle Meshing plugin running inside Blender.
</p>

The project combines two meshing backends:

- `CPU / Thinkbox Frost`: the original production-proven Frost meshing core.
- `GPU / CUDA`: a custom Zhu-Bridson GPU path with Marching Cubes extraction.

## Current State

- CPU meshing is stable and fully functional.
- GPU meshing now uses a true Zhu-Bridson-style scalar field built from particle positions and radii.
- Recent CUDA work focused on mesh quality, closed topology, and UI cleanup rather than adding more exposed settings.

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
- GPU meshing mode:
  - Zhu-Bridson GPU
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
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" && cmake --build K:\Codex_Projects\Projet_Frost_Plugin_For_Blender_CODEX\build_codex --config Release --parallel 8'
```

After building, copy the generated `.pyd` into `frost_blender_addon/` so Blender picks up the updated native module on next launch.

## Repository Layout

- `frost_blender_addon/`: Blender addon Python files and bundled runtime dependencies
- `blender_frost_adapter/`: Pybind11 bridge and native interface
- `blender_frost_adapter/src/cuda/`: CUDA implementation
- `thinkbox-frost/`: upstream Thinkbox Frost sources
- `documentations/`: user and technical docs

## Credits

- Arthur Reboul Salze
- AWS Thinkbox Frost core
- Codex
- Antigravity project generation / iteration workflow
