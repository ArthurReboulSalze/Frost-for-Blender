# Frost Particle Meshing - Documentation Index

This folder contains the maintained documentation for the current addon release.

Current addon version: `1.24.1`
Last update: `2026-03-26`

## Available Documents

### User Guide

[USER_GUIDE.md](USER_GUIDE.md)

For artists and day-to-day usage:

- installation
- panel overview
- CPU and GPU settings
- animation workflow
- troubleshooting

### Technical Reference

[TECHNICAL_REFERENCE.md](TECHNICAL_REFERENCE.md)

For development and maintenance:

- architecture
- native bridge
- CUDA pipeline
- parameter mapping
- build notes
- current limitations

### Changelog

[CHANGELOG.md](CHANGELOG.md)

Release history and recent technical changes.

### Versioning Workflow

[VERSIONING_WORKFLOW.md](VERSIONING_WORKFLOW.md)

How to use the root `logs/` folder to track internal build iterations and map them cleanly to public addon releases.

## Quick Start

1. Install the current release zip from GitHub Releases, or symlink `frost_blender_addon` into Blender's addon folder.
2. Enable the addon in Blender.
3. Create or select a Frost object.
4. Set a source object in the Frost panel.
5. Choose CPU or GPU meshing.

For GPU mode, the main controls are:

1. `Resolution Mode`
2. `Subdivisions` or `Voxel Length`
3. `Search Radius Scale`

## Notes

- The GPU UI no longer exposes `Block Size`, `Low Density Trimming`, or `Surface Refinement`.
- For `MESH` sources, Frost currently reads the raw mesh data and does not automatically evaluate a non-applied `Subdivision Surface` modifier.
- The current public release package is a single Blender-installable zip for the `CPU + Vulkan` workflow.
