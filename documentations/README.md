# Frost for Blender - Documentation Index

Current addon version: `1.25.0`  
Last update: `2026-03-27`

## Main Documents

### User Guide

[USER_GUIDE.md](USER_GUIDE.md)

For day-to-day usage:

- installation
- panel overview
- CPU and GPU workflow
- animation
- troubleshooting

### Technical Reference

[TECHNICAL_REFERENCE.md](TECHNICAL_REFERENCE.md)

For development and maintenance:

- addon architecture
- native bridge
- CPU / CUDA / Vulkan backend structure
- current limits
- build notes

### Changelog

[CHANGELOG.md](CHANGELOG.md)

Release history and notable technical changes.

## Quick Notes

- The public package is now a single Blender zip for the current `CPU + Vulkan` workflow.
- The Vulkan backend is still in active development.
- Current testing indicates that Vulkan is usually slower than CPU on low-poly scenes, while heavier / high-poly scenes can already favor Vulkan depending on the case.
- `Vertex Refinement` still forces the final surface build back to the CPU path.
- `MESH` sources still read raw mesh data, so non-applied mesh modifiers are not automatically evaluated yet.
