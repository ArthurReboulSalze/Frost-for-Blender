# Frost for Blender Addon Package

Current addon version: `1.25.0`  
Supported Blender version: `5+`

## Included

- Thinkbox Frost CPU meshing path
- Current Vulkan GPU backend work
- Blender Python addon files
- `frost_native.dll` and bundled runtime dependencies

## Notes

- The current public package is a single `CPU + Vulkan` addon package.
- The Vulkan backend is still in active development.
- Current testing indicates that Vulkan is usually slower than CPU on low-poly scenes, while heavier / high-poly scenes can already favor Vulkan depending on the case.
- `Vertex Refinement` still forces the final surface build back to CPU.
- Restart Blender after replacing the addon files or updating `frost_native.dll`.

## Credits

- Arthur Reboul Salze
- AWS Thinkbox Frost core
- Codex
