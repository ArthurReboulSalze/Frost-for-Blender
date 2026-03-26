# Frost Particle Meshing - Changelog

## Current Release

Current addon version: `1.24.1`
Last updated: `2026-03-26`

Older entries below keep some of the original internal milestone numbering from development logs, but the addon version shown in Blender now follows the `1.x.y` release line.

---

## v1.24.1 - GPU Mesh Stability and Unified Release Package (2026-03-26)

### Added / Improved

- Prepared a single Blender-installable addon package for the current `CPU + Vulkan` workflow.
- Updated repository references for the simplified GitHub project name: `Frost-for-Blender`.
- Reduced unnecessary host work on the Vulkan direct path by skipping output particle readback when the raw GPU mesh path does not need it.

### Fixed

- Disabled direct GPU surface extraction whenever Frost `Vertex Refinement` is enabled, so those cases now intentionally fall back to the stable Frost CPU meshing path.
- Fixed a dense Vulkan direct-surface assembly issue where invalid leftover faces could survive the face-buffer build and produce holes, spikes, or stretched triangles in GPU mode.
- Revalidated the rebuilt native path under Blender `5.1` after the latest GPU meshing fixes.

### Notes

- The Vulkan backend is still experimental and remains slower than the Thinkbox CPU mesher on dense scenes.
- The current public release should now be distributed as one Blender addon zip for Blender `5+`.

---

## v1.24.0 - Blender 5.1 Support and Experimental Vulkan Backend (2026-03-26)

### Added / Improved

- Added a stable `ctypes -> frost_native.dll` native bridge so the addon is no longer tied only to Blender's embedded Python extension ABI.
- Verified addon loading and native meshing under Blender `5.1`.
- Introduced a GPU backend abstraction layer so the addon can expose different native GPU implementations behind one package.
- Added Vulkan runtime, compute-context, and storage-buffer probing in the native layer.
- Added an initial `vulkan-experimental` backend that dispatches real Vulkan compute shaders on particle buffers.
- The Vulkan path now preserves the selected Frost CPU meshing method for final surface extraction:
  - `Union of Spheres`
  - `Metaball`
  - `Zhu-Bridson`
  - `Anisotropic`
- Added per-particle voxel influence planning in the Vulkan compute stage.
- Added a first integer voxel coverage buffer on the planned domain as a preparatory step toward real field evaluation on Vulkan.
- Added a first dense scalar field stage on Vulkan that now branches by meshing method instead of applying one generic approximation everywhere, with automatic fallback to the cheaper coverage-derived field on heavier scenes.
- Added a dedicated experimental `Metaball` threshold field mode on Vulkan, so `Metaball` no longer reuses the same generic signed-distance field as the other approximated methods.
- Added a dedicated experimental `Plain Marching Cubes` density field mode on Vulkan, so its `plain_marching_cubes_isovalue` now affects the experimental surface path too.
- Added a first dedicated experimental `Zhu-Bridson` blended field approximation on Vulkan instead of routing `Zhu-Bridson` through the same simple sphere signed-distance field as the default fallback.
- Tightened that experimental `Zhu-Bridson` field to follow the Frost CPU policy more closely by using a cubic compact-support kernel, the raw particle radius for blended-radius accumulation, and a uniform kernel support radius derived from the maximum particle radius during voxel planning.
- Added a first dedicated experimental `Anisotropic` velocity-aware field mode on Vulkan, using per-particle velocity to stretch the field along motion direction instead of falling back to the generic isotropic sphere path.
- Added velocity-aware anisotropic voxel bound planning in the first Vulkan dispatch, so the planned domain now follows the stretched particle support per particle instead of expanding everything isotropically.
- Added voxel-bound pruning in the Vulkan scalar-field stage and in the CPU-side exact edge refinement path, so particles outside the current voxel bounds are skipped instead of being re-evaluated everywhere.
- Added active-voxel compaction before Vulkan scalar-field dispatch, so the dense field buffer is now prefilled once on the CPU and the compute shader only runs on covered voxels instead of scanning the whole planned domain.
- Added compact `active voxel -> candidate particle` tables to the Vulkan path, so the scalar-field shader and exact edge refinement can iterate only the particles that overlap each active voxel instead of rescanning the full particle set every time.
- Reworked that compact `active voxel -> candidate particle` build to walk sparse active voxel rows instead of scanning every voxel inside each particle AABB, so larger sparse domains avoid a lot of unnecessary CPU-side setup before the scalar-field dispatch.
- Added cached exact corner samples during experimental Vulkan Marching Cubes refinement, and now gate that refinement from the compact voxel-particle budget instead of using only a crude raw particle-count cutoff.
- Added compact per-edge particle candidate reuse during experimental Vulkan edge refinement, now built by directly merging neighboring voxel candidate ranges and reused for both cached edge endpoints and midpoint bisection instead of rebuilding broader searches for every exact sample on the same edge.
- Added candidate-cell compaction for the experimental CPU Marching Cubes extraction stage, so surface extraction now skips large empty regions and iterates only cells adjacent to active voxel samples when coverage data is available.
- Refined that candidate-cell compaction to prioritize negative scalar-field samples first, then fall back to coverage samples only when needed, so experimental extraction focuses more tightly on actual isosurface neighborhoods.
- Replaced the remaining full-grid negative-sample scan in experimental extraction with the compact active-voxel list produced by the Vulkan path, so Marching Cubes candidate discovery now stays on the compact representation end-to-end.
- Added a dedicated Vulkan compute stage that classifies candidate Marching Cubes cells and returns active cell `cubeIndex` values, so the experimental extraction path now starts from GPU-classified surface cells instead of recomputing every cell state on the CPU.
- Tightened the experimental `Union of Spheres` field on Vulkan so it behaves closer to the Frost CPU policy instead of using the broader planning radius directly as the field radius.
- Fixed an experimental Vulkan `Metaball` artifact source where voxels inside particle AABBs but outside the true support could still pick an overly aggressive coverage fallback and grow unwanted blobs.
- Added exact field-based edge refinement during experimental Vulkan Marching Cubes extraction, so local surface bumps and overextended edge vertices are reduced without reintroducing the old global surface-refinement option.
- Confirmed that the compact Vulkan path now stays quiet when `show_debug_log` is disabled during normal generation.
- Added an experimental Marching Cubes extraction stage that can now build a first mesh directly from Vulkan-generated field data before falling back to the stable Frost CPU mesher.
- Added a dedicated Vulkan compute stage that now emits Marching Cubes triangle vertices directly from GPU-classified surface cells, so the experimental backend can reach a first full GPU meshing path before Blender-side mesh packaging.
- Added reconstruction of a welded `trimesh3` from those Vulkan-emitted triangles, while keeping the older CPU scalar-field extraction as a fallback path.
- Replaced the first direct-GPU weld pass based on raw float vertex hashing with shared Marching Cubes edge keys, so the CPU-side packaging of Vulkan triangles is lighter and more stable topologically.
- Added a direct raw-buffer return path for the Vulkan backend when no CPU post-process is requested, so simple GPU generations can bypass `trimesh3` entirely on the native side.
- The direct Vulkan raw-buffer path now also skips the CPU-side compact `active voxel -> candidate particle` setup and lets the scalar-field shader walk per-particle voxel bounds directly on the GPU, removing another CPU bottleneck from the hot path.
- Added a first dense full-cell Vulkan surface-mesh compute path for tiny domains, while keeping the sparse direct path as the default for larger domains where it is currently faster.
- Added a first sparse GPU candidate-cell classification path driven from active voxels, and kept it behind a very conservative activation gate until it consistently outperforms the current hybrid candidate preparation path.
- Cached the shared Vulkan runtime/device/queue/command-pool setup between generations so repeated updates no longer recreate the full context every time.
- Cached the shared Vulkan compute pipelines and layouts as well, so repeated updates avoid rebuilding the shader pipeline stack on every generation.
- Reused the main Vulkan host-visible working buffers between generations so repeated updates no longer recreate and remap the core particle and field buffers every time.
- Reused shared Vulkan descriptor pools, descriptor sets, and command buffers for the hot particle-planning, coverage, and scalar-field dispatches, so repeated updates no longer rebuild those per-pass dispatch resources every generation.
- Added a dedicated GPU active-voxel compaction pass after voxel coverage, so the hot path now reads back a compact active-voxel list and coverage stats instead of rescanning the entire dense coverage grid on the CPU.
- The direct Vulkan raw-buffer path now leaves the scalar field resident on the GPU and only reads it back lazily when a CPU fallback path truly needs dense scalar samples.
- Added a dedicated GPU surface-cell compaction pass after sparse surface classification, so the hot path no longer rescans a dense `cubeIndex` grid on the CPU just to recover active Marching Cubes cells.
- The sparse Vulkan direct-surface path now reuses those resident compact active-cell buffers directly for triangle emission, removing an extra CPU-to-GPU upload roundtrip before surface generation.
- The direct Vulkan triangle buffers now store compact `xyz` floats instead of padded `vec4` vertices, reducing triangle buffer bandwidth on the hot path.
- The direct Vulkan weld/index build now uses an adaptive strategy: packed-edge sorting on smaller meshes, and a packed-edge hash path on larger ones.
- The sparse direct Vulkan path now skips the intermediate `VulkanSurfaceMeshResult` triangle copy and rebuilds output buffers directly from resident mapped surface buffers after GPU triangle emission.

### Fixed

- Removed the main compatibility blocker with newer Blender versions caused by the old Python ABI-bound native path.
- Clarified the UI so the experimental Vulkan path no longer implies a Zhu-Bridson-only GPU algorithm when the final meshing method still comes from Frost CPU settings.
- Fixed the Vulkan surface-cell classification corner ordering so GPU `cubeIndex` values now match CPU validation again before direct GPU triangle emission.

### UI / Workflow

- Moved the project toward a single addon distribution with CPU always available and one optional GPU backend path.
- Updated documentation to reflect Blender `5+`, the native bridge, the experimental Vulkan path, and the local `logs/` versioning workflow.
- Kept `logs/` as a local-only technical archive and out of the GitHub source repository.

### Notes

- The experimental Vulkan backend is functional and now reaches a first full GPU meshing path for Marching Cubes-style extraction.
- CPU fallback is still kept for safety, exact refinement, and unsupported edge cases.
- `MESH` source extraction still samples `obj.data` directly, so non-applied mesh modifiers are not yet used automatically as Frost source geometry.

---

## v1.23.0 - CUDA Zhu-Bridson Stability and UI Cleanup (2026-03-22)

### Added / Improved

- Reworked the GPU scalar field so it behaves like a real Zhu-Bridson-style field instead of a simple blob density field.
- Uploads particle radii to the GPU and uses blended radius / blended offset terms during field evaluation.
- Uses an isovalue of `0.0` for the GPU Zhu-Bridson field.
- Improved Marching Cubes edge interpolation stability for near-equal scalar values.

### Fixed

- Fixed remaining open seams / unwelded-looking cracks in the CUDA mesh path by removing premature triangle culling in the CUDA kernel.
- Fixed additional topology loss in the post-weld stage by only rejecting triangles that collapse to duplicate indices.
- Verified closed topology on Suzanne-based regression tests, including subdivided source geometry.

### UI / Workflow

- Removed `Block Size` from the normal GPU UI.
- Disabled `Low Density Trimming` in the current exposed workflow.
- Disabled `Surface Refinement` in the current exposed workflow.
- Kept the internal GPU block size at `256` as an implementation detail.

### Notes

- `MESH` source extraction still samples `obj.data` directly, so non-applied mesh modifiers are not yet used automatically as Frost source geometry.

---

## Historical Milestones

### v22.0 - Debug Log Toggle and Alembic Export Fix (2026-03-03)

- Added `Show Debug Log` toggle.
- Reworked Bake to Alembic to export frame by frame.
- Fixed build script handling for version-suffixed `.pyd` outputs.

### v20.6 - GPU Grid Alignment and Stability (2026-01-07)

- Fixed GPU mesh shifts caused by grid alignment changes.
- Cleaned up CPU / GPU separation in the panel.

### v20.4 - Marching Cubes Table Correction (2026-01-06)

- Replaced corrupted Marching Cubes lookup tables with verified reference tables.
- Resolved major holes and malformed triangle output caused by table corruption.

### v19.x - Initial GPU Marching Cubes Bring-Up

- First usable CUDA meshing path.
- Established FrostGPUManager and native GPU bridge flow.

### v17.x - CPU TBB and Data Pipeline Optimization

- Restored broad TBB parallelism.
- Reduced Python-side mesh update overhead with `foreach_set`.

### v15.x - Alembic Export

- Added Bake to Alembic workflow.

### v13.x - Multi-Source Support

- Added multiple additional Frost source objects.

### v12.x - Additional CPU Methods

- Added Metaball and Anisotropic CPU modes.

### v11.x - Zhu-Bridson CPU

- Exposed the Thinkbox Zhu-Bridson path through the Blender addon.

### v10.x - First Functional Meshing in Blender

- First working Frost mesh generation from Blender-side particle data.

---

## Next Likely Improvements

- Support evaluated mesh extraction for `MESH` sources so Subdivision Surface modifiers can be used without applying them.
- Continue reducing the quality gap between CUDA Zhu-Bridson and the CPU Thinkbox path.
- Add better GPU-focused regression tests around topology and surface quality.

---

## Credits

- Arthur Reboul Salze
- AWS Thinkbox Frost core
- Codex
