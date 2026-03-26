#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

enum class VulkanScalarFieldMode : uint32_t {
  sphere_signed_distance = 0,
  metaball = 1,
  plain_marching_cubes = 2,
  zhu_bridson_blend = 3,
  coverage_fallback = 4,
  anisotropic_velocity = 5,
};

struct VulkanParticleComputeSettings {
  float planningRadiusScale = 1.0f;
  float voxelLength = 1.0f;
  float fieldRadiusScale = 1.0f;
  float fieldThreshold = 0.0f;
  float surfaceIsoValue = 0.0f;
  float anisotropyMaxScale = 1.0f;
  float kernelSupportRadius = 0.0f;
  bool allowCpuCompactedPairs = true;
  bool preferGpuCompactedPairs = false;
  bool limitToBoundaryActiveVoxels = false;
  bool readbackCoverageCounts = false;
  bool readbackScalarField = true;
  bool retainParticleData = true;
  bool sortActiveVoxelIndices = true;
  VulkanScalarFieldMode fieldMode =
      VulkanScalarFieldMode::sphere_signed_distance;
};

struct VulkanParticleComputeResult {
  std::vector<float> particles;
  std::vector<float> velocities;
  std::vector<int32_t> minVoxelBounds;
  std::vector<int32_t> maxVoxelBoundsExclusive;
  std::vector<uint32_t> voxelCoverageCounts;
  std::vector<uint32_t> activeVoxelIndices;
  std::vector<int32_t> activeVoxelCompactLookup;
  std::vector<uint32_t> activeVoxelParticleOffsets;
  std::vector<uint32_t> activeVoxelParticleIndices;
  std::vector<float> voxelScalarField;
  bool scalarFieldResidentOnGpu = false;
  VulkanScalarFieldMode scalarFieldMode =
      VulkanScalarFieldMode::coverage_fallback;
  int32_t domainMinVoxel[3] = {0, 0, 0};
  int32_t domainDimensions[3] = {0, 0, 0};
  uint32_t activeVoxelCount = 0;
  uint32_t maxVoxelCoverage = 0;
  uint64_t coveredParticleVoxelPairs = 0;
  float voxelLength = 0.0f;
  float planningRadiusScale = 1.0f;
  float fieldRadiusScale = 1.0f;
  float fieldThreshold = 0.0f;
  float surfaceIsoValue = 0.0f;
  float anisotropyMaxScale = 1.0f;
  float kernelSupportRadius = 0.0f;
};

struct VulkanSurfaceMeshResult {
  std::vector<uint32_t> triangleCounts;
  std::vector<float> triangleVertices;
  uint32_t totalTriangleCount = 0;
};

struct VulkanResidentSurfaceMeshView {
  const uint32_t *triangleCounts = nullptr;
  const float *triangleVertices = nullptr;
  uint32_t activeCellCount = 0;
  bool triangleVerticesCompacted = false;
};

bool run_frost_vulkan_compute_particles(const float *inputParticles,
                                        const float *inputVelocities,
                                        std::size_t particleCount,
                                        const VulkanParticleComputeSettings &settings,
                                        VulkanParticleComputeResult &outResult,
                                        std::string &outError);

bool run_frost_vulkan_classify_surface_cells(
    const VulkanParticleComputeResult &computeResult,
    const std::vector<uint32_t> &candidateCellIndices,
    std::vector<uint32_t> &outActiveCellIndices,
    std::vector<uint32_t> &outActiveCellCubeIndices,
    std::string &outError);

bool run_frost_vulkan_classify_surface_cells_from_active_voxels(
    const VulkanParticleComputeResult &computeResult,
    std::vector<uint32_t> &outActiveCellIndices,
    std::vector<uint32_t> &outActiveCellCubeIndices,
    std::string &outError);

bool run_frost_vulkan_generate_sparse_surface_mesh_from_active_voxels(
    const VulkanParticleComputeResult &computeResult,
    std::vector<uint32_t> &outActiveCellIndices,
    std::vector<uint32_t> &outActiveCellCubeIndices,
    VulkanSurfaceMeshResult &outMesh, std::string &outError,
    bool copyMeshData = true);

bool run_frost_vulkan_generate_surface_mesh(
    const VulkanParticleComputeResult &computeResult,
    const std::vector<uint32_t> &activeCellIndices,
    const std::vector<uint32_t> &activeCellCubeIndices,
    VulkanSurfaceMeshResult &outMesh,
    std::string &outError);

bool run_frost_vulkan_generate_surface_mesh_from_resident_cells(
    const VulkanParticleComputeResult &computeResult,
    uint32_t activeCellCount, VulkanSurfaceMeshResult &outMesh,
    std::string &outError, bool copyMeshData = true);

bool run_frost_vulkan_generate_compact_surface_vertices_from_resident_cells(
    const VulkanParticleComputeResult &computeResult,
    uint32_t activeCellCount, const std::vector<uint32_t> &triangleOffsets,
    uint32_t totalTriangleCount, std::string &outError);

bool get_frost_vulkan_resident_surface_mesh_view(
    uint32_t activeCellCount, VulkanResidentSurfaceMeshView &outView,
    std::string &outError);

bool get_frost_vulkan_resident_compact_surface_vertex_view(
    uint32_t totalTriangleCount, const float *&outTriangleVertices,
    std::string &outError);

bool run_frost_vulkan_generate_dense_surface_mesh(
    const VulkanParticleComputeResult &computeResult,
    std::vector<uint32_t> &outActiveCellIndices,
    std::vector<uint32_t> &outActiveCellCubeIndices,
    VulkanSurfaceMeshResult &outMesh,
    std::string &outError);

bool ensure_frost_vulkan_scalar_field_readback(
    VulkanParticleComputeResult &ioResult, std::string &outError);
