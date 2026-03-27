#include "vulkan_backend.hpp"

#include "vulkan_compute_shader.hpp"
#include "vulkan_buffer_probe.hpp"
#include "cuda/tables.h"

#include <boost/make_shared.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <frantic/logging/progress_logger.hpp>
#include <frantic/particles/streams/particle_array_particle_istream.hpp>
#include <frost/frost.hpp>
#include <frost/frost_parameter_interface.hpp>
#include <frantic/graphics/vector3f.hpp>

#if defined(FROST_ENABLE_VULKAN_PROBE) && defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#if defined(FROST_ENABLE_VULKAN_PROBE) && !defined(_WIN32)
#include <dlfcn.h>
#endif

namespace {

const char *get_meshing_method_name(int meshingMethod) {
  switch (meshingMethod) {
  case meshing_method::union_of_spheres:
    return "Union of Spheres";
  case meshing_method::metaballs:
    return "Metaball";
  case meshing_method::plain_marching_cubes:
    return "Plain Marching Cubes";
  case meshing_method::zhu_bridson:
    return "Zhu-Bridson";
  case meshing_method::anisotropic:
    return "Anisotropic";
  default:
    return "Unknown";
  }
}

const char *get_vulkan_scalar_field_mode_name(VulkanScalarFieldMode mode) {
  switch (mode) {
  case VulkanScalarFieldMode::sphere_signed_distance:
    return "particle-signed-distance";
  case VulkanScalarFieldMode::metaball:
    return "metaball-threshold";
  case VulkanScalarFieldMode::plain_marching_cubes:
    return "plain-marching-density";
  case VulkanScalarFieldMode::zhu_bridson_blend:
    return "zhu-bridson-blend";
  case VulkanScalarFieldMode::coverage_fallback:
    return "coverage-fallback";
  case VulkanScalarFieldMode::anisotropic_velocity:
    return "anisotropic-velocity";
  default:
    return "unknown";
  }
}

float compute_vulkan_meshing_voxel_length(const frost_parameters &params,
                                          float maximumParticleRadius,
                                          std::size_t particleCount) {
  if (params.get_meshing_resolution_mode() ==
      meshing_resolution_mode::subdivide_max_radius) {
    const float meshingResolution = params.get_meshing_resolution();
    if (meshingResolution <= 0.0f) {
      return 0.0f;
    }
    if (particleCount == 0 || maximumParticleRadius <= 0.0f) {
      return 1.0f;
    }
    return maximumParticleRadius / meshingResolution;
  }

  return params.get_meshing_voxel_length();
}

float compute_vulkan_effect_radius_scale(const frost_parameters &params,
                                         float voxelLength,
                                         float minimumParticleRadius) {
  switch (params.get_meshing_method()) {
  case meshing_method::union_of_spheres:
    if (minimumParticleRadius <= 0.0f) {
      return 2.0f;
    }
    return std::min(2.0f, 1.0f + 2.0f * voxelLength / minimumParticleRadius);
  case meshing_method::metaballs:
    return params.get_metaball_radius_scale();
  case meshing_method::plain_marching_cubes:
    return params.get_plain_marching_cubes_radius_scale();
  case meshing_method::zhu_bridson:
    return params.get_zhu_bridson_blend_radius_scale();
  case meshing_method::anisotropic:
    return params.get_anisotropic_radius_scale();
  default:
    return 1.0f;
  }
}

VulkanParticleComputeSettings build_vulkan_compute_settings(
    const frost_parameters &params, float planningRadiusScale,
    float voxelLength, float maximumParticleRadius) {
  VulkanParticleComputeSettings settings;
  settings.planningRadiusScale = planningRadiusScale;
  settings.voxelLength = voxelLength;
  settings.fieldRadiusScale = planningRadiusScale;
  settings.fieldThreshold = 0.0f;
  settings.surfaceIsoValue = 0.0f;
  settings.kernelSupportRadius = 0.0f;
  settings.fieldMode = VulkanScalarFieldMode::sphere_signed_distance;

  switch (params.get_meshing_method()) {
  case meshing_method::union_of_spheres:
    settings.fieldMode = VulkanScalarFieldMode::sphere_signed_distance;
    settings.fieldRadiusScale = 1.0f;
    settings.fieldThreshold = 2.0f * voxelLength;
    settings.surfaceIsoValue = 0.0f;
    break;
  case meshing_method::metaballs:
    settings.fieldMode = VulkanScalarFieldMode::metaball;
    settings.fieldRadiusScale = params.get_metaball_radius_scale();
    settings.fieldThreshold = params.get_metaball_isosurface_level();
    settings.surfaceIsoValue = 0.0f;
    break;
  case meshing_method::plain_marching_cubes:
    settings.fieldMode = VulkanScalarFieldMode::plain_marching_cubes;
    settings.fieldRadiusScale = params.get_plain_marching_cubes_radius_scale();
    settings.fieldThreshold = params.get_plain_marching_cubes_isovalue();
    settings.surfaceIsoValue = 0.0f;
    break;
  case meshing_method::zhu_bridson:
    settings.fieldMode = VulkanScalarFieldMode::zhu_bridson_blend;
    settings.fieldRadiusScale = params.get_zhu_bridson_blend_radius_scale();
    settings.fieldThreshold = 0.0f;
    settings.surfaceIsoValue = 0.0f;
    settings.kernelSupportRadius =
        std::max(maximumParticleRadius * settings.fieldRadiusScale, 0.0f);
    break;
  case meshing_method::anisotropic:
    settings.fieldMode = VulkanScalarFieldMode::anisotropic_velocity;
    settings.fieldRadiusScale = params.get_anisotropic_radius_scale();
    settings.fieldThreshold = params.get_anisotropic_isosurface_level();
    settings.surfaceIsoValue = 0.0f;
    settings.anisotropyMaxScale =
        std::max(params.get_anisotropic_max_anisotropy(), 1.0f);
    break;
  default:
    break;
  }

  return settings;
}

bool reduce_vulkan_voxel_bounds(const VulkanParticleComputeResult &computeResult,
                                std::size_t particleCount, int32_t &outMinX,
                                int32_t &outMinY, int32_t &outMinZ,
                                int32_t &outMaxXExclusive,
                                int32_t &outMaxYExclusive,
                                int32_t &outMaxZExclusive) {
  if (computeResult.minVoxelBounds.size() != particleCount * 4 ||
      computeResult.maxVoxelBoundsExclusive.size() != particleCount * 4 ||
      particleCount == 0) {
    return false;
  }

  outMinX = std::numeric_limits<int32_t>::max();
  outMinY = std::numeric_limits<int32_t>::max();
  outMinZ = std::numeric_limits<int32_t>::max();
  outMaxXExclusive = std::numeric_limits<int32_t>::min();
  outMaxYExclusive = std::numeric_limits<int32_t>::min();
  outMaxZExclusive = std::numeric_limits<int32_t>::min();

  for (std::size_t i = 0; i < particleCount; ++i) {
    const std::size_t offset = i * 4;
    outMinX = std::min(outMinX, computeResult.minVoxelBounds[offset + 0]);
    outMinY = std::min(outMinY, computeResult.minVoxelBounds[offset + 1]);
    outMinZ = std::min(outMinZ, computeResult.minVoxelBounds[offset + 2]);
    outMaxXExclusive =
        std::max(outMaxXExclusive,
                 computeResult.maxVoxelBoundsExclusive[offset + 0]);
    outMaxYExclusive =
        std::max(outMaxYExclusive,
                 computeResult.maxVoxelBoundsExclusive[offset + 1]);
    outMaxZExclusive =
        std::max(outMaxZExclusive,
                 computeResult.maxVoxelBoundsExclusive[offset + 2]);
  }

  return true;
}

struct GridEdgeKey {
  int x = 0;
  int y = 0;
  int z = 0;
  int axis = 0;

  bool operator==(const GridEdgeKey &rhs) const {
    return x == rhs.x && y == rhs.y && z == rhs.z && axis == rhs.axis;
  }
};

struct GridEdgeKeyHasher {
  std::size_t operator()(const GridEdgeKey &key) const noexcept {
    std::size_t seed = 1469598103934665603ull;
    auto mix = [&seed](std::uint64_t value) {
      seed ^= value + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
    };
    mix(static_cast<std::uint64_t>(static_cast<std::int64_t>(key.x)));
    mix(static_cast<std::uint64_t>(static_cast<std::int64_t>(key.y)));
    mix(static_cast<std::uint64_t>(static_cast<std::int64_t>(key.z)));
    mix(static_cast<std::uint64_t>(key.axis));
    return seed;
  }
};

constexpr int kMarchingCubeCornerOffsets[8][3] = {
    {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
    {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1},
};

constexpr int kMarchingCubeEdgeCorners[12][2] = {
    {0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6},
    {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7},
};

constexpr uint64_t kMaxDenseVulkanSurfaceCells = 128ull;
constexpr uint64_t kMaxGpuSparseCandidateScanCells = 1048576ull;
constexpr uint64_t kMinGpuSparseCandidateVoxels = 8192ull;
constexpr uint64_t kDirectSurfaceDenseRemapMaxEdges = 16777216ull;
constexpr uint32_t kMaxResidentRawSurfaceCells = 100000u;
constexpr uint32_t kMaxResidentRawSurfaceTriangles = 200000u;
constexpr uint64_t kMaxResidentRawSurfaceWorkingSetBytes = 64ull * 1024ull *
                                                           1024ull;

bool exceeds_resident_raw_surface_budget(int domainDimX, int domainDimY,
                                         int domainDimZ,
                                         uint32_t activeCellCount,
                                         uint32_t totalTriangleCount) {
  if (activeCellCount > kMaxResidentRawSurfaceCells ||
      totalTriangleCount > kMaxResidentRawSurfaceTriangles) {
    return true;
  }

  if (domainDimX <= 0 || domainDimY <= 0 || domainDimZ <= 0) {
    return true;
  }

  const uint64_t xEdgeCount = static_cast<uint64_t>(domainDimX - 1) *
                              static_cast<uint64_t>(domainDimY) *
                              static_cast<uint64_t>(domainDimZ);
  const uint64_t yEdgeCount = static_cast<uint64_t>(domainDimX) *
                              static_cast<uint64_t>(domainDimY - 1) *
                              static_cast<uint64_t>(domainDimZ);
  const uint64_t zEdgeCount = static_cast<uint64_t>(domainDimX) *
                              static_cast<uint64_t>(domainDimY) *
                              static_cast<uint64_t>(domainDimZ - 1);
  const uint64_t totalEdgeCount = xEdgeCount + yEdgeCount + zEdgeCount;
  if (totalEdgeCount == 0ull) {
    return false;
  }

  const uint64_t compactTriangleVertexBytes =
      static_cast<uint64_t>(totalTriangleCount) * 9ull * sizeof(float);
  const uint64_t compactTriangleEdgeIdBytes =
      static_cast<uint64_t>(totalTriangleCount) * 3ull * sizeof(uint32_t);
  const uint64_t dedupVertexBytes =
      static_cast<uint64_t>(totalTriangleCount) * 9ull * sizeof(float);
  const uint64_t dedupIndexBytes =
      static_cast<uint64_t>(totalTriangleCount) * 3ull * sizeof(uint32_t);
  const uint64_t edgeMappingBytes = totalEdgeCount * sizeof(uint32_t);

  const uint64_t estimatedWorkingSetBytes =
      compactTriangleVertexBytes + compactTriangleEdgeIdBytes + dedupVertexBytes +
      dedupIndexBytes + edgeMappingBytes;

  return estimatedWorkingSetBytes > kMaxResidentRawSurfaceWorkingSetBytes;
}

bool validate_vulkan_surface_geometry(
    const std::vector<float> &vertices, const std::vector<int> &faces,
    VulkanScalarFieldMode fieldMode, std::string &outError) {
  outError.clear();

  if (vertices.empty() || faces.empty()) {
    outError = "Vulkan direct surface mesh is empty.";
    return false;
  }
  if ((vertices.size() % 3u) != 0u || (faces.size() % 3u) != 0u) {
    outError = "Vulkan direct surface mesh buffer sizes are invalid.";
    return false;
  }

  const std::size_t vertexCount = vertices.size() / 3u;
  const std::size_t faceCount = faces.size() / 3u;
  if (vertexCount == 0u || faceCount == 0u) {
    outError = "Vulkan direct surface mesh contains no geometry.";
    return false;
  }

  std::vector<float> edgeLengths;
  edgeLengths.reserve(faceCount * 3u);
  float maxEdgeLength = 0.0f;

  std::unordered_map<uint64_t, uint32_t> edgeUseCounts;
  edgeUseCounts.reserve(faceCount * 3u);
  auto makeEdgeKey = [](int a, int b) -> uint64_t {
    const uint32_t lo = static_cast<uint32_t>(std::min(a, b));
    const uint32_t hi = static_cast<uint32_t>(std::max(a, b));
    return (static_cast<uint64_t>(lo) << 32) | static_cast<uint64_t>(hi);
  };

  for (std::size_t faceIndex = 0; faceIndex < faceCount; ++faceIndex) {
    const std::size_t base = faceIndex * 3u;
    const int i0 = faces[base + 0u];
    const int i1 = faces[base + 1u];
    const int i2 = faces[base + 2u];
    if (i0 < 0 || i1 < 0 || i2 < 0 ||
        static_cast<std::size_t>(i0) >= vertexCount ||
        static_cast<std::size_t>(i1) >= vertexCount ||
        static_cast<std::size_t>(i2) >= vertexCount) {
      outError = "Vulkan direct surface mesh contains out-of-range indices.";
      return false;
    }
    if (i0 == i1 || i1 == i2 || i0 == i2) {
      outError = "Vulkan direct surface mesh contains degenerate faces.";
      return false;
    }

    const float *v0 = vertices.data() + static_cast<std::size_t>(i0) * 3u;
    const float *v1 = vertices.data() + static_cast<std::size_t>(i1) * 3u;
    const float *v2 = vertices.data() + static_cast<std::size_t>(i2) * 3u;

    const auto measure_edge = [&](const float *a, const float *b) {
      const float dx = b[0] - a[0];
      const float dy = b[1] - a[1];
      const float dz = b[2] - a[2];
      return std::sqrt(dx * dx + dy * dy + dz * dz);
    };

    const float e01 = measure_edge(v0, v1);
    const float e12 = measure_edge(v1, v2);
    const float e20 = measure_edge(v2, v0);
    if (!std::isfinite(e01) || !std::isfinite(e12) || !std::isfinite(e20)) {
      outError = "Vulkan direct surface mesh contains non-finite edge lengths.";
      return false;
    }

    maxEdgeLength = std::max(maxEdgeLength, std::max(e01, std::max(e12, e20)));
    edgeLengths.push_back(e01);
    edgeLengths.push_back(e12);
    edgeLengths.push_back(e20);

    ++edgeUseCounts[makeEdgeKey(i0, i1)];
    ++edgeUseCounts[makeEdgeKey(i1, i2)];
    ++edgeUseCounts[makeEdgeKey(i2, i0)];
  }

  if (edgeLengths.empty()) {
    outError = "Vulkan direct surface mesh contains no measurable edges.";
    return false;
  }

  const std::size_t medianIndex = edgeLengths.size() / 2u;
  std::nth_element(edgeLengths.begin(), edgeLengths.begin() + medianIndex,
                   edgeLengths.end());
  const float medianEdgeLength =
      std::max(edgeLengths[medianIndex], 1.0e-6f);
  const float maxToMedianRatio = maxEdgeLength / medianEdgeLength;

  std::size_t boundaryOrNonManifoldEdgeCount = 0u;
  for (const auto &entry : edgeUseCounts) {
    if (entry.second != 2u) {
      ++boundaryOrNonManifoldEdgeCount;
    }
  }

  const float edgeRatioThreshold =
      fieldMode == VulkanScalarFieldMode::anisotropic_velocity ? 30.0f : 12.0f;
  const std::size_t boundaryThreshold =
      std::max<std::size_t>(32u, edgeUseCounts.size() / 2000u);

  if (maxToMedianRatio > edgeRatioThreshold) {
    std::ostringstream stream;
    stream << "Vulkan direct surface mesh produced pathological long edges "
              "(max/median ratio "
           << maxToMedianRatio << ").";
    outError = stream.str();
    return false;
  }

  if (boundaryOrNonManifoldEdgeCount > boundaryThreshold) {
    std::ostringstream stream;
    stream << "Vulkan direct surface mesh produced too many boundary or "
              "non-manifold edges ("
           << boundaryOrNonManifoldEdgeCount << ").";
    outError = stream.str();
    return false;
  }

  return true;
}

struct DirectSurfaceDenseWorkspace {
  std::vector<int> denseEdgeToVertex;
  std::vector<uint32_t> denseEdgeMarks;
  uint32_t denseEdgeGeneration = 1u;
  std::vector<uint32_t> validTriangleCountsPerCell;
  std::vector<uint64_t> usedEdgeIds;
  std::vector<float> usedEdgeVertices;
  std::vector<std::size_t> faceWriteOffsets;

  void prepare(std::size_t totalEdgeCount, std::size_t activeCellCount,
               uint32_t totalTriangleCount) {
    if (denseEdgeToVertex.size() < totalEdgeCount) {
      denseEdgeToVertex.resize(totalEdgeCount, -1);
    }
    if (denseEdgeMarks.size() < totalEdgeCount) {
      denseEdgeMarks.resize(totalEdgeCount, 0u);
      denseEdgeGeneration = 1u;
    } else {
      ++denseEdgeGeneration;
      if (denseEdgeGeneration == 0u) {
        std::fill(denseEdgeMarks.begin(), denseEdgeMarks.end(), 0u);
        denseEdgeGeneration = 1u;
      }
    }
    if (validTriangleCountsPerCell.size() < activeCellCount) {
      validTriangleCountsPerCell.resize(activeCellCount);
    }
    if (faceWriteOffsets.size() < activeCellCount + 1u) {
      faceWriteOffsets.resize(activeCellCount + 1u);
    }

    usedEdgeIds.clear();
    usedEdgeVertices.clear();
    usedEdgeIds.reserve(static_cast<std::size_t>(
        std::min<uint64_t>(static_cast<uint64_t>(totalTriangleCount) * 3ull,
                           static_cast<uint64_t>(totalEdgeCount))));
    usedEdgeVertices.reserve(static_cast<std::size_t>(totalTriangleCount) *
                             9u);
  }

  bool mark_edge(std::size_t edgeIndex) {
    if (denseEdgeMarks[edgeIndex] == denseEdgeGeneration) {
      return false;
    }
    denseEdgeMarks[edgeIndex] = denseEdgeGeneration;
    denseEdgeToVertex[edgeIndex] = -1;
    return true;
  }
};

DirectSurfaceDenseWorkspace &get_direct_surface_dense_workspace() {
  thread_local DirectSurfaceDenseWorkspace workspace;
  return workspace;
}

uint32_t count_marching_cubes_triangles(uint32_t cubeIndex) {
  if (cubeIndex == 0u || cubeIndex == 255u) {
    return 0u;
  }
  uint32_t triangleCount = 0u;
  for (int slot = 0; slot < 15; slot += 3) {
    if (triTable[cubeIndex][slot] < 0) {
      break;
    }
    ++triangleCount;
  }
  return triangleCount;
}

GridEdgeKey make_grid_edge_key(int cellX, int cellY, int cellZ, int edgeIndex) {
  switch (edgeIndex) {
  case 0:
    return {cellX, cellY, cellZ, 0};
  case 1:
    return {cellX + 1, cellY, cellZ, 1};
  case 2:
    return {cellX, cellY + 1, cellZ, 0};
  case 3:
    return {cellX, cellY, cellZ, 1};
  case 4:
    return {cellX, cellY, cellZ + 1, 0};
  case 5:
    return {cellX + 1, cellY, cellZ + 1, 1};
  case 6:
    return {cellX, cellY + 1, cellZ + 1, 0};
  case 7:
    return {cellX, cellY, cellZ + 1, 1};
  case 8:
    return {cellX, cellY, cellZ, 2};
  case 9:
    return {cellX + 1, cellY, cellZ, 2};
  case 10:
    return {cellX + 1, cellY + 1, cellZ, 2};
  case 11:
    return {cellX, cellY + 1, cellZ, 2};
  default:
    return {cellX, cellY, cellZ, 0};
  }
}

uint64_t make_packed_grid_edge_id(int domainDimX, int domainDimY, int domainDimZ,
                                  int cellX, int cellY, int cellZ,
                                  int edgeIndex) {
  const GridEdgeKey key = make_grid_edge_key(cellX, cellY, cellZ, edgeIndex);
  const uint64_t xEdgeCount =
      static_cast<uint64_t>(std::max(domainDimX - 1, 0)) *
      static_cast<uint64_t>(std::max(domainDimY, 0)) *
      static_cast<uint64_t>(std::max(domainDimZ, 0));
  const uint64_t yEdgeCount = static_cast<uint64_t>(std::max(domainDimX, 0)) *
                              static_cast<uint64_t>(std::max(domainDimY - 1, 0)) *
                              static_cast<uint64_t>(std::max(domainDimZ, 0));

  switch (key.axis) {
  case 0:
    return static_cast<uint64_t>(key.x) +
           static_cast<uint64_t>(std::max(domainDimX - 1, 0)) *
               (static_cast<uint64_t>(key.y) +
                static_cast<uint64_t>(domainDimY) *
                    static_cast<uint64_t>(key.z));
  case 1:
    return xEdgeCount + static_cast<uint64_t>(key.x) +
           static_cast<uint64_t>(domainDimX) *
               (static_cast<uint64_t>(key.y) +
                static_cast<uint64_t>(std::max(domainDimY - 1, 0)) *
                    static_cast<uint64_t>(key.z));
  case 2:
    return xEdgeCount + yEdgeCount + static_cast<uint64_t>(key.x) +
           static_cast<uint64_t>(domainDimX) *
               (static_cast<uint64_t>(key.y) +
                static_cast<uint64_t>(domainDimY) *
                    static_cast<uint64_t>(key.z));
  default:
    return 0ull;
  }
}

uint64_t compute_total_packed_grid_edge_count(int domainDimX, int domainDimY,
                                              int domainDimZ) {
  const uint64_t xEdgeCount =
      static_cast<uint64_t>(std::max(domainDimX - 1, 0)) *
      static_cast<uint64_t>(std::max(domainDimY, 0)) *
      static_cast<uint64_t>(std::max(domainDimZ, 0));
  const uint64_t yEdgeCount =
      static_cast<uint64_t>(std::max(domainDimX, 0)) *
      static_cast<uint64_t>(std::max(domainDimY - 1, 0)) *
      static_cast<uint64_t>(std::max(domainDimZ, 0));
  const uint64_t zEdgeCount =
      static_cast<uint64_t>(std::max(domainDimX, 0)) *
      static_cast<uint64_t>(std::max(domainDimY, 0)) *
      static_cast<uint64_t>(std::max(domainDimZ - 1, 0));
  return xEdgeCount + yEdgeCount + zEdgeCount;
}

template <class TriangleCountFn, class TriangleVertexBaseFn>
bool build_vulkan_direct_surface_buffers_impl(
    int domainDimX, int domainDimY, int domainDimZ, int cellDimX, int cellDimY,
    std::size_t activeCellCount, uint32_t totalTriangleCount,
    const std::vector<uint32_t> &activeCellIndices,
    const std::vector<uint32_t> &activeCellCubeIndices,
    TriangleCountFn &&get_triangle_count,
    TriangleVertexBaseFn &&get_triangle_vertex_base,
    std::vector<float> &outVertices, std::vector<int> &outFaces,
    std::string &outError) {
  const uint64_t totalEdgeCount =
      compute_total_packed_grid_edge_count(domainDimX, domainDimY, domainDimZ);
  const bool useDenseEdgeRemap =
      totalEdgeCount > 0ull && totalEdgeCount <= kDirectSurfaceDenseRemapMaxEdges;

  std::unordered_map<uint64_t, int> sparseEdgeToVertex;
  if (!useDenseEdgeRemap) {
    sparseEdgeToVertex.reserve(static_cast<std::size_t>(totalTriangleCount) * 2u);
  }

  if (useDenseEdgeRemap) {
    auto &denseWorkspace = get_direct_surface_dense_workspace();
    denseWorkspace.prepare(static_cast<std::size_t>(totalEdgeCount),
                           activeCellCount, totalTriangleCount);
    auto &denseEdgeToVertex = denseWorkspace.denseEdgeToVertex;
    auto &validTriangleCountsPerCell =
        denseWorkspace.validTriangleCountsPerCell;
    auto &usedEdgeVertices = denseWorkspace.usedEdgeVertices;
    auto &faceWriteOffsets = denseWorkspace.faceWriteOffsets;
    std::size_t validTriangleCount = 0;
    const std::size_t floatsPerTriangle = 3ull * 3ull;

    for (std::size_t cellIndex = 0; cellIndex < activeCellCount; ++cellIndex) {
      const uint32_t triangleCount = get_triangle_count(cellIndex);
      if (triangleCount > 5u) {
        outError =
            "Vulkan direct surface mesh triangle count exceeds Marching Cubes limits.";
        return false;
      }

      const uint32_t packedCellIndex = activeCellIndices[cellIndex];
      const uint32_t cubeIndex = activeCellCubeIndices[cellIndex];
      const int z = static_cast<int>(
          packedCellIndex / static_cast<uint32_t>(cellDimX * cellDimY));
      const uint32_t rem = packedCellIndex -
                           static_cast<uint32_t>(z * cellDimX * cellDimY);
      const int y = static_cast<int>(rem / static_cast<uint32_t>(cellDimX));
      const int x = static_cast<int>(rem - static_cast<uint32_t>(y * cellDimX));
      const float *cellTriangleVertices = get_triangle_vertex_base(cellIndex);
      uint32_t validTriangleCountForCell = 0u;

      for (uint32_t triangleIndex = 0; triangleIndex < triangleCount;
           ++triangleIndex) {
        const std::size_t triangleBase =
            static_cast<std::size_t>(triangleIndex) * floatsPerTriangle;
        const int triBaseIndex = static_cast<int>(triangleIndex) * 3;
        const int e0 = triTable[cubeIndex][triBaseIndex + 0];
        const int e1 = triTable[cubeIndex][triBaseIndex + 1];
        const int e2 = triTable[cubeIndex][triBaseIndex + 2];
        if (e0 < 0 || e1 < 0 || e2 < 0) {
          break;
        }

        const uint64_t edgeIds[3] = {
            make_packed_grid_edge_id(domainDimX, domainDimY, domainDimZ, x, y, z,
                                     e0),
            make_packed_grid_edge_id(domainDimX, domainDimY, domainDimZ, x, y, z,
                                     e1),
            make_packed_grid_edge_id(domainDimX, domainDimY, domainDimZ, x, y, z,
                                     e2)};
        if (edgeIds[0] >= totalEdgeCount || edgeIds[1] >= totalEdgeCount ||
            edgeIds[2] >= totalEdgeCount) {
          outError = "Vulkan direct surface mesh produced an out-of-range edge id.";
          return false;
        }

        const float *edgeVertices[3] = {
            cellTriangleVertices + triangleBase + 0u,
            cellTriangleVertices + triangleBase + 3u,
            cellTriangleVertices + triangleBase + 6u};
        for (int edgeSlot = 0; edgeSlot < 3; ++edgeSlot) {
          const std::size_t edgeIndex =
              static_cast<std::size_t>(edgeIds[edgeSlot]);
          if (!denseWorkspace.mark_edge(edgeIndex)) {
            continue;
          }
          denseEdgeToVertex[edgeIndex] =
              static_cast<int>(usedEdgeVertices.size() / 3u);
          usedEdgeVertices.push_back(edgeVertices[edgeSlot][0]);
          usedEdgeVertices.push_back(edgeVertices[edgeSlot][1]);
          usedEdgeVertices.push_back(edgeVertices[edgeSlot][2]);
        }
        ++validTriangleCount;
        ++validTriangleCountForCell;
      }

      validTriangleCountsPerCell[cellIndex] = validTriangleCountForCell;
    }

    outVertices = usedEdgeVertices;

    faceWriteOffsets[0] = 0u;
    for (std::size_t cellIndex = 0; cellIndex < activeCellCount; ++cellIndex) {
      faceWriteOffsets[cellIndex + 1u] =
          faceWriteOffsets[cellIndex] +
          static_cast<std::size_t>(validTriangleCountsPerCell[cellIndex]) * 3u;
    }
    outFaces.assign(faceWriteOffsets.back(), -1);

    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, activeCellCount),
        [&](const tbb::blocked_range<std::size_t> &range) {
          for (std::size_t cellIndex = range.begin(); cellIndex != range.end();
               ++cellIndex) {
            const uint32_t triangleCount = get_triangle_count(cellIndex);
            const uint32_t packedCellIndex = activeCellIndices[cellIndex];
            const uint32_t cubeIndex = activeCellCubeIndices[cellIndex];
            const int z = static_cast<int>(
                packedCellIndex / static_cast<uint32_t>(cellDimX * cellDimY));
            const uint32_t rem = packedCellIndex -
                                 static_cast<uint32_t>(z * cellDimX * cellDimY);
            const int y = static_cast<int>(rem / static_cast<uint32_t>(cellDimX));
            const int x =
                static_cast<int>(rem - static_cast<uint32_t>(y * cellDimX));
            const float *cellTriangleVertices = get_triangle_vertex_base(cellIndex);
            std::size_t faceWriteOffset = faceWriteOffsets[cellIndex];

            for (uint32_t triangleIndex = 0; triangleIndex < triangleCount;
                 ++triangleIndex) {
              const std::size_t triangleBase =
                  static_cast<std::size_t>(triangleIndex) * floatsPerTriangle;
              const int triBaseIndex = static_cast<int>(triangleIndex) * 3;
              const int e0 = triTable[cubeIndex][triBaseIndex + 0];
              const int e1 = triTable[cubeIndex][triBaseIndex + 1];
              const int e2 = triTable[cubeIndex][triBaseIndex + 2];
              if (e0 < 0 || e1 < 0 || e2 < 0) {
                break;
              }

              const uint64_t edgeIds[3] = {
                  make_packed_grid_edge_id(domainDimX, domainDimY, domainDimZ, x,
                                           y, z, e0),
                  make_packed_grid_edge_id(domainDimX, domainDimY, domainDimZ, x,
                                           y, z, e1),
                  make_packed_grid_edge_id(domainDimX, domainDimY, domainDimZ, x,
                                           y, z, e2)};
              const std::size_t edgeIndex0 =
                  static_cast<std::size_t>(edgeIds[0]);
              const std::size_t edgeIndex1 =
                  static_cast<std::size_t>(edgeIds[1]);
              const std::size_t edgeIndex2 =
                  static_cast<std::size_t>(edgeIds[2]);
              if (denseWorkspace.denseEdgeMarks[edgeIndex0] !=
                      denseWorkspace.denseEdgeGeneration ||
                  denseWorkspace.denseEdgeMarks[edgeIndex1] !=
                      denseWorkspace.denseEdgeGeneration ||
                  denseWorkspace.denseEdgeMarks[edgeIndex2] !=
                      denseWorkspace.denseEdgeGeneration) {
                continue;
              }
              const int i0 =
                  denseEdgeToVertex[edgeIndex0];
              const int i1 =
                  denseEdgeToVertex[edgeIndex1];
              const int i2 =
                  denseEdgeToVertex[edgeIndex2];
              if (i0 < 0 || i1 < 0 || i2 < 0 || i0 == i1 || i1 == i2 ||
                  i0 == i2) {
                continue;
              }

              outFaces[faceWriteOffset + 0u] = i0;
              outFaces[faceWriteOffset + 1u] = i1;
              outFaces[faceWriteOffset + 2u] = i2;
              faceWriteOffset += 3u;
            }
          }
        });

    std::size_t faceWriteOffset = 0u;
    for (std::size_t readOffset = 0u; readOffset + 2u < outFaces.size();
         readOffset += 3u) {
      const int i0 = outFaces[readOffset + 0u];
      const int i1 = outFaces[readOffset + 1u];
      const int i2 = outFaces[readOffset + 2u];
      if (i0 < 0 || i1 < 0 || i2 < 0 || i0 == i1 || i1 == i2 || i0 == i2) {
        continue;
      }
      if (faceWriteOffset != readOffset) {
        outFaces[faceWriteOffset + 0u] = i0;
        outFaces[faceWriteOffset + 1u] = i1;
        outFaces[faceWriteOffset + 2u] = i2;
      }
      faceWriteOffset += 3u;
    }

    outFaces.resize(faceWriteOffset);
    if (outFaces.empty()) {
      outError = "Vulkan direct surface mesh produced no valid faces.";
      return false;
    }

    outError.clear();
    return true;
  }

  outVertices.reserve(static_cast<std::size_t>(totalTriangleCount) * 9u);
  outFaces.reserve(static_cast<std::size_t>(totalTriangleCount) * 3u);

  const auto get_or_create_vertex_index =
      [&](uint64_t edgeId, const float *vertex) -> int {
    auto it = sparseEdgeToVertex.find(edgeId);
    if (it == sparseEdgeToVertex.end()) {
      const int vertexIndex = static_cast<int>(outVertices.size() / 3u);
      outVertices.push_back(vertex[0]);
      outVertices.push_back(vertex[1]);
      outVertices.push_back(vertex[2]);
      it = sparseEdgeToVertex.emplace(edgeId, vertexIndex).first;
    }
    return it->second;
  };

  const std::size_t floatsPerTriangle = 3ull * 3ull;
  for (std::size_t cellIndex = 0; cellIndex < activeCellCount; ++cellIndex) {
    const uint32_t triangleCount = get_triangle_count(cellIndex);
    if (triangleCount > 5u) {
      outError =
          "Vulkan direct surface mesh triangle count exceeds Marching Cubes limits.";
      return false;
    }

    const uint32_t packedCellIndex = activeCellIndices[cellIndex];
    const uint32_t cubeIndex = activeCellCubeIndices[cellIndex];
    const int z = static_cast<int>(
        packedCellIndex / static_cast<uint32_t>(cellDimX * cellDimY));
    const uint32_t rem = packedCellIndex -
                         static_cast<uint32_t>(z * cellDimX * cellDimY);
    const int y = static_cast<int>(rem / static_cast<uint32_t>(cellDimX));
    const int x = static_cast<int>(rem - static_cast<uint32_t>(y * cellDimX));
    const float *cellTriangleVertices = get_triangle_vertex_base(cellIndex);

    for (uint32_t triangleIndex = 0; triangleIndex < triangleCount;
         ++triangleIndex) {
      const std::size_t triangleBase =
          static_cast<std::size_t>(triangleIndex) * floatsPerTriangle;
      const float *v0 = cellTriangleVertices + triangleBase + 0u;
      const float *v1 = cellTriangleVertices + triangleBase + 3u;
      const float *v2 = cellTriangleVertices + triangleBase + 6u;
      const int triBaseIndex = static_cast<int>(triangleIndex) * 3;
      const int e0 = triTable[cubeIndex][triBaseIndex + 0];
      const int e1 = triTable[cubeIndex][triBaseIndex + 1];
      const int e2 = triTable[cubeIndex][triBaseIndex + 2];
      if (e0 < 0 || e1 < 0 || e2 < 0) {
        break;
      }

      const int i0 = get_or_create_vertex_index(
          make_packed_grid_edge_id(domainDimX, domainDimY, domainDimZ, x, y, z,
                                   e0),
          v0);
      const int i1 = get_or_create_vertex_index(
          make_packed_grid_edge_id(domainDimX, domainDimY, domainDimZ, x, y, z,
                                   e1),
          v1);
      const int i2 = get_or_create_vertex_index(
          make_packed_grid_edge_id(domainDimX, domainDimY, domainDimZ, x, y, z,
                                   e2),
          v2);
      if (i0 < 0 || i1 < 0 || i2 < 0 || i0 == i1 || i1 == i2 || i0 == i2) {
        continue;
      }

      outFaces.push_back(i0);
      outFaces.push_back(i1);
      outFaces.push_back(i2);
    }
  }

  if (outFaces.empty()) {
    outError = "Vulkan direct surface mesh produced no valid faces.";
    return false;
  }

  outError.clear();
  return true;
}

bool build_vulkan_direct_surface_buffers_from_compact_edge_ids(
    int domainDimX, int domainDimY, int domainDimZ,
    const VulkanResidentSurfaceMeshView &surfaceMeshView,
    std::vector<float> &outVertices, std::vector<int> &outFaces,
    std::string &outError) {
  outVertices.clear();
  outFaces.clear();

  if (surfaceMeshView.activeCellCount == 0u ||
      surfaceMeshView.triangleCounts == nullptr ||
      surfaceMeshView.triangleVertices == nullptr ||
      surfaceMeshView.triangleOffsets == nullptr ||
      surfaceMeshView.triangleEdgeIds == nullptr) {
    outError = "Vulkan compact resident surface buffers are incomplete.";
    return false;
  }

  const uint64_t totalEdgeCount =
      compute_total_packed_grid_edge_count(domainDimX, domainDimY, domainDimZ);
  if (totalEdgeCount == 0ull ||
      totalEdgeCount > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    outError =
        "Vulkan compact resident surface edge-id range is unsupported for direct packing.";
    return false;
  }

  // Phase C: GPU Accelerated Edge Deduplication
  return run_frost_vulkan_generate_deduplicated_surface_mesh(
      surfaceMeshView.totalTriangleCount, static_cast<uint32_t>(totalEdgeCount),
      outVertices, outFaces, outError);
}

std::size_t scalar_field_index(const VulkanParticleComputeResult &computeResult,
                               int x, int y, int z) {
  const std::size_t dimX =
      static_cast<std::size_t>(computeResult.domainDimensions[0]);
  const std::size_t dimY =
      static_cast<std::size_t>(computeResult.domainDimensions[1]);
  return static_cast<std::size_t>(x) +
         dimX * (static_cast<std::size_t>(y) +
                 dimY * static_cast<std::size_t>(z));
}

frantic::graphics::vector3f world_position_from_grid(
    const VulkanParticleComputeResult &computeResult, int gridX, int gridY,
    int gridZ) {
  return frantic::graphics::vector3f(
      (computeResult.domainMinVoxel[0] + gridX) * computeResult.voxelLength,
      (computeResult.domainMinVoxel[1] + gridY) * computeResult.voxelLength,
      (computeResult.domainMinVoxel[2] + gridZ) * computeResult.voxelLength);
}

float evaluate_vulkan_metaball_weight(float distance, float effectRadius) {
  if (effectRadius <= 0.0f) {
    return 0.0f;
  }

  const float normalizedDistance = distance / effectRadius;
  if (distance < (0.33333333f) * effectRadius) {
    return 1.0f - 3.0f * normalizedDistance * normalizedDistance;
  }
  if (distance < effectRadius) {
    const float blend = 1.0f - normalizedDistance;
    return 1.5f * blend * blend;
  }
  return 0.0f;
}

float evaluate_vulkan_plain_density(float distance, float effectRadius) {
  if (effectRadius <= 0.0f || distance >= effectRadius) {
    return 0.0f;
  }
  return 1.0f - (distance / effectRadius);
}

float evaluate_vulkan_zhu_bridson_kernel(float distanceSquared,
                                         float kernelSupportRadius) {
  if (kernelSupportRadius <= 0.0f) {
    return 0.0f;
  }
  const float kernelSupportSquared = kernelSupportRadius * kernelSupportRadius;
  if (distanceSquared >= kernelSupportSquared) {
    return 0.0f;
  }
  const float x = 1.0f - distanceSquared / kernelSupportSquared;
  return x * x * x;
}

float evaluate_vulkan_anisotropic_signed_distance(
    const VulkanParticleComputeResult &computeResult,
    const frantic::graphics::vector3f &samplePosition,
    const frantic::graphics::vector3f &particlePosition,
    const frantic::graphics::vector3f &velocity, float effectRadius) {
  if (effectRadius <= 0.0f) {
    return std::numeric_limits<float>::max();
  }

  const float vx = velocity.x;
  const float vy = velocity.y;
  const float vz = velocity.z;
  const float speed = std::sqrt(vx * vx + vy * vy + vz * vz);
  const float anisotropyLimit = std::max(computeResult.anisotropyMaxScale, 1.0f);
  const float safeRadius =
      std::max(effectRadius, std::max(computeResult.voxelLength, 1.0e-4f));
  const float stretch =
      (speed > 1.0e-5f)
          ? std::min(1.0f + 0.5f * (speed / safeRadius), anisotropyLimit)
          : 1.0f;
  const float majorRadius = effectRadius * stretch;
  const float minorRadius = effectRadius / std::sqrt(std::max(stretch, 1.0f));
  const float isoBiasDistance =
      (computeResult.fieldThreshold - 0.5f) * effectRadius * 0.5f;

  const frantic::graphics::vector3f delta = samplePosition - particlePosition;
  float signedDistance = 0.0f;
  if (speed > 1.0e-5f && majorRadius > 1.0e-6f && minorRadius > 1.0e-6f) {
    const float invSpeed = 1.0f / speed;
    const frantic::graphics::vector3f direction(vx * invSpeed, vy * invSpeed,
                                                vz * invSpeed);
    const float parallelDistance =
        delta.x * direction.x + delta.y * direction.y + delta.z * direction.z;
    const frantic::graphics::vector3f perpendicular =
        delta - direction * parallelDistance;
    const float perpendicularSquared =
        perpendicular.x * perpendicular.x + perpendicular.y * perpendicular.y +
        perpendicular.z * perpendicular.z;
    const float normalizedDistance = std::sqrt(
        (parallelDistance * parallelDistance) / (majorRadius * majorRadius) +
        perpendicularSquared / (minorRadius * minorRadius));
    signedDistance = (normalizedDistance - 1.0f) * effectRadius;
  } else {
    signedDistance =
        frantic::graphics::vector3f::distance(samplePosition, particlePosition) -
        effectRadius;
  }

  return signedDistance - isoBiasDistance;
}

frantic::graphics::vector3 sample_position_to_vulkan_voxel(
    const VulkanParticleComputeResult &computeResult,
    const frantic::graphics::vector3f &samplePosition) {
  return frantic::graphics::vector3(
      static_cast<int>(std::floor(samplePosition.x / computeResult.voxelLength)),
      static_cast<int>(std::floor(samplePosition.y / computeResult.voxelLength)),
      static_cast<int>(std::floor(samplePosition.z / computeResult.voxelLength)));
}

bool sample_voxel_inside_particle_bounds(
    const VulkanParticleComputeResult &computeResult,
    const frantic::graphics::vector3f &samplePosition, std::size_t particleIndex) {
  if (computeResult.minVoxelBounds.size() != computeResult.particles.size() ||
      computeResult.maxVoxelBoundsExclusive.size() !=
          computeResult.particles.size()) {
    return true;
  }

  const frantic::graphics::vector3 sampleVoxel =
      sample_position_to_vulkan_voxel(computeResult, samplePosition);
  const std::size_t offset = particleIndex * 4;
  return sampleVoxel.x >= computeResult.minVoxelBounds[offset + 0] &&
         sampleVoxel.y >= computeResult.minVoxelBounds[offset + 1] &&
         sampleVoxel.z >= computeResult.minVoxelBounds[offset + 2] &&
         sampleVoxel.x < computeResult.maxVoxelBoundsExclusive[offset + 0] &&
         sampleVoxel.y < computeResult.maxVoxelBoundsExclusive[offset + 1] &&
         sampleVoxel.z < computeResult.maxVoxelBoundsExclusive[offset + 2];
}

struct VulkanCandidateRange {
  const uint32_t *current = nullptr;
  const uint32_t *end = nullptr;
};

bool merge_compacted_vulkan_candidate_ranges(VulkanCandidateRange *ranges,
                                             std::size_t rangeCount,
                                             std::vector<uint32_t> &outCandidates) {
  outCandidates.clear();
  outCandidates.reserve(rangeCount * 8);

  uint32_t lastValue = 0;
  bool hasLastValue = false;
  for (;;) {
    uint32_t nextValue = 0;
    bool foundValue = false;
    for (std::size_t i = 0; i < rangeCount; ++i) {
      if (ranges[i].current == ranges[i].end) {
        continue;
      }
      if (!foundValue || *ranges[i].current < nextValue) {
        nextValue = *ranges[i].current;
        foundValue = true;
      }
    }
    if (!foundValue) {
      break;
    }
    if (!hasLastValue || nextValue != lastValue) {
      outCandidates.push_back(nextValue);
      lastValue = nextValue;
      hasLastValue = true;
    }
    for (std::size_t i = 0; i < rangeCount; ++i) {
      while (ranges[i].current != ranges[i].end &&
             *ranges[i].current == nextValue) {
        ++ranges[i].current;
      }
    }
  }

  return !outCandidates.empty();
}

bool collect_compacted_vulkan_sample_ranges(
    const VulkanParticleComputeResult &computeResult,
    const frantic::graphics::vector3f &samplePosition,
    VulkanCandidateRange *outRanges, std::size_t maxRangeCount,
    std::size_t &outRangeCount) {
  if (computeResult.activeVoxelIndices.empty() ||
      computeResult.activeVoxelParticleOffsets.size() !=
          computeResult.activeVoxelIndices.size() + 1 ||
      computeResult.domainDimensions[0] <= 0 ||
      computeResult.domainDimensions[1] <= 0 ||
      computeResult.domainDimensions[2] <= 0 ||
      computeResult.voxelLength <= 0.0f) {
    return false;
  }

  outRangeCount = 0;

  const auto collect_axis_voxels = [&](float positionComponent, int *outValues,
                                       int &outCount) {
    const float scaled = positionComponent / computeResult.voxelLength;
    const int base = static_cast<int>(std::floor(scaled));
    outValues[0] = base;
    outCount = 1;
    const float nearestInteger = std::round(scaled);
    if (std::fabs(scaled - nearestInteger) <= 1e-4f) {
      outValues[outCount++] = base - 1;
    }
  };

  int voxelXs[2];
  int voxelYs[2];
  int voxelZs[2];
  int xCount = 0;
  int yCount = 0;
  int zCount = 0;
  const std::size_t voxelCount =
      static_cast<std::size_t>(computeResult.domainDimensions[0]) *
      static_cast<std::size_t>(computeResult.domainDimensions[1]) *
      static_cast<std::size_t>(computeResult.domainDimensions[2]);
  collect_axis_voxels(samplePosition.x, voxelXs, xCount);
  collect_axis_voxels(samplePosition.y, voxelYs, yCount);
  collect_axis_voxels(samplePosition.z, voxelZs, zCount);

  for (int zi = 0; zi < zCount; ++zi) {
    for (int yi = 0; yi < yCount; ++yi) {
      for (int xi = 0; xi < xCount; ++xi) {
        const int localX = voxelXs[xi] - computeResult.domainMinVoxel[0];
        const int localY = voxelYs[yi] - computeResult.domainMinVoxel[1];
        const int localZ = voxelZs[zi] - computeResult.domainMinVoxel[2];
        if (localX < 0 || localY < 0 || localZ < 0 ||
            localX >= computeResult.domainDimensions[0] ||
            localY >= computeResult.domainDimensions[1] ||
            localZ >= computeResult.domainDimensions[2]) {
          continue;
        }

        const uint32_t packedVoxelIndex = static_cast<uint32_t>(
            localX + computeResult.domainDimensions[0] *
                         (localY + computeResult.domainDimensions[1] * localZ));
        std::size_t compactIndex = 0;
        if (computeResult.activeVoxelCompactLookup.size() == voxelCount) {
          const int32_t lookup =
              computeResult.activeVoxelCompactLookup[packedVoxelIndex];
          if (lookup < 0) {
            continue;
          }
          compactIndex = static_cast<std::size_t>(lookup);
        } else {
          const auto it =
              std::lower_bound(computeResult.activeVoxelIndices.begin(),
                               computeResult.activeVoxelIndices.end(),
                               packedVoxelIndex);
          if (it == computeResult.activeVoxelIndices.end() ||
              *it != packedVoxelIndex) {
            continue;
          }
          compactIndex = static_cast<std::size_t>(
              std::distance(computeResult.activeVoxelIndices.begin(), it));
        }
        const std::size_t begin = static_cast<std::size_t>(
            computeResult.activeVoxelParticleOffsets[compactIndex]);
        const std::size_t end = static_cast<std::size_t>(
            computeResult.activeVoxelParticleOffsets[compactIndex + 1]);
        if (begin >= end ||
            end > computeResult.activeVoxelParticleIndices.size()) {
          continue;
        }
        if (outRangeCount < maxRangeCount) {
          outRanges[outRangeCount].current =
              computeResult.activeVoxelParticleIndices.data() + begin;
          outRanges[outRangeCount].end =
              computeResult.activeVoxelParticleIndices.data() + end;
          ++outRangeCount;
        }
      }
    }
  }

  return outRangeCount > 0;
}

bool build_compacted_vulkan_sample_candidates(
    const VulkanParticleComputeResult &computeResult,
    const frantic::graphics::vector3f &samplePosition,
    std::vector<uint32_t> &outCandidates) {
  VulkanCandidateRange ranges[8];
  std::size_t rangeCount = 0;
  if (!collect_compacted_vulkan_sample_ranges(computeResult, samplePosition,
                                              ranges, std::size(ranges),
                                              rangeCount)) {
    return false;
  }

  return merge_compacted_vulkan_candidate_ranges(ranges, rangeCount,
                                                 outCandidates);
}

bool build_compacted_vulkan_edge_candidates(
    const VulkanParticleComputeResult &computeResult,
    const frantic::graphics::vector3f &edgeStart,
    const frantic::graphics::vector3f &edgeEnd,
    std::vector<uint32_t> &outCandidates) {
  VulkanCandidateRange ranges[24];
  std::size_t rangeCount = 0;

  const auto append_sample_ranges =
      [&](const frantic::graphics::vector3f &samplePosition) {
        std::size_t sampleRangeCount = 0;
        const bool collected = collect_compacted_vulkan_sample_ranges(
            computeResult, samplePosition, ranges + rangeCount,
            std::size(ranges) - rangeCount, sampleRangeCount);
        rangeCount += sampleRangeCount;
        return collected;
      };

  bool foundCandidates = append_sample_ranges(edgeStart);
  foundCandidates =
      append_sample_ranges((edgeStart + edgeEnd) * 0.5f) || foundCandidates;
  foundCandidates = append_sample_ranges(edgeEnd) || foundCandidates;

  if (!foundCandidates || rangeCount == 0) {
    outCandidates.clear();
    return false;
  }

  return merge_compacted_vulkan_candidate_ranges(ranges, rangeCount,
                                                 outCandidates);
}

bool get_compacted_vulkan_voxel_particle_range(
    const VulkanParticleComputeResult &computeResult,
    const frantic::graphics::vector3f &samplePosition, std::size_t &outBegin,
    std::size_t &outEnd) {
  if (computeResult.activeVoxelIndices.empty() ||
      computeResult.activeVoxelParticleOffsets.size() !=
          computeResult.activeVoxelIndices.size() + 1 ||
      computeResult.domainDimensions[0] <= 0 ||
      computeResult.domainDimensions[1] <= 0 ||
      computeResult.domainDimensions[2] <= 0 ||
      computeResult.voxelLength <= 0.0f) {
    return false;
  }

  const frantic::graphics::vector3 sampleVoxel =
      sample_position_to_vulkan_voxel(computeResult, samplePosition);
  const int localX = sampleVoxel.x - computeResult.domainMinVoxel[0];
  const int localY = sampleVoxel.y - computeResult.domainMinVoxel[1];
  const int localZ = sampleVoxel.z - computeResult.domainMinVoxel[2];
  if (localX < 0 || localY < 0 || localZ < 0 ||
      localX >= computeResult.domainDimensions[0] ||
      localY >= computeResult.domainDimensions[1] ||
      localZ >= computeResult.domainDimensions[2]) {
    return false;
  }

  const uint32_t packedVoxelIndex = static_cast<uint32_t>(
      localX + computeResult.domainDimensions[0] *
                   (localY + computeResult.domainDimensions[1] * localZ));
  std::size_t compactIndex = 0;
  if (computeResult.activeVoxelCompactLookup.size() ==
      static_cast<std::size_t>(computeResult.domainDimensions[0]) *
          static_cast<std::size_t>(computeResult.domainDimensions[1]) *
          static_cast<std::size_t>(computeResult.domainDimensions[2])) {
    const int32_t lookup = computeResult.activeVoxelCompactLookup[packedVoxelIndex];
    if (lookup < 0) {
      return false;
    }
    compactIndex = static_cast<std::size_t>(lookup);
  } else {
    const auto it =
        std::lower_bound(computeResult.activeVoxelIndices.begin(),
                         computeResult.activeVoxelIndices.end(),
                         packedVoxelIndex);
    if (it == computeResult.activeVoxelIndices.end() || *it != packedVoxelIndex) {
      return false;
    }
    const std::size_t foundIndex = static_cast<std::size_t>(
        std::distance(computeResult.activeVoxelIndices.begin(), it));
    compactIndex = foundIndex;
  }
  outBegin = static_cast<std::size_t>(
      computeResult.activeVoxelParticleOffsets[compactIndex]);
  outEnd = static_cast<std::size_t>(
      computeResult.activeVoxelParticleOffsets[compactIndex + 1]);
  return outBegin <= outEnd &&
         outEnd <= computeResult.activeVoxelParticleIndices.size();
}

int count_vulkan_sample_boundary_axes(
    const VulkanParticleComputeResult &computeResult,
    const frantic::graphics::vector3f &samplePosition) {
  if (computeResult.voxelLength <= 0.0f) {
    return 0;
  }

  const auto is_on_boundary = [&](float component) {
    const float scaled = component / computeResult.voxelLength;
    return std::fabs(scaled - std::round(scaled)) <= 1e-4f;
  };

  int boundaryAxes = 0;
  if (is_on_boundary(samplePosition.x)) {
    ++boundaryAxes;
  }
  if (is_on_boundary(samplePosition.y)) {
    ++boundaryAxes;
  }
  if (is_on_boundary(samplePosition.z)) {
    ++boundaryAxes;
  }
  return boundaryAxes;
}

float evaluate_vulkan_scalar_field_exact_with_candidates(
    const VulkanParticleComputeResult &computeResult,
    const frantic::graphics::vector3f &samplePosition,
    const std::vector<uint32_t> *fixedCandidateIndices) {
  const float outsideField =
      (computeResult.scalarFieldMode == VulkanScalarFieldMode::metaball ||
       computeResult.scalarFieldMode ==
           VulkanScalarFieldMode::plain_marching_cubes)
          ? std::max(computeResult.fieldThreshold,
                     computeResult.surfaceIsoValue + 1.0e-3f)
          : std::max(computeResult.fieldThreshold,
                     std::max(computeResult.voxelLength, 1.0f));

  const std::size_t particleCount = computeResult.particles.size() / 4;
  if (particleCount == 0) {
    return outsideField;
  }

  const auto for_each_candidate_particle = [&](auto &&callback) {
    if (fixedCandidateIndices != nullptr && !fixedCandidateIndices->empty()) {
      for (uint32_t particleIndex : *fixedCandidateIndices) {
        callback(static_cast<std::size_t>(particleIndex));
      }
      return;
    }

    if (count_vulkan_sample_boundary_axes(computeResult, samplePosition) >= 3) {
      std::vector<uint32_t> compactedCandidates;
      if (build_compacted_vulkan_sample_candidates(computeResult, samplePosition,
                                                   compactedCandidates)) {
        for (uint32_t particleIndex : compactedCandidates) {
          callback(static_cast<std::size_t>(particleIndex));
        }
        return;
      }
    }

    std::size_t candidateBegin = 0;
    std::size_t candidateEnd = particleCount;
    if (get_compacted_vulkan_voxel_particle_range(computeResult, samplePosition,
                                                  candidateBegin,
                                                  candidateEnd)) {
      for (std::size_t candidateIndex = candidateBegin;
           candidateIndex < candidateEnd; ++candidateIndex) {
        callback(static_cast<std::size_t>(
            computeResult.activeVoxelParticleIndices[candidateIndex]));
      }
      return;
    }

    for (std::size_t particleIndex = 0; particleIndex < particleCount;
         ++particleIndex) {
      if (!sample_voxel_inside_particle_bounds(computeResult, samplePosition,
                                               particleIndex)) {
        continue;
      }
      callback(particleIndex);
    }
  };

  switch (computeResult.scalarFieldMode) {
  case VulkanScalarFieldMode::metaball: {
    float field = computeResult.fieldThreshold;
    bool foundInfluence = false;
    for_each_candidate_particle([&](std::size_t i) {
      const std::size_t offset = i * 4;
      const frantic::graphics::vector3f particlePosition(
          computeResult.particles[offset + 0], computeResult.particles[offset + 1],
          computeResult.particles[offset + 2]);
      const float effectRadius =
          std::max(computeResult.particles[offset + 3] *
                       computeResult.fieldRadiusScale,
                   0.0f);
      if (effectRadius <= 0.0f) {
        return;
      }
      const float weight = evaluate_vulkan_metaball_weight(
          frantic::graphics::vector3f::distance(samplePosition, particlePosition),
          effectRadius);
      if (weight <= 0.0f) {
        return;
      }
      field -= weight;
      foundInfluence = true;
    });
    return foundInfluence ? field : outsideField;
  }
  case VulkanScalarFieldMode::plain_marching_cubes: {
    float field = computeResult.fieldThreshold;
    bool foundInfluence = false;
    for_each_candidate_particle([&](std::size_t i) {
      const std::size_t offset = i * 4;
      const frantic::graphics::vector3f particlePosition(
          computeResult.particles[offset + 0], computeResult.particles[offset + 1],
          computeResult.particles[offset + 2]);
      const float effectRadius =
          std::max(computeResult.particles[offset + 3] *
                       computeResult.fieldRadiusScale,
                   0.0f);
      if (effectRadius <= 0.0f) {
        return;
      }
      const float density = evaluate_vulkan_plain_density(
          frantic::graphics::vector3f::distance(samplePosition, particlePosition),
          effectRadius);
      if (density <= 0.0f) {
        return;
      }
      field -= density;
      foundInfluence = true;
    });
    return foundInfluence ? field : outsideField;
  }
  case VulkanScalarFieldMode::zhu_bridson_blend: {
    float totalWeight = 0.0f;
    frantic::graphics::vector3f blendedCenter(0.0f);
    float blendedRadius = 0.0f;
    for_each_candidate_particle([&](std::size_t i) {
      const std::size_t offset = i * 4;
      const frantic::graphics::vector3f particlePosition(
          computeResult.particles[offset + 0], computeResult.particles[offset + 1],
          computeResult.particles[offset + 2]);
      const float particleRadius =
          std::max(computeResult.particles[offset + 3], 0.0f);
      if (particleRadius <= 0.0f) {
        return;
      }
      const float kernelSupportRadius =
          computeResult.kernelSupportRadius > 0.0f
              ? computeResult.kernelSupportRadius
              : std::max(particleRadius * computeResult.fieldRadiusScale, 0.0f);
      const frantic::graphics::vector3f delta =
          samplePosition - particlePosition;
      const float distanceSquared =
          delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
      const float weight = evaluate_vulkan_zhu_bridson_kernel(
          distanceSquared, kernelSupportRadius);
      if (weight <= 0.0f) {
        return;
      }
      totalWeight += weight;
      blendedCenter += particlePosition * weight;
      blendedRadius += weight * particleRadius;
    });
    if (totalWeight <= 1e-6f) {
      return outsideField;
    }
    blendedCenter /= totalWeight;
    blendedRadius /= totalWeight;
    return frantic::graphics::vector3f::distance(samplePosition, blendedCenter) -
           blendedRadius;
  }
  case VulkanScalarFieldMode::anisotropic_velocity: {
    float minSignedDistance = std::numeric_limits<float>::max();
    const bool hasVelocity =
        computeResult.velocities.size() == computeResult.particles.size();
    for_each_candidate_particle([&](std::size_t i) {
      const std::size_t offset = i * 4;
      const frantic::graphics::vector3f particlePosition(
          computeResult.particles[offset + 0], computeResult.particles[offset + 1],
          computeResult.particles[offset + 2]);
      const float effectRadius =
          std::max(computeResult.particles[offset + 3] *
                       computeResult.fieldRadiusScale,
                   0.0f);
      if (effectRadius <= 0.0f) {
        return;
      }
      const frantic::graphics::vector3f velocity =
          hasVelocity
              ? frantic::graphics::vector3f(computeResult.velocities[offset + 0],
                                            computeResult.velocities[offset + 1],
                                            computeResult.velocities[offset + 2])
              : frantic::graphics::vector3f(0.0f);
      minSignedDistance = std::min(
          minSignedDistance, evaluate_vulkan_anisotropic_signed_distance(
                                 computeResult, samplePosition, particlePosition,
                                 velocity, effectRadius));
    });
    return minSignedDistance == std::numeric_limits<float>::max()
               ? outsideField
               : minSignedDistance;
  }
  case VulkanScalarFieldMode::sphere_signed_distance:
  case VulkanScalarFieldMode::coverage_fallback:
  default: {
    float minSignedDistance = std::numeric_limits<float>::max();
    for_each_candidate_particle([&](std::size_t i) {
      const std::size_t offset = i * 4;
      const frantic::graphics::vector3f particlePosition(
          computeResult.particles[offset + 0], computeResult.particles[offset + 1],
          computeResult.particles[offset + 2]);
      const float effectRadius =
          std::max(computeResult.particles[offset + 3] *
                       computeResult.fieldRadiusScale,
                   0.0f);
      if (effectRadius <= 0.0f) {
        return;
      }
      minSignedDistance = std::min(
          minSignedDistance,
          frantic::graphics::vector3f::distance(samplePosition, particlePosition) -
              effectRadius);
    });
    if (minSignedDistance == std::numeric_limits<float>::max()) {
      return outsideField;
    }
    return computeResult.fieldThreshold > 0.0f
               ? std::min(minSignedDistance, computeResult.fieldThreshold)
               : minSignedDistance;
  }
  }
}

float evaluate_vulkan_scalar_field_exact(
    const VulkanParticleComputeResult &computeResult,
    const frantic::graphics::vector3f &samplePosition) {
  return evaluate_vulkan_scalar_field_exact_with_candidates(
      computeResult, samplePosition, nullptr);
}

frantic::graphics::vector3f interpolate_isosurface_vertex(
    float isoValue, const frantic::graphics::vector3f &p1,
    const frantic::graphics::vector3f &p2, float v1, float v2) {
  if (std::fabs(isoValue - v1) < 1e-6f) {
    return p1;
  }
  if (std::fabs(isoValue - v2) < 1e-6f) {
    return p2;
  }
  if (std::fabs(v1 - v2) < 1e-6f) {
    return (p1 + p2) * 0.5f;
  }

  const float t = (isoValue - v1) / (v2 - v1);
  return p1 + (p2 - p1) * t;
}

bool should_refine_vulkan_isosurface_edge(
    const VulkanParticleComputeResult &computeResult) {
  if (computeResult.scalarFieldMode == VulkanScalarFieldMode::coverage_fallback) {
    return false;
  }
  if (!computeResult.activeVoxelParticleIndices.empty() &&
      !computeResult.activeVoxelParticleOffsets.empty()) {
    return computeResult.activeVoxelParticleIndices.size() <= 350000ull;
  }
  const std::size_t particleCount = computeResult.particles.size() / 4;
  return particleCount > 0 && particleCount <= 4096;
}

float get_exact_vulkan_corner_field_cached(
    const VulkanParticleComputeResult &computeResult, std::size_t packedIndex,
    const frantic::graphics::vector3f &position,
    const std::vector<uint32_t> *fixedCandidateIndices,
    std::vector<float> &cache, std::vector<uint8_t> &cacheValid);

float get_exact_vulkan_corner_field_cached(
    const VulkanParticleComputeResult &computeResult, std::size_t packedIndex,
    const frantic::graphics::vector3f &position, std::vector<float> &cache,
    std::vector<uint8_t> &cacheValid) {
  return get_exact_vulkan_corner_field_cached(
      computeResult, packedIndex, position, nullptr, cache, cacheValid);
}

float get_exact_vulkan_corner_field_cached(
    const VulkanParticleComputeResult &computeResult, std::size_t packedIndex,
    const frantic::graphics::vector3f &position,
    const std::vector<uint32_t> *fixedCandidateIndices,
    std::vector<float> &cache, std::vector<uint8_t> &cacheValid) {
  if (packedIndex < cacheValid.size() && cacheValid[packedIndex] != 0) {
    return cache[packedIndex];
  }

  const float value = evaluate_vulkan_scalar_field_exact_with_candidates(
      computeResult, position, fixedCandidateIndices);
  if (packedIndex < cacheValid.size()) {
    cache[packedIndex] = value;
    cacheValid[packedIndex] = 1;
  }
  return value;
}

frantic::graphics::vector3f refine_vulkan_isosurface_vertex(
    const VulkanParticleComputeResult &computeResult, float isoValue,
    std::size_t p1PackedIndex, std::size_t p2PackedIndex,
    const frantic::graphics::vector3f &p1,
    const frantic::graphics::vector3f &p2, float v1, float v2,
    std::vector<float> &exactCornerFieldCache,
    std::vector<uint8_t> &exactCornerFieldValid) {
  frantic::graphics::vector3f a = p1;
  frantic::graphics::vector3f b = p2;
  std::vector<uint32_t> edgeCandidateIndices;
  const std::vector<uint32_t> *edgeCandidates = nullptr;
  if (build_compacted_vulkan_edge_candidates(computeResult, p1, p2,
                                             edgeCandidateIndices)) {
    edgeCandidates = &edgeCandidateIndices;
  }
  float fa = get_exact_vulkan_corner_field_cached(
      computeResult, p1PackedIndex, a, edgeCandidates, exactCornerFieldCache,
      exactCornerFieldValid);
  float fb = get_exact_vulkan_corner_field_cached(
      computeResult, p2PackedIndex, b, edgeCandidates, exactCornerFieldCache,
      exactCornerFieldValid);
  if (((fa < isoValue) && (fb < isoValue)) ||
      ((fa >= isoValue) && (fb >= isoValue))) {
    fa = v1;
    fb = v2;
  }

  for (int iteration = 0; iteration < 4; ++iteration) {
    const frantic::graphics::vector3f mid = (a + b) * 0.5f;
    const float fm = evaluate_vulkan_scalar_field_exact_with_candidates(
        computeResult, mid, edgeCandidates);
    if ((fa < isoValue) == (fm < isoValue)) {
      a = mid;
      fa = fm;
    } else {
      b = mid;
      fb = fm;
    }
  }

  return interpolate_isosurface_vertex(isoValue, a, b, fa, fb);
}

int compute_vulkan_cell_cube_index(
    const VulkanParticleComputeResult &computeResult, int x, int y, int z,
    float isoValue) {
  int cubeIndex = 0;
  for (int corner = 0; corner < 8; ++corner) {
    const int gx = x + kMarchingCubeCornerOffsets[corner][0];
    const int gy = y + kMarchingCubeCornerOffsets[corner][1];
    const int gz = z + kMarchingCubeCornerOffsets[corner][2];
    const std::size_t packedIndex =
        scalar_field_index(computeResult, gx, gy, gz);
    if (computeResult.voxelScalarField[packedIndex] < isoValue) {
      cubeIndex |= (1 << corner);
    }
  }
  return cubeIndex;
}

void build_vulkan_candidate_cell_indices(
    const VulkanParticleComputeResult &computeResult, float isoValue,
    std::vector<uint32_t> &outCandidateCellIndices) {
  outCandidateCellIndices.clear();

  const int dimX = computeResult.domainDimensions[0];
  const int dimY = computeResult.domainDimensions[1];
  const int dimZ = computeResult.domainDimensions[2];
  const int cellDimX = dimX - 1;
  const int cellDimY = dimY - 1;
  const int cellDimZ = dimZ - 1;
  if (cellDimX <= 0 || cellDimY <= 0 || cellDimZ <= 0) {
    return;
  }

  const std::size_t expectedSamples =
      static_cast<std::size_t>(dimX) * static_cast<std::size_t>(dimY) *
      static_cast<std::size_t>(dimZ);
  const std::size_t candidateCellCount =
      static_cast<std::size_t>(cellDimX) * static_cast<std::size_t>(cellDimY) *
      static_cast<std::size_t>(cellDimZ);
  std::vector<uint8_t> candidateMask(candidateCellCount, 0);
  outCandidateCellIndices.reserve(std::min(candidateCellCount, expectedSamples));

  const auto add_candidate_cell = [&](int x, int y, int z) {
    if (x < 0 || y < 0 || z < 0 || x >= cellDimX || y >= cellDimY ||
        z >= cellDimZ) {
      return;
    }
    const std::size_t cellIndex =
        static_cast<std::size_t>(x) +
        static_cast<std::size_t>(cellDimX) *
            (static_cast<std::size_t>(y) +
             static_cast<std::size_t>(cellDimY) * static_cast<std::size_t>(z));
    if (candidateMask[cellIndex] != 0) {
      return;
    }
    candidateMask[cellIndex] = 1;
    outCandidateCellIndices.push_back(static_cast<uint32_t>(cellIndex));
  };

  const auto add_cells_around_voxel_index = [&](uint32_t packedVoxelIndex) {
    const int z = static_cast<int>(
        packedVoxelIndex / static_cast<uint32_t>(dimX * dimY));
    const uint32_t rem =
        packedVoxelIndex - static_cast<uint32_t>(z * dimX * dimY);
    const int y = static_cast<int>(rem / static_cast<uint32_t>(dimX));
    const int x = static_cast<int>(rem - static_cast<uint32_t>(y * dimX));
    add_candidate_cell(x, y, z);
    add_candidate_cell(x - 1, y, z);
    add_candidate_cell(x, y - 1, z);
    add_candidate_cell(x, y, z - 1);
    add_candidate_cell(x - 1, y - 1, z);
    add_candidate_cell(x - 1, y, z - 1);
    add_candidate_cell(x, y - 1, z - 1);
    add_candidate_cell(x - 1, y - 1, z - 1);
  };

  bool foundNegativeSample = false;
  if (!computeResult.activeVoxelIndices.empty()) {
    for (uint32_t packedVoxelIndex : computeResult.activeVoxelIndices) {
      if (packedVoxelIndex >= computeResult.voxelScalarField.size()) {
        continue;
      }
      if (computeResult.voxelScalarField[packedVoxelIndex] >= isoValue) {
        continue;
      }
      foundNegativeSample = true;
      add_cells_around_voxel_index(packedVoxelIndex);
    }
    if (!foundNegativeSample) {
      for (uint32_t packedVoxelIndex : computeResult.activeVoxelIndices) {
        add_cells_around_voxel_index(packedVoxelIndex);
      }
    }
  } else if (!foundNegativeSample &&
             computeResult.voxelCoverageCounts.size() == expectedSamples) {
    for (int z = 0; z < dimZ; ++z) {
      for (int y = 0; y < dimY; ++y) {
        for (int x = 0; x < dimX; ++x) {
          const std::size_t voxelIndex =
              scalar_field_index(computeResult, x, y, z);
          if (computeResult.voxelCoverageCounts[voxelIndex] == 0) {
            continue;
          }
          add_cells_around_voxel_index(static_cast<uint32_t>(voxelIndex));
        }
      }
    }
  }
}

bool build_vulkan_direct_surface_buffers(
    const VulkanParticleComputeResult &computeResult,
    const VulkanSurfaceMeshResult &surfaceMeshResult,
    const std::vector<uint32_t> &activeCellIndices,
    const std::vector<uint32_t> &activeCellCubeIndices,
    std::vector<float> &outVertices,
    std::vector<int> &outFaces, std::string &outError) {
  outVertices.clear();
  outFaces.clear();
  if (surfaceMeshResult.triangleCounts.empty() ||
      surfaceMeshResult.totalTriangleCount == 0) {
    outError = "Vulkan direct surface mesh produced no triangles.";
    return false;
  }
  if (surfaceMeshResult.triangleCounts.size() != activeCellIndices.size() ||
      activeCellIndices.size() != activeCellCubeIndices.size()) {
    outError =
        "Vulkan direct surface mesh cell buffers do not match the emitted triangle buffers.";
    return false;
  }

  const std::size_t expectedVertexFloatCount =
      static_cast<std::size_t>(surfaceMeshResult.triangleCounts.size()) *
      15ull * 3ull;
  if (surfaceMeshResult.triangleVertices.size() != expectedVertexFloatCount) {
    outError =
        "Vulkan direct surface mesh vertex buffer size does not match the triangle layout.";
    return false;
  }

  const int domainDimX = computeResult.domainDimensions[0];
  const int domainDimY = computeResult.domainDimensions[1];
  const int domainDimZ = computeResult.domainDimensions[2];
  const int cellDimX = computeResult.domainDimensions[0] - 1;
  const int cellDimY = computeResult.domainDimensions[1] - 1;
  const std::size_t floatsPerCell = 15ull * 3ull;
  const float *triangleVertexData = surfaceMeshResult.triangleVertices.data();
  const bool built = build_vulkan_direct_surface_buffers_impl(
      domainDimX, domainDimY, domainDimZ, cellDimX, cellDimY,
      surfaceMeshResult.triangleCounts.size(), surfaceMeshResult.totalTriangleCount,
      activeCellIndices, activeCellCubeIndices,
      [&](std::size_t cellIndex) {
        return surfaceMeshResult.triangleCounts[cellIndex];
      },
      [&](std::size_t cellIndex) {
        return triangleVertexData + cellIndex * floatsPerCell;
      },
      outVertices, outFaces, outError);
  if (!built) {
    return false;
  }
  return validate_vulkan_surface_geometry(outVertices, outFaces,
                                          computeResult.scalarFieldMode,
                                          outError);
}

bool build_vulkan_direct_surface_buffers_from_resident_view(
    const VulkanParticleComputeResult &computeResult,
    const VulkanResidentSurfaceMeshView &surfaceMeshView,
    const std::vector<uint32_t> &activeCellIndices,
    const std::vector<uint32_t> &activeCellCubeIndices,
    std::vector<float> &outVertices, std::vector<int> &outFaces,
    std::string &outError) {
  outVertices.clear();
  outFaces.clear();
  if (surfaceMeshView.activeCellCount == 0u || surfaceMeshView.triangleCounts == nullptr ||
      surfaceMeshView.triangleVertices == nullptr) {
    outError = "Vulkan resident surface mesh view is unavailable.";
    return false;
  }
  if (static_cast<std::size_t>(surfaceMeshView.activeCellCount) !=
          activeCellIndices.size() ||
      activeCellIndices.size() != activeCellCubeIndices.size()) {
    outError =
        "Vulkan resident surface mesh cell buffers do not match the active-cell lists.";
    return false;
  }

  uint32_t totalTriangleCount = 0u;
  for (uint32_t i = 0; i < surfaceMeshView.activeCellCount; ++i) {
    const uint32_t triangleCount = surfaceMeshView.triangleCounts[i];
    if (triangleCount > 5u) {
      outError =
          "Vulkan resident surface mesh triangle count exceeds Marching Cubes limits.";
      return false;
    }
    totalTriangleCount += triangleCount;
  }
  if (totalTriangleCount == 0u) {
    outError = "Vulkan resident surface mesh produced no triangles.";
    return false;
  }

  const int domainDimX = computeResult.domainDimensions[0];
  const int domainDimY = computeResult.domainDimensions[1];
  const int domainDimZ = computeResult.domainDimensions[2];
  if (surfaceMeshView.triangleVerticesCompacted &&
      surfaceMeshView.triangleOffsets != nullptr &&
      surfaceMeshView.triangleEdgeIds != nullptr) {
    VulkanResidentSurfaceMeshView compactView = surfaceMeshView;
    compactView.totalTriangleCount = totalTriangleCount;
    const bool built = build_vulkan_direct_surface_buffers_from_compact_edge_ids(
        domainDimX, domainDimY, domainDimZ, compactView, outVertices, outFaces,
        outError);
    if (!built) {
      return false;
    }
    return validate_vulkan_surface_geometry(outVertices, outFaces,
                                            computeResult.scalarFieldMode,
                                            outError);
  }
  const int cellDimX = computeResult.domainDimensions[0] - 1;
  const int cellDimY = computeResult.domainDimensions[1] - 1;
  const std::size_t floatsPerCell = 15ull * 3ull;
  const float *triangleVertexData = surfaceMeshView.triangleVertices;
  std::vector<std::size_t> compactTriangleOffsets;
  if (surfaceMeshView.triangleVerticesCompacted) {
    compactTriangleOffsets.assign(activeCellIndices.size() + 1u, 0u);
    for (std::size_t cellIndex = 0; cellIndex < activeCellIndices.size();
         ++cellIndex) {
      compactTriangleOffsets[cellIndex + 1u] =
          compactTriangleOffsets[cellIndex] +
          static_cast<std::size_t>(surfaceMeshView.triangleCounts[cellIndex]) *
              9u;
    }
  }
  const bool built = build_vulkan_direct_surface_buffers_impl(
      domainDimX, domainDimY, domainDimZ, cellDimX, cellDimY,
      activeCellIndices.size(), totalTriangleCount, activeCellIndices,
      activeCellCubeIndices,
      [&](std::size_t cellIndex) { return surfaceMeshView.triangleCounts[cellIndex]; },
      [&](std::size_t cellIndex) {
        if (surfaceMeshView.triangleVerticesCompacted) {
          return triangleVertexData + compactTriangleOffsets[cellIndex];
        }
        return triangleVertexData + cellIndex * floatsPerCell;
      },
      outVertices, outFaces, outError);
  if (!built) {
    return false;
  }
  return validate_vulkan_surface_geometry(outVertices, outFaces,
                                          computeResult.scalarFieldMode,
                                          outError);
}

bool extract_vulkan_direct_surface_mesh(
    const VulkanParticleComputeResult &computeResult,
    const VulkanSurfaceMeshResult &surfaceMeshResult,
    const std::vector<uint32_t> &activeCellIndices,
    const std::vector<uint32_t> &activeCellCubeIndices,
    frantic::geometry::trimesh3 &outMesh, std::string &outError) {
  std::vector<float> flatVertices;
  std::vector<int> flatFaces;
  if (!build_vulkan_direct_surface_buffers(
          computeResult, surfaceMeshResult, activeCellIndices,
          activeCellCubeIndices, flatVertices, flatFaces, outError)) {
    return false;
  }

  const std::size_t vertexCount = flatVertices.size() / 3u;
  const std::size_t faceCount = flatFaces.size() / 3u;
  frantic::geometry::trimesh3 mesh;
  mesh.set_vertex_count(vertexCount);
  mesh.set_face_count(faceCount);

  tbb::parallel_for(tbb::blocked_range<size_t>(0, vertexCount),
                    [&](const tbb::blocked_range<size_t> &r) {
                      for (size_t i = r.begin(); i != r.end(); ++i) {
                        mesh.get_vertex(i) = frantic::graphics::vector3f(
                            flatVertices[i * 3 + 0], flatVertices[i * 3 + 1],
                            flatVertices[i * 3 + 2]);
                      }
                    });

  tbb::parallel_for(tbb::blocked_range<size_t>(0, faceCount),
                    [&](const tbb::blocked_range<size_t> &r) {
                      for (size_t i = r.begin(); i != r.end(); ++i) {
                        mesh.get_face(i) = frantic::graphics::vector3(
                            static_cast<float>(flatFaces[i * 3 + 0]),
                            static_cast<float>(flatFaces[i * 3 + 1]),
                            static_cast<float>(flatFaces[i * 3 + 2]));
                      }
                    });

  outMesh.swap(mesh);
  outError.clear();
  return true;
}

bool extract_vulkan_scalar_field_mesh(
    const VulkanParticleComputeResult &computeResult,
    frantic::geometry::trimesh3 &outMesh, std::string &outError,
    const std::vector<uint32_t> *preclassifiedCellIndices,
    const std::vector<uint32_t> *preclassifiedCubeIndices) {
  const int dimX = computeResult.domainDimensions[0];
  const int dimY = computeResult.domainDimensions[1];
  const int dimZ = computeResult.domainDimensions[2];
  const int cellDimX = dimX - 1;
  const int cellDimY = dimY - 1;
  const int cellDimZ = dimZ - 1;

  if (computeResult.voxelScalarField.empty()) {
    outError = "Vulkan scalar field is empty.";
    return false;
  }
  if (dimX < 2 || dimY < 2 || dimZ < 2) {
    outError = "Vulkan scalar field domain is too small for Marching Cubes.";
    return false;
  }

  const std::size_t expectedSamples =
      static_cast<std::size_t>(dimX) * static_cast<std::size_t>(dimY) *
      static_cast<std::size_t>(dimZ);
  if (computeResult.voxelScalarField.size() != expectedSamples) {
    outError = "Vulkan scalar field sample count does not match the planned domain.";
    return false;
  }

  const float isoValue = computeResult.surfaceIsoValue;
  std::vector<uint32_t> candidateCellIndices;
  std::vector<uint32_t> classifiedCubeIndices;
  const bool hasPreclassifiedCells =
      preclassifiedCellIndices != nullptr && preclassifiedCubeIndices != nullptr &&
      preclassifiedCellIndices->size() == preclassifiedCubeIndices->size() &&
      !preclassifiedCellIndices->empty();
  if (hasPreclassifiedCells) {
    candidateCellIndices = *preclassifiedCellIndices;
    classifiedCubeIndices = *preclassifiedCubeIndices;
  } else if (cellDimX > 0 && cellDimY > 0 && cellDimZ > 0) {
    build_vulkan_candidate_cell_indices(computeResult, isoValue,
                                        candidateCellIndices);
  }

  frantic::geometry::trimesh3 mesh;
  const std::size_t reservedCellCount =
      candidateCellIndices.empty()
          ? expectedSamples / 2
          : std::max<std::size_t>(candidateCellIndices.size(), 64);
  mesh.reserve_vertices(reservedCellCount * 3);
  mesh.reserve_faces(reservedCellCount * 2);
  std::unordered_map<GridEdgeKey, int, GridEdgeKeyHasher> edgeVertexMap;
  edgeVertexMap.reserve(reservedCellCount * 3);

  const bool refineEdgeVertices =
      should_refine_vulkan_isosurface_edge(computeResult);
  std::vector<float> exactCornerFieldCache;
  std::vector<uint8_t> exactCornerFieldValid;
  if (refineEdgeVertices) {
    exactCornerFieldCache.resize(expectedSamples, 0.0f);
    exactCornerFieldValid.assign(expectedSamples, 0);
  }

  const auto process_cell = [&](int x, int y, int z, int precomputedCubeIndex) {
    float cornerValues[8];
    std::size_t cornerPackedIndices[8];
    frantic::graphics::vector3f cornerPositions[8];

    for (int corner = 0; corner < 8; ++corner) {
      const int gx = x + kMarchingCubeCornerOffsets[corner][0];
      const int gy = y + kMarchingCubeCornerOffsets[corner][1];
      const int gz = z + kMarchingCubeCornerOffsets[corner][2];
      cornerPackedIndices[corner] = scalar_field_index(computeResult, gx, gy, gz);
      cornerValues[corner] =
          computeResult.voxelScalarField[cornerPackedIndices[corner]];
      cornerPositions[corner] =
          world_position_from_grid(computeResult, gx, gy, gz);
    }

    int cubeIndex = precomputedCubeIndex;
    if (cubeIndex < 0) {
      cubeIndex = 0;
      for (int corner = 0; corner < 8; ++corner) {
        if (cornerValues[corner] < isoValue) {
          cubeIndex |= (1 << corner);
        }
      }
    }

    const int edges = edgeTable[cubeIndex];
    if (edges == 0) {
      return;
    }

    int edgeVertexIndices[12];
    std::fill(std::begin(edgeVertexIndices), std::end(edgeVertexIndices), -1);

    for (int edge = 0; edge < 12; ++edge) {
      if ((edges & (1 << edge)) == 0) {
        continue;
      }

      const GridEdgeKey key = make_grid_edge_key(x, y, z, edge);
      auto it = edgeVertexMap.find(key);
      if (it != edgeVertexMap.end()) {
        edgeVertexIndices[edge] = it->second;
        continue;
      }

      const int c0 = kMarchingCubeEdgeCorners[edge][0];
      const int c1 = kMarchingCubeEdgeCorners[edge][1];
      const frantic::graphics::vector3f vertex =
          refineEdgeVertices
              ? refine_vulkan_isosurface_vertex(
                    computeResult, isoValue, cornerPackedIndices[c0],
                    cornerPackedIndices[c1], cornerPositions[c0],
                    cornerPositions[c1], cornerValues[c0], cornerValues[c1],
                    exactCornerFieldCache, exactCornerFieldValid)
              : interpolate_isosurface_vertex(
                    isoValue, cornerPositions[c0], cornerPositions[c1],
                    cornerValues[c0], cornerValues[c1]);
      const int vertexIndex = static_cast<int>(mesh.vertex_count());
      mesh.add_vertex(vertex);
      edgeVertexMap.emplace(key, vertexIndex);
      edgeVertexIndices[edge] = vertexIndex;
    }

    for (int i = 0; i < 15 && triTable[cubeIndex][i] != -1; i += 3) {
      const int e0 = triTable[cubeIndex][i];
      const int e1 = triTable[cubeIndex][i + 1];
      const int e2 = triTable[cubeIndex][i + 2];
      const int v0 = edgeVertexIndices[e0];
      const int v1 = edgeVertexIndices[e1];
      const int v2 = edgeVertexIndices[e2];
      if (v0 < 0 || v1 < 0 || v2 < 0 || v0 == v1 || v1 == v2 ||
          v0 == v2) {
        continue;
      }
      mesh.add_face(v0, v1, v2);
    }
  };

  if (!candidateCellIndices.empty()) {
    for (std::size_t cellListIndex = 0; cellListIndex < candidateCellIndices.size();
         ++cellListIndex) {
      const uint32_t packedCellIndex = candidateCellIndices[cellListIndex];
      const int z = static_cast<int>(
          packedCellIndex / static_cast<uint32_t>(cellDimX * cellDimY));
      const uint32_t rem =
          packedCellIndex -
          static_cast<uint32_t>(z * cellDimX * cellDimY);
      const int y = static_cast<int>(rem / static_cast<uint32_t>(cellDimX));
      const int x =
          static_cast<int>(rem - static_cast<uint32_t>(y * cellDimX));
      const int precomputedCubeIndex =
          cellListIndex < classifiedCubeIndices.size()
              ? static_cast<int>(classifiedCubeIndices[cellListIndex])
              : -1;
      process_cell(x, y, z, precomputedCubeIndex);
    }
  } else {
    for (int z = 0; z < dimZ - 1; ++z) {
      for (int y = 0; y < dimY - 1; ++y) {
        for (int x = 0; x < dimX - 1; ++x) {
          process_cell(x, y, z, -1);
        }
      }
    }
  }

  if (mesh.face_count() == 0) {
    outError = "Vulkan scalar field extraction produced no faces.";
    return false;
  }

  outMesh.swap(mesh);
  outError.clear();
  return true;
}

#if defined(FROST_ENABLE_VULKAN_PROBE)

struct VkInstance_T;
struct VkPhysicalDevice_T;
struct VkDevice_T;
struct VkAllocationCallbacks;
struct VkBuffer_T;
struct VkDeviceMemory_T;
struct VkQueue_T;
struct VkCommandPool_T;
struct VkCommandBuffer_T;
struct VkFence_T;
struct VkSemaphore_T;

using VkInstance = VkInstance_T *;
using VkPhysicalDevice = VkPhysicalDevice_T *;
using VkDevice = VkDevice_T *;
using VkBuffer = VkBuffer_T *;
using VkDeviceMemory = VkDeviceMemory_T *;
using VkQueue = VkQueue_T *;
using VkCommandPool = VkCommandPool_T *;
using VkCommandBuffer = VkCommandBuffer_T *;
using VkFence = VkFence_T *;
using VkSemaphore = VkSemaphore_T *;
using VkDeviceSize = uint64_t;
using VkFlags = uint32_t;
using VkQueueFlags = VkFlags;
using VkBufferUsageFlags = VkFlags;
using VkMemoryPropertyFlags = VkFlags;
using VkPipelineStageFlags = VkFlags;
using VkCommandPoolCreateFlags = VkFlags;
using VkCommandBufferUsageFlags = VkFlags;
using VkStructureType = int32_t;
using VkResult = int32_t;
using VkSharingMode = int32_t;
using VkCommandBufferLevel = int32_t;

constexpr VkResult VK_SUCCESS = 0;
constexpr VkResult VK_INCOMPLETE = 5;

constexpr VkStructureType VK_STRUCTURE_TYPE_APPLICATION_INFO = 0;
constexpr VkStructureType VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1;
constexpr VkStructureType VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = 2;
constexpr VkStructureType VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 3;
constexpr VkStructureType VK_STRUCTURE_TYPE_SUBMIT_INFO = 4;
constexpr VkStructureType VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = 5;
constexpr VkStructureType VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO = 12;
constexpr VkStructureType VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO = 39;
constexpr VkStructureType VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO = 40;
constexpr VkStructureType VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO = 42;

constexpr VkQueueFlags VK_QUEUE_COMPUTE_BIT = 0x00000002;
constexpr VkBufferUsageFlags VK_BUFFER_USAGE_TRANSFER_SRC_BIT = 0x00000001;
constexpr VkBufferUsageFlags VK_BUFFER_USAGE_TRANSFER_DST_BIT = 0x00000002;
constexpr VkBufferUsageFlags VK_BUFFER_USAGE_STORAGE_BUFFER_BIT = 0x00000020;
constexpr VkMemoryPropertyFlags VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT =
    0x00000002;
constexpr VkMemoryPropertyFlags VK_MEMORY_PROPERTY_HOST_COHERENT_BIT =
    0x00000004;
constexpr VkSharingMode VK_SHARING_MODE_EXCLUSIVE = 0;
constexpr VkCommandPoolCreateFlags
    VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT = 0x00000002;
constexpr VkCommandBufferLevel VK_COMMAND_BUFFER_LEVEL_PRIMARY = 0;
constexpr uint32_t VK_MAX_MEMORY_TYPES = 32;
constexpr uint32_t VK_MAX_MEMORY_HEAPS = 16;

constexpr uint32_t make_vk_api_version(uint32_t variant, uint32_t major,
                                       uint32_t minor, uint32_t patch) {
  return (variant << 29) | (major << 22) | (minor << 12) | patch;
}

constexpr uint32_t VK_API_VERSION_1_0 = make_vk_api_version(0, 1, 0, 0);

struct VkApplicationInfo {
  VkStructureType sType;
  const void *pNext;
  const char *pApplicationName;
  uint32_t applicationVersion;
  const char *pEngineName;
  uint32_t engineVersion;
  uint32_t apiVersion;
};

struct VkInstanceCreateInfo {
  VkStructureType sType;
  const void *pNext;
  VkFlags flags;
  const VkApplicationInfo *pApplicationInfo;
  uint32_t enabledLayerCount;
  const char *const *ppEnabledLayerNames;
  uint32_t enabledExtensionCount;
  const char *const *ppEnabledExtensionNames;
};

struct VkExtent3D {
  uint32_t width;
  uint32_t height;
  uint32_t depth;
};

struct VkQueueFamilyProperties {
  VkQueueFlags queueFlags;
  uint32_t queueCount;
  uint32_t timestampValidBits;
  VkExtent3D minImageTransferGranularity;
};

struct VkDeviceQueueCreateInfo {
  VkStructureType sType;
  const void *pNext;
  VkFlags flags;
  uint32_t queueFamilyIndex;
  uint32_t queueCount;
  const float *pQueuePriorities;
};

struct VkDeviceCreateInfo {
  VkStructureType sType;
  const void *pNext;
  VkFlags flags;
  uint32_t queueCreateInfoCount;
  const VkDeviceQueueCreateInfo *pQueueCreateInfos;
  uint32_t enabledLayerCount;
  const char *const *ppEnabledLayerNames;
  uint32_t enabledExtensionCount;
  const char *const *ppEnabledExtensionNames;
  const void *pEnabledFeatures;
};

struct VkBufferCreateInfo {
  VkStructureType sType;
  const void *pNext;
  VkFlags flags;
  VkDeviceSize size;
  VkBufferUsageFlags usage;
  VkSharingMode sharingMode;
  uint32_t queueFamilyIndexCount;
  const uint32_t *pQueueFamilyIndices;
};

struct VkMemoryRequirements {
  VkDeviceSize size;
  VkDeviceSize alignment;
  uint32_t memoryTypeBits;
};

struct VkMemoryAllocateInfo {
  VkStructureType sType;
  const void *pNext;
  VkDeviceSize allocationSize;
  uint32_t memoryTypeIndex;
};

struct VkMemoryType {
  VkMemoryPropertyFlags propertyFlags;
  uint32_t heapIndex;
};

struct VkMemoryHeap {
  VkDeviceSize size;
  VkFlags flags;
};

struct VkPhysicalDeviceMemoryProperties {
  uint32_t memoryTypeCount;
  VkMemoryType memoryTypes[VK_MAX_MEMORY_TYPES];
  uint32_t memoryHeapCount;
  VkMemoryHeap memoryHeaps[VK_MAX_MEMORY_HEAPS];
};

struct VkCommandPoolCreateInfo {
  VkStructureType sType;
  const void *pNext;
  VkCommandPoolCreateFlags flags;
  uint32_t queueFamilyIndex;
};

struct VkCommandBufferAllocateInfo {
  VkStructureType sType;
  const void *pNext;
  VkCommandPool commandPool;
  VkCommandBufferLevel level;
  uint32_t commandBufferCount;
};

struct VkCommandBufferBeginInfo {
  VkStructureType sType;
  const void *pNext;
  VkCommandBufferUsageFlags flags;
  const void *pInheritanceInfo;
};

struct VkBufferCopy {
  VkDeviceSize srcOffset;
  VkDeviceSize dstOffset;
  VkDeviceSize size;
};

struct VkSubmitInfo {
  VkStructureType sType;
  const void *pNext;
  uint32_t waitSemaphoreCount;
  const VkSemaphore *pWaitSemaphores;
  const VkPipelineStageFlags *pWaitDstStageMask;
  uint32_t commandBufferCount;
  const VkCommandBuffer *pCommandBuffers;
  uint32_t signalSemaphoreCount;
  const VkSemaphore *pSignalSemaphores;
};

using PFN_vkVoidFunction = void (*)();
using PFN_vkGetInstanceProcAddr = PFN_vkVoidFunction (*)(VkInstance instance,
                                                         const char *pName);
using PFN_vkEnumerateInstanceVersion = VkResult (*)(uint32_t *pApiVersion);
using PFN_vkCreateInstance =
    VkResult (*)(const VkInstanceCreateInfo *pCreateInfo,
                 const VkAllocationCallbacks *pAllocator,
                 VkInstance *pInstance);
using PFN_vkDestroyInstance = void (*)(VkInstance instance,
                                       const VkAllocationCallbacks *pAllocator);
using PFN_vkEnumeratePhysicalDevices =
    VkResult (*)(VkInstance instance, uint32_t *pPhysicalDeviceCount,
                 VkPhysicalDevice *pPhysicalDevices);
using PFN_vkGetPhysicalDeviceQueueFamilyProperties =
    void (*)(VkPhysicalDevice physicalDevice, uint32_t *pQueueFamilyPropertyCount,
             VkQueueFamilyProperties *pQueueFamilyProperties);
using PFN_vkGetPhysicalDeviceMemoryProperties =
    void (*)(VkPhysicalDevice physicalDevice,
             VkPhysicalDeviceMemoryProperties *pMemoryProperties);
using PFN_vkCreateDevice =
    VkResult (*)(VkPhysicalDevice physicalDevice,
                 const VkDeviceCreateInfo *pCreateInfo,
                 const VkAllocationCallbacks *pAllocator, VkDevice *pDevice);
using PFN_vkGetDeviceProcAddr = PFN_vkVoidFunction (*)(VkDevice device,
                                                       const char *pName);
using PFN_vkDestroyDevice = void (*)(VkDevice device,
                                     const VkAllocationCallbacks *pAllocator);
using PFN_vkGetDeviceQueue = void (*)(VkDevice device, uint32_t queueFamilyIndex,
                                      uint32_t queueIndex, VkQueue *pQueue);
using PFN_vkCreateBuffer =
    VkResult (*)(VkDevice device, const VkBufferCreateInfo *pCreateInfo,
                 const VkAllocationCallbacks *pAllocator, VkBuffer *pBuffer);
using PFN_vkDestroyBuffer = void (*)(VkDevice device, VkBuffer buffer,
                                     const VkAllocationCallbacks *pAllocator);
using PFN_vkGetBufferMemoryRequirements =
    void (*)(VkDevice device, VkBuffer buffer,
             VkMemoryRequirements *pMemoryRequirements);
using PFN_vkAllocateMemory =
    VkResult (*)(VkDevice device, const VkMemoryAllocateInfo *pAllocateInfo,
                 const VkAllocationCallbacks *pAllocator,
                 VkDeviceMemory *pMemory);
using PFN_vkFreeMemory = void (*)(VkDevice device, VkDeviceMemory memory,
                                  const VkAllocationCallbacks *pAllocator);
using PFN_vkBindBufferMemory =
    VkResult (*)(VkDevice device, VkBuffer buffer, VkDeviceMemory memory,
                 VkDeviceSize memoryOffset);
using PFN_vkMapMemory =
    VkResult (*)(VkDevice device, VkDeviceMemory memory, VkDeviceSize offset,
                 VkDeviceSize size, VkFlags flags, void **ppData);
using PFN_vkUnmapMemory = void (*)(VkDevice device, VkDeviceMemory memory);
using PFN_vkCreateCommandPool =
    VkResult (*)(VkDevice device, const VkCommandPoolCreateInfo *pCreateInfo,
                 const VkAllocationCallbacks *pAllocator,
                 VkCommandPool *pCommandPool);
using PFN_vkDestroyCommandPool =
    void (*)(VkDevice device, VkCommandPool commandPool,
             const VkAllocationCallbacks *pAllocator);
using PFN_vkAllocateCommandBuffers =
    VkResult (*)(VkDevice device,
                 const VkCommandBufferAllocateInfo *pAllocateInfo,
                 VkCommandBuffer *pCommandBuffers);
using PFN_vkBeginCommandBuffer =
    VkResult (*)(VkCommandBuffer commandBuffer,
                 const VkCommandBufferBeginInfo *pBeginInfo);
using PFN_vkEndCommandBuffer = VkResult (*)(VkCommandBuffer commandBuffer);
using PFN_vkCmdCopyBuffer =
    void (*)(VkCommandBuffer commandBuffer, VkBuffer srcBuffer,
             VkBuffer dstBuffer, uint32_t regionCount,
             const VkBufferCopy *pRegions);
using PFN_vkQueueSubmit =
    VkResult (*)(VkQueue queue, uint32_t submitCount,
                 const VkSubmitInfo *pSubmits, VkFence fence);
using PFN_vkQueueWaitIdle = VkResult (*)(VkQueue queue);

#if defined(_WIN32)
using LibraryHandle = HMODULE;

LibraryHandle load_vulkan_library() { return LoadLibraryW(L"vulkan-1.dll"); }

void close_vulkan_library(LibraryHandle handle) {
  if (handle) {
    FreeLibrary(handle);
  }
}

void *load_symbol(LibraryHandle handle, const char *name) {
  return handle ? reinterpret_cast<void *>(GetProcAddress(handle, name))
                : nullptr;
}
#else
using LibraryHandle = void *;

LibraryHandle load_vulkan_library() {
  return dlopen("libvulkan.so.1", RTLD_LAZY | RTLD_LOCAL);
}

void close_vulkan_library(LibraryHandle handle) {
  if (handle) {
    dlclose(handle);
  }
}

void *load_symbol(LibraryHandle handle, const char *name) {
  return handle ? dlsym(handle, name) : nullptr;
}
#endif

bool find_memory_type_index(const VkPhysicalDeviceMemoryProperties &properties,
                            uint32_t memoryTypeBits,
                            VkMemoryPropertyFlags requiredFlags,
                            uint32_t &outIndex) {
  for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
    const bool supportedByBuffer = (memoryTypeBits & (1u << i)) != 0;
    const bool hasRequiredFlags =
        (properties.memoryTypes[i].propertyFlags & requiredFlags) ==
        requiredFlags;
    if (supportedByBuffer && hasRequiredFlags) {
      outIndex = i;
      return true;
    }
  }
  return false;
}

struct PackedParticle {
  float x, y, z, radius;
};

bool roundtrip_particles_vulkan(const std::vector<PackedParticle> &inputParticles,
                                std::vector<PackedParticle> &outputParticles,
                                std::string &outError) {
  outputParticles.clear();

  if (inputParticles.empty()) {
    outError = "no particles were provided to the Vulkan backend";
    return false;
  }

  LibraryHandle library = nullptr;
  VkInstance instance = nullptr;
  VkDevice device = nullptr;
  VkBuffer inputBuffer = nullptr;
  VkBuffer outputBuffer = nullptr;
  VkDeviceMemory inputMemory = nullptr;
  VkDeviceMemory outputMemory = nullptr;
  VkCommandPool commandPool = nullptr;

  PFN_vkDestroyInstance vkDestroyInstance = nullptr;
  PFN_vkDestroyDevice vkDestroyDevice = nullptr;
  PFN_vkDestroyBuffer vkDestroyBuffer = nullptr;
  PFN_vkFreeMemory vkFreeMemory = nullptr;
  PFN_vkDestroyCommandPool vkDestroyCommandPool = nullptr;

  auto cleanup = [&]() {
    if (device) {
      if (commandPool && vkDestroyCommandPool) {
        vkDestroyCommandPool(device, commandPool, nullptr);
        commandPool = nullptr;
      }
      if (inputBuffer && vkDestroyBuffer) {
        vkDestroyBuffer(device, inputBuffer, nullptr);
        inputBuffer = nullptr;
      }
      if (outputBuffer && vkDestroyBuffer) {
        vkDestroyBuffer(device, outputBuffer, nullptr);
        outputBuffer = nullptr;
      }
      if (inputMemory && vkFreeMemory) {
        vkFreeMemory(device, inputMemory, nullptr);
        inputMemory = nullptr;
      }
      if (outputMemory && vkFreeMemory) {
        vkFreeMemory(device, outputMemory, nullptr);
        outputMemory = nullptr;
      }
      if (vkDestroyDevice) {
        vkDestroyDevice(device, nullptr);
      }
      device = nullptr;
    }

    if (instance && vkDestroyInstance) {
      vkDestroyInstance(instance, nullptr);
      instance = nullptr;
    }

    if (library) {
      close_vulkan_library(library);
      library = nullptr;
    }
  };

  auto fail = [&](const std::string &message) {
    outError = message;
    cleanup();
    return false;
  };

  library = load_vulkan_library();
  if (!library) {
    return fail("Vulkan runtime not found on this system.");
  }

  auto vkGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
      load_symbol(library, "vkGetInstanceProcAddr"));
  if (!vkGetInstanceProcAddr) {
    return fail("Vulkan loader found, but vkGetInstanceProcAddr is unavailable.");
  }

  auto vkEnumerateInstanceVersion =
      reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
          vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceVersion"));
  uint32_t apiVersion = VK_API_VERSION_1_0;
  if (vkEnumerateInstanceVersion) {
    vkEnumerateInstanceVersion(&apiVersion);
  }

  auto vkCreateInstance = reinterpret_cast<PFN_vkCreateInstance>(
      vkGetInstanceProcAddr(nullptr, "vkCreateInstance"));
  if (!vkCreateInstance) {
    return fail("Vulkan loader found, but vkCreateInstance is unavailable.");
  }

  VkApplicationInfo appInfo{};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "Frost for Blender";
  appInfo.applicationVersion = 1;
  appInfo.pEngineName = "Frost";
  appInfo.engineVersion = 1;
  appInfo.apiVersion = apiVersion;

  VkInstanceCreateInfo instanceInfo{};
  instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instanceInfo.pApplicationInfo = &appInfo;

  const VkResult createInstanceResult =
      vkCreateInstance(&instanceInfo, nullptr, &instance);
  if (createInstanceResult != VK_SUCCESS || !instance) {
    std::ostringstream stream;
    stream << "Vulkan instance creation failed (VkResult "
           << createInstanceResult << ").";
    return fail(stream.str());
  }

  vkDestroyInstance = reinterpret_cast<PFN_vkDestroyInstance>(
      vkGetInstanceProcAddr(instance, "vkDestroyInstance"));
  auto vkEnumeratePhysicalDevices =
      reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(
          vkGetInstanceProcAddr(instance, "vkEnumeratePhysicalDevices"));
  auto vkGetPhysicalDeviceQueueFamilyProperties =
      reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
          vkGetInstanceProcAddr(instance,
                                "vkGetPhysicalDeviceQueueFamilyProperties"));
  auto vkGetPhysicalDeviceMemoryProperties =
      reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(
          vkGetInstanceProcAddr(instance,
                                "vkGetPhysicalDeviceMemoryProperties"));
  auto vkCreateDevice = reinterpret_cast<PFN_vkCreateDevice>(
      vkGetInstanceProcAddr(instance, "vkCreateDevice"));
  auto vkGetDeviceProcAddr = reinterpret_cast<PFN_vkGetDeviceProcAddr>(
      vkGetInstanceProcAddr(instance, "vkGetDeviceProcAddr"));

  if (!vkDestroyInstance || !vkEnumeratePhysicalDevices ||
      !vkGetPhysicalDeviceQueueFamilyProperties ||
      !vkGetPhysicalDeviceMemoryProperties || !vkCreateDevice ||
      !vkGetDeviceProcAddr) {
    return fail("Vulkan instance created, but required device setup functions are missing.");
  }

  uint32_t physicalDeviceCount = 0;
  VkResult enumerateResult =
      vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, nullptr);
  if (enumerateResult != VK_SUCCESS && enumerateResult != VK_INCOMPLETE) {
    std::ostringstream stream;
    stream << "Vulkan physical device enumeration failed (VkResult "
           << enumerateResult << ").";
    return fail(stream.str());
  }

  if (physicalDeviceCount == 0) {
    return fail("Vulkan runtime detected, but no physical device was enumerated.");
  }

  std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
  enumerateResult = vkEnumeratePhysicalDevices(instance, &physicalDeviceCount,
                                               physicalDevices.data());
  if (enumerateResult != VK_SUCCESS && enumerateResult != VK_INCOMPLETE) {
    std::ostringstream stream;
    stream << "Vulkan physical device enumeration returned an error (VkResult "
           << enumerateResult << ").";
    return fail(stream.str());
  }

  VkPhysicalDevice selectedPhysicalDevice = nullptr;
  uint32_t selectedQueueFamilyIndex = 0xFFFFFFFFu;

  for (VkPhysicalDevice candidate : physicalDevices) {
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queueFamilyCount,
                                             nullptr);
    if (queueFamilyCount == 0) {
      continue;
    }

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queueFamilyCount,
                                             queueFamilies.data());

    for (uint32_t i = 0; i < queueFamilyCount; ++i) {
      if (queueFamilies[i].queueCount > 0 &&
          (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
        selectedPhysicalDevice = candidate;
        selectedQueueFamilyIndex = i;
        break;
      }
    }

    if (selectedPhysicalDevice) {
      break;
    }
  }

  if (!selectedPhysicalDevice || selectedQueueFamilyIndex == 0xFFFFFFFFu) {
    return fail("No Vulkan compute queue family was found for the Frost backend.");
  }

  const float queuePriority = 1.0f;
  VkDeviceQueueCreateInfo queueCreateInfo{};
  queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueCreateInfo.queueFamilyIndex = selectedQueueFamilyIndex;
  queueCreateInfo.queueCount = 1;
  queueCreateInfo.pQueuePriorities = &queuePriority;

  VkDeviceCreateInfo deviceCreateInfo{};
  deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  deviceCreateInfo.queueCreateInfoCount = 1;
  deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;

  const VkResult createDeviceResult = vkCreateDevice(
      selectedPhysicalDevice, &deviceCreateInfo, nullptr, &device);
  if (createDeviceResult != VK_SUCCESS || !device) {
    std::ostringstream stream;
    stream << "Vulkan logical device creation failed (VkResult "
           << createDeviceResult << ").";
    return fail(stream.str());
  }

  vkDestroyDevice = reinterpret_cast<PFN_vkDestroyDevice>(
      vkGetDeviceProcAddr(device, "vkDestroyDevice"));
  auto vkGetDeviceQueue = reinterpret_cast<PFN_vkGetDeviceQueue>(
      vkGetDeviceProcAddr(device, "vkGetDeviceQueue"));
  auto vkCreateBuffer = reinterpret_cast<PFN_vkCreateBuffer>(
      vkGetDeviceProcAddr(device, "vkCreateBuffer"));
  vkDestroyBuffer = reinterpret_cast<PFN_vkDestroyBuffer>(
      vkGetDeviceProcAddr(device, "vkDestroyBuffer"));
  auto vkGetBufferMemoryRequirements =
      reinterpret_cast<PFN_vkGetBufferMemoryRequirements>(
          vkGetDeviceProcAddr(device, "vkGetBufferMemoryRequirements"));
  auto vkAllocateMemory = reinterpret_cast<PFN_vkAllocateMemory>(
      vkGetDeviceProcAddr(device, "vkAllocateMemory"));
  vkFreeMemory = reinterpret_cast<PFN_vkFreeMemory>(
      vkGetDeviceProcAddr(device, "vkFreeMemory"));
  auto vkBindBufferMemory = reinterpret_cast<PFN_vkBindBufferMemory>(
      vkGetDeviceProcAddr(device, "vkBindBufferMemory"));
  auto vkMapMemory = reinterpret_cast<PFN_vkMapMemory>(
      vkGetDeviceProcAddr(device, "vkMapMemory"));
  auto vkUnmapMemory = reinterpret_cast<PFN_vkUnmapMemory>(
      vkGetDeviceProcAddr(device, "vkUnmapMemory"));
  auto vkCreateCommandPool = reinterpret_cast<PFN_vkCreateCommandPool>(
      vkGetDeviceProcAddr(device, "vkCreateCommandPool"));
  vkDestroyCommandPool = reinterpret_cast<PFN_vkDestroyCommandPool>(
      vkGetDeviceProcAddr(device, "vkDestroyCommandPool"));
  auto vkAllocateCommandBuffers =
      reinterpret_cast<PFN_vkAllocateCommandBuffers>(
          vkGetDeviceProcAddr(device, "vkAllocateCommandBuffers"));
  auto vkBeginCommandBuffer = reinterpret_cast<PFN_vkBeginCommandBuffer>(
      vkGetDeviceProcAddr(device, "vkBeginCommandBuffer"));
  auto vkEndCommandBuffer = reinterpret_cast<PFN_vkEndCommandBuffer>(
      vkGetDeviceProcAddr(device, "vkEndCommandBuffer"));
  auto vkCmdCopyBuffer = reinterpret_cast<PFN_vkCmdCopyBuffer>(
      vkGetDeviceProcAddr(device, "vkCmdCopyBuffer"));
  auto vkQueueSubmit = reinterpret_cast<PFN_vkQueueSubmit>(
      vkGetDeviceProcAddr(device, "vkQueueSubmit"));
  auto vkQueueWaitIdle = reinterpret_cast<PFN_vkQueueWaitIdle>(
      vkGetDeviceProcAddr(device, "vkQueueWaitIdle"));

  if (!vkDestroyDevice || !vkGetDeviceQueue || !vkCreateBuffer ||
      !vkDestroyBuffer || !vkGetBufferMemoryRequirements || !vkAllocateMemory ||
      !vkFreeMemory || !vkBindBufferMemory || !vkMapMemory || !vkUnmapMemory ||
      !vkCreateCommandPool || !vkDestroyCommandPool ||
      !vkAllocateCommandBuffers || !vkBeginCommandBuffer ||
      !vkEndCommandBuffer || !vkCmdCopyBuffer || !vkQueueSubmit ||
      !vkQueueWaitIdle) {
    return fail("Vulkan logical device created, but required transfer functions are missing.");
  }

  VkQueue queue = nullptr;
  vkGetDeviceQueue(device, selectedQueueFamilyIndex, 0, &queue);
  if (!queue) {
    return fail("Vulkan queue retrieval failed for the Frost backend.");
  }

  const VkDeviceSize bufferSize =
      static_cast<VkDeviceSize>(inputParticles.size() * sizeof(PackedParticle));

  auto create_buffer = [&](VkBufferUsageFlags usage, VkBuffer &buffer,
                           VkDeviceMemory &memory) -> bool {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = bufferSize;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    const VkResult createBufferResult =
        vkCreateBuffer(device, &bufferInfo, nullptr, &buffer);
    if (createBufferResult != VK_SUCCESS || !buffer) {
      std::ostringstream stream;
      stream << "Vulkan buffer creation failed (VkResult "
             << createBufferResult << ").";
      outError = stream.str();
      return false;
    }

    VkMemoryRequirements memoryRequirements{};
    vkGetBufferMemoryRequirements(device, buffer, &memoryRequirements);

    VkPhysicalDeviceMemoryProperties memoryProperties{};
    vkGetPhysicalDeviceMemoryProperties(selectedPhysicalDevice,
                                        &memoryProperties);

    uint32_t memoryTypeIndex = 0;
    if (!find_memory_type_index(memoryProperties,
                                memoryRequirements.memoryTypeBits,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                memoryTypeIndex)) {
      outError =
          "No host-visible coherent Vulkan memory type was found for the Frost backend.";
      return false;
    }

    VkMemoryAllocateInfo allocationInfo{};
    allocationInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocationInfo.allocationSize = memoryRequirements.size;
    allocationInfo.memoryTypeIndex = memoryTypeIndex;

    const VkResult allocateResult =
        vkAllocateMemory(device, &allocationInfo, nullptr, &memory);
    if (allocateResult != VK_SUCCESS || !memory) {
      std::ostringstream stream;
      stream << "Vulkan memory allocation failed (VkResult "
             << allocateResult << ").";
      outError = stream.str();
      return false;
    }

    const VkResult bindResult = vkBindBufferMemory(device, buffer, memory, 0);
    if (bindResult != VK_SUCCESS) {
      std::ostringstream stream;
      stream << "Vulkan buffer binding failed (VkResult " << bindResult
             << ").";
      outError = stream.str();
      return false;
    }

    return true;
  };

  if (!create_buffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     inputBuffer, inputMemory)) {
    return fail(outError);
  }

  if (!create_buffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     outputBuffer, outputMemory)) {
    return fail(outError);
  }

  void *mappedInput = nullptr;
  const VkResult mapInputResult =
      vkMapMemory(device, inputMemory, 0, bufferSize, 0, &mappedInput);
  if (mapInputResult != VK_SUCCESS || !mappedInput) {
    std::ostringstream stream;
    stream << "Vulkan input memory mapping failed (VkResult "
           << mapInputResult << ").";
    return fail(stream.str());
  }
  std::memcpy(mappedInput, inputParticles.data(),
              static_cast<size_t>(bufferSize));
  vkUnmapMemory(device, inputMemory);

  VkCommandPoolCreateInfo commandPoolInfo{};
  commandPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  commandPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  commandPoolInfo.queueFamilyIndex = selectedQueueFamilyIndex;

  const VkResult createCommandPoolResult =
      vkCreateCommandPool(device, &commandPoolInfo, nullptr, &commandPool);
  if (createCommandPoolResult != VK_SUCCESS || !commandPool) {
    std::ostringstream stream;
    stream << "Vulkan command pool creation failed (VkResult "
           << createCommandPoolResult << ").";
    return fail(stream.str());
  }

  VkCommandBufferAllocateInfo commandBufferInfo{};
  commandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  commandBufferInfo.commandPool = commandPool;
  commandBufferInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  commandBufferInfo.commandBufferCount = 1;

  VkCommandBuffer commandBuffer = nullptr;
  const VkResult allocateCommandBufferResult =
      vkAllocateCommandBuffers(device, &commandBufferInfo, &commandBuffer);
  if (allocateCommandBufferResult != VK_SUCCESS || !commandBuffer) {
    std::ostringstream stream;
    stream << "Vulkan command buffer allocation failed (VkResult "
           << allocateCommandBufferResult << ").";
    return fail(stream.str());
  }

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

  const VkResult beginResult = vkBeginCommandBuffer(commandBuffer, &beginInfo);
  if (beginResult != VK_SUCCESS) {
    std::ostringstream stream;
    stream << "Vulkan command buffer begin failed (VkResult " << beginResult
           << ").";
    return fail(stream.str());
  }

  VkBufferCopy copyRegion{};
  copyRegion.size = bufferSize;
  vkCmdCopyBuffer(commandBuffer, inputBuffer, outputBuffer, 1, &copyRegion);

  const VkResult endResult = vkEndCommandBuffer(commandBuffer);
  if (endResult != VK_SUCCESS) {
    std::ostringstream stream;
    stream << "Vulkan command buffer end failed (VkResult " << endResult
           << ").";
    return fail(stream.str());
  }

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;

  const VkResult submitResult = vkQueueSubmit(queue, 1, &submitInfo, nullptr);
  if (submitResult != VK_SUCCESS) {
    std::ostringstream stream;
    stream << "Vulkan queue submit failed (VkResult " << submitResult
           << ").";
    return fail(stream.str());
  }

  const VkResult waitResult = vkQueueWaitIdle(queue);
  if (waitResult != VK_SUCCESS) {
    std::ostringstream stream;
    stream << "Vulkan queue wait failed (VkResult " << waitResult << ").";
    return fail(stream.str());
  }

  void *mappedOutput = nullptr;
  const VkResult mapOutputResult =
      vkMapMemory(device, outputMemory, 0, bufferSize, 0, &mappedOutput);
  if (mapOutputResult != VK_SUCCESS || !mappedOutput) {
    std::ostringstream stream;
    stream << "Vulkan output memory mapping failed (VkResult "
           << mapOutputResult << ").";
    return fail(stream.str());
  }

  outputParticles.resize(inputParticles.size());
  std::memcpy(outputParticles.data(), mappedOutput,
              static_cast<size_t>(bufferSize));
  vkUnmapMemory(device, outputMemory);

  cleanup();
  outError.clear();
  return true;
}

class VulkanTransferGpuBackend : public FrostGpuBackend {
public:
  const char *name() const override { return "vulkan-experimental"; }

  bool is_available() const override {
    return get_vulkan_gpu_backend_info().available;
  }

  bool generate_mesh_buffers(const frantic::particles::particle_array &particles,
                             const frost_parameters &params,
                             const FrostGpuMeshingOptions &options,
                             std::vector<float> &outVertices,
                             std::vector<int> &outFaces,
                             std::string &outError) override {
    outVertices.clear();
    outFaces.clear();

    try {
      if (particles.size() == 0) {
        outError.clear();
        return false;
      }

      const auto &channelMap = particles.get_channel_map();
      if (!channelMap.has_channel(_T("Position")) ||
          !channelMap.has_channel(_T("Radius"))) {
        outError =
            "Vulkan backend requires Position and Radius channels on particles.";
        return false;
      }

      auto positionAccessor =
          channelMap.get_accessor<frantic::graphics::vector3f>(_T("Position"));
      auto radiusAccessor = channelMap.get_accessor<float>(_T("Radius"));
      const bool hasVelocityChannel = channelMap.has_channel(_T("Velocity"));

      const std::size_t particleCount = particles.size();
      std::vector<float> inputParticleFloats(particleCount * 4);
      std::vector<float> inputVelocityFloats;
      float maximumParticleRadius = 0.0f;
      float minimumParticleRadius = std::numeric_limits<float>::max();
      for (size_t i = 0; i < particleCount; ++i) {
        const char *particle = particles.at(i);
        const frantic::graphics::vector3f position =
            positionAccessor.get(particle);
        const float particleRadius = radiusAccessor.get(particle);
        inputParticleFloats[i * 4 + 0] = position.x;
        inputParticleFloats[i * 4 + 1] = position.y;
        inputParticleFloats[i * 4 + 2] = position.z;
        inputParticleFloats[i * 4 + 3] = particleRadius;
        if (particleRadius > 0.0f) {
          maximumParticleRadius =
              std::max(maximumParticleRadius, particleRadius);
          minimumParticleRadius =
              std::min(minimumParticleRadius, particleRadius);
        }
      }
      if (minimumParticleRadius == std::numeric_limits<float>::max()) {
        minimumParticleRadius = 0.0f;
      }

      if (hasVelocityChannel) {
        auto velocityAccessor =
            channelMap.get_accessor<frantic::graphics::vector3f>(_T("Velocity"));
        inputVelocityFloats.resize(particleCount * 4, 0.0f);
        for (size_t i = 0; i < particleCount; ++i) {
          const frantic::graphics::vector3f velocity =
              velocityAccessor.get(particles.at(i));
          inputVelocityFloats[i * 4 + 0] = velocity.x;
          inputVelocityFloats[i * 4 + 1] = velocity.y;
          inputVelocityFloats[i * 4 + 2] = velocity.z;
        }
      }

      const float voxelLength = compute_vulkan_meshing_voxel_length(
          params, maximumParticleRadius, particles.size());
      const float planningRadiusScale = compute_vulkan_effect_radius_scale(
          params, voxelLength, minimumParticleRadius);
      VulkanParticleComputeSettings computeSettings =
          build_vulkan_compute_settings(params, planningRadiusScale,
                                        voxelLength, maximumParticleRadius);
      computeSettings.allowCpuCompactedPairs = true;
      computeSettings.preferGpuCompactedPairs = true;
      computeSettings.readbackCoverageCounts = true;
      computeSettings.readbackScalarField = false;
      computeSettings.retainParticleData = false;
      using Clock = std::chrono::steady_clock;
      const auto directPathStart = Clock::now();
      auto phaseStart = directPathStart;
      double computeSeconds = 0.0;
      double classifySeconds = 0.0;
      double emitSeconds = 0.0;
      double packSeconds = 0.0;

      VulkanParticleComputeResult computeResult;
      if (!run_frost_vulkan_compute_particles(
              inputParticleFloats.data(),
              inputVelocityFloats.empty() ? nullptr
                                          : inputVelocityFloats.data(),
              particleCount, computeSettings, computeResult,
              outError)) {
        return false;
      }
      computeSeconds = std::chrono::duration<double>(Clock::now() - phaseStart).count();
      phaseStart = Clock::now();

      if (options.showDebugLog) {
        const uint64_t estimatedVoxelCount =
            static_cast<uint64_t>(computeResult.domainDimensions[0]) *
            static_cast<uint64_t>(computeResult.domainDimensions[1]) *
            static_cast<uint64_t>(computeResult.domainDimensions[2]);
        if (computeResult.activeVoxelCount > 0) {
          std::cout << "[Vulkan] GPU active-voxel compaction prepared "
                    << computeResult.activeVoxelCount << " active voxels out of "
                    << estimatedVoxelCount << ", max particle overlaps per voxel "
                    << computeResult.maxVoxelCoverage
                    << ", candidate voxel-particle pairs "
                    << computeResult.coveredParticleVoxelPairs << "."
                    << std::endl;
        }
      }

      if (computeResult.voxelScalarField.empty() &&
          !computeResult.scalarFieldResidentOnGpu) {
        outError = "Vulkan direct mesh path requires a scalar field.";
        return false;
      }

      std::vector<uint32_t> gpuSurfaceCellIndices;
      std::vector<uint32_t> gpuSurfaceCellCubeIndices;
      VulkanSurfaceMeshResult gpuSurfaceMesh;

      const int64_t denseCellDimX =
          static_cast<int64_t>(computeResult.domainDimensions[0]) - 1ll;
      const int64_t denseCellDimY =
          static_cast<int64_t>(computeResult.domainDimensions[1]) - 1ll;
      const int64_t denseCellDimZ =
          static_cast<int64_t>(computeResult.domainDimensions[2]) - 1ll;
      const uint64_t denseCellCount =
          (denseCellDimX > 0 && denseCellDimY > 0 && denseCellDimZ > 0)
              ? static_cast<uint64_t>(denseCellDimX) *
                    static_cast<uint64_t>(denseCellDimY) *
                    static_cast<uint64_t>(denseCellDimZ)
              : 0ull;

      bool usedDenseGpuSurfacePath = false;
      bool usedSparseGpuCandidatePath = false;
      std::string denseSurfaceError;
      if (denseCellCount > 0 &&
          denseCellCount <= kMaxDenseVulkanSurfaceCells &&
          run_frost_vulkan_generate_dense_surface_mesh(
              computeResult, gpuSurfaceCellIndices, gpuSurfaceCellCubeIndices,
              gpuSurfaceMesh, denseSurfaceError)) {
        usedDenseGpuSurfacePath = true;
      } else if (options.showDebugLog && !denseSurfaceError.empty()) {
        std::cout << "[Vulkan] Dense direct surface path unavailable, "
                     "falling back to sparse GPU surface path: "
                  << denseSurfaceError << std::endl;
      }

      if (!usedDenseGpuSurfacePath) {
        std::string sparseGpuCandidateError;
        if (denseCellCount > 0 &&
            denseCellCount <= kMaxGpuSparseCandidateScanCells &&
            computeResult.activeVoxelIndices.size() >=
                kMinGpuSparseCandidateVoxels &&
            run_frost_vulkan_classify_surface_cells_from_active_voxels(
                computeResult, gpuSurfaceCellIndices, gpuSurfaceCellCubeIndices,
                sparseGpuCandidateError)) {
          if (!gpuSurfaceCellIndices.empty()) {
            usedSparseGpuCandidatePath = true;
          } else {
            gpuSurfaceCellIndices.clear();
            gpuSurfaceCellCubeIndices.clear();
          }
        } else if (options.showDebugLog &&
                   !sparseGpuCandidateError.empty()) {
          std::cout << "[Vulkan] Sparse GPU candidate-cell path unavailable, "
                       "falling back to CPU candidate preparation: "
                    << sparseGpuCandidateError << std::endl;
        }

        if (!usedSparseGpuCandidatePath) {
          if (computeResult.voxelScalarField.empty() &&
              !ensure_frost_vulkan_scalar_field_readback(computeResult,
                                                         outError)) {
            return false;
          }
          std::vector<uint32_t> candidateCellIndices;
          build_vulkan_candidate_cell_indices(
              computeResult, computeResult.surfaceIsoValue, candidateCellIndices);
          if (candidateCellIndices.empty()) {
            outError = "Vulkan direct mesh path found no candidate cells.";
            return false;
          }

          if (!run_frost_vulkan_classify_surface_cells(
                  computeResult, candidateCellIndices, gpuSurfaceCellIndices,
                  gpuSurfaceCellCubeIndices, outError)) {
            return false;
          }
        }
        const double sparseSurfacePhaseSeconds =
            std::chrono::duration<double>(Clock::now() - phaseStart).count();
        classifySeconds = sparseSurfacePhaseSeconds;
        phaseStart = Clock::now();
        if (gpuSurfaceCellIndices.empty()) {
          outError = "Vulkan direct mesh path found no active surface cells.";
          return false;
        }

        uint32_t classifiedTriangleCount = 0u;
        for (uint32_t cubeIndex : gpuSurfaceCellCubeIndices) {
          classifiedTriangleCount += count_marching_cubes_triangles(cubeIndex);
        }

        if (exceeds_resident_raw_surface_budget(
                computeResult.domainDimensions[0],
                computeResult.domainDimensions[1],
                computeResult.domainDimensions[2],
                static_cast<uint32_t>(gpuSurfaceCellIndices.size()),
                classifiedTriangleCount)) {
          if (options.showDebugLog) {
            std::cout << "[Vulkan] Skipping raw direct GPU mesh path: "
                      << gpuSurfaceCellIndices.size() << " active cells, "
                      << classifiedTriangleCount
                      << " triangles exceed the current safe stability budget."
                      << std::endl;
          }
          outError =
              "Vulkan raw direct surface path exceeds the current safe "
              "stability budget for this scene.";
          return false;
        }

        if (usedSparseGpuCandidatePath) {
          std::vector<uint32_t> residentTriangleCounts(
              gpuSurfaceCellCubeIndices.size(), 0u);
          std::vector<uint32_t> residentTriangleOffsets(
              gpuSurfaceCellCubeIndices.size(), 0u);
          uint32_t residentTotalTriangleCount = 0u;
          for (std::size_t cellIndex = 0;
               cellIndex < gpuSurfaceCellCubeIndices.size(); ++cellIndex) {
            residentTriangleOffsets[cellIndex] = residentTotalTriangleCount;
            const uint32_t triangleCount = count_marching_cubes_triangles(
                gpuSurfaceCellCubeIndices[cellIndex]);
            residentTriangleCounts[cellIndex] = triangleCount;
            residentTotalTriangleCount += triangleCount;
          }

          if (residentTotalTriangleCount == 0u) {
            outError =
                "Vulkan direct mesh path found no resident surface triangles.";
            return false;
          }

          if (exceeds_resident_raw_surface_budget(
                  computeResult.domainDimensions[0],
                  computeResult.domainDimensions[1],
                  computeResult.domainDimensions[2],
                  static_cast<uint32_t>(gpuSurfaceCellIndices.size()),
                  residentTotalTriangleCount)) {
            if (options.showDebugLog) {
              std::cout << "[Vulkan] Skipping raw resident direct surface path: "
                        << gpuSurfaceCellIndices.size() << " active cells, "
                        << residentTotalTriangleCount
                        << " triangles exceed the current safe stability "
                           "budget."
                        << std::endl;
            }
            outError =
                "Vulkan raw direct surface path exceeds the current safe "
                "stability budget for this scene.";
            return false;
          }

          const auto compactSurfacePhaseStart = Clock::now();
          if (!run_frost_vulkan_generate_compact_surface_vertices_from_resident_cells(
                  computeResult,
                  static_cast<uint32_t>(gpuSurfaceCellIndices.size()),
                  residentTriangleOffsets, residentTotalTriangleCount,
                  outError)) {
            return false;
          }
          emitSeconds = std::chrono::duration<double>(Clock::now() -
                                                      compactSurfacePhaseStart)
                            .count();

          const float *residentCompactTriangleVertices = nullptr;
          const uint32_t *residentCompactTriangleEdgeIds = nullptr;
          if (!get_frost_vulkan_resident_compact_surface_vertex_view(
                  residentTotalTriangleCount, residentCompactTriangleVertices,
                  outError)) {
            return false;
          }
          if (!get_frost_vulkan_resident_compact_surface_edge_id_view(
                  residentTotalTriangleCount, residentCompactTriangleEdgeIds,
                  outError)) {
            return false;
          }
          VulkanResidentSurfaceMeshView residentSurfaceView;
          residentSurfaceView.triangleCounts = residentTriangleCounts.data();
          residentSurfaceView.triangleVertices = residentCompactTriangleVertices;
          residentSurfaceView.triangleOffsets = residentTriangleOffsets.data();
          residentSurfaceView.triangleEdgeIds = residentCompactTriangleEdgeIds;
          residentSurfaceView.activeCellCount =
              static_cast<uint32_t>(gpuSurfaceCellIndices.size());
          residentSurfaceView.totalTriangleCount = residentTotalTriangleCount;
          residentSurfaceView.triangleVerticesCompacted = true;
          const auto buildSurfacePhaseStart = Clock::now();
          if (!build_vulkan_direct_surface_buffers_from_resident_view(
                  computeResult, residentSurfaceView, gpuSurfaceCellIndices,
                  gpuSurfaceCellCubeIndices, outVertices, outFaces, outError)) {
            return false;
          }
          packSeconds = std::chrono::duration<double>(Clock::now() -
                                                      buildSurfacePhaseStart)
                            .count();
        } else if (!run_frost_vulkan_generate_surface_mesh(
                       computeResult, gpuSurfaceCellIndices,
                       gpuSurfaceCellCubeIndices, gpuSurfaceMesh, outError)) {
          return false;
        }
        emitSeconds =
            std::chrono::duration<double>(Clock::now() - phaseStart).count();
        phaseStart = Clock::now();
        if (!usedSparseGpuCandidatePath &&
            !build_vulkan_direct_surface_buffers(
                computeResult, gpuSurfaceMesh, gpuSurfaceCellIndices,
                gpuSurfaceCellCubeIndices, outVertices, outFaces, outError)) {
          return false;
        }
        if (!usedSparseGpuCandidatePath) {
          packSeconds =
              std::chrono::duration<double>(Clock::now() - phaseStart).count();
        }
      } else {
        classifySeconds =
            std::chrono::duration<double>(Clock::now() - phaseStart).count();
        phaseStart = Clock::now();
      }

      if (usedDenseGpuSurfacePath &&
          !build_vulkan_direct_surface_buffers(
              computeResult, gpuSurfaceMesh, gpuSurfaceCellIndices,
              gpuSurfaceCellCubeIndices, outVertices, outFaces, outError)) {
        return false;
      }
      if (usedDenseGpuSurfacePath) {
        packSeconds =
            std::chrono::duration<double>(Clock::now() - phaseStart).count();
      }

      if (options.showDebugLog) {
        if (usedDenseGpuSurfacePath) {
          std::cout << "[Vulkan] Dense direct GPU surface path emitted "
                    << gpuSurfaceMesh.totalTriangleCount << " triangles across "
                    << gpuSurfaceCellIndices.size() << " active cells."
                    << std::endl;
        } else if (usedSparseGpuCandidatePath) {
          std::cout << "[Vulkan] Sparse GPU candidate-cell path prepared "
                    << gpuSurfaceCellIndices.size()
                    << " active cells before direct GPU triangle emission."
                    << std::endl;
        }
        std::cout << "[Vulkan] Direct GPU raw buffer path returned "
                  << (outVertices.size() / 3u) << " vertices and "
                  << (outFaces.size() / 3u) << " faces." << std::endl;
        const double totalSeconds =
            std::chrono::duration<double>(Clock::now() - directPathStart).count();
        std::cout << "[Vulkan] Direct path timings: compute=" << computeSeconds
                  << "s, classify=" << classifySeconds << "s, emit="
                  << emitSeconds << "s, pack=" << packSeconds
                  << "s, total=" << totalSeconds << "s." << std::endl;
      }

      outError.clear();
      return true;
    } catch (const std::exception &e) {
      outVertices.clear();
      outFaces.clear();
      outError = e.what();
      return false;
    } catch (...) {
      outVertices.clear();
      outFaces.clear();
      outError = "unknown exception in Vulkan raw mesh path";
      return false;
    }
  }

  bool generate_mesh(const frantic::particles::particle_array &particles,
                     const frost_parameters &params,
                     const FrostGpuMeshingOptions &options,
                     frantic::geometry::trimesh3 &outMesh,
                     std::string &outError) override {
    try {
      if (particles.size() == 0) {
        outError.clear();
        return false;
      }

      const auto &channelMap = particles.get_channel_map();
      if (!channelMap.has_channel(_T("Position")) ||
          !channelMap.has_channel(_T("Radius"))) {
        outError =
            "Vulkan backend requires Position and Radius channels on particles.";
        return false;
      }

      auto positionAccessor =
          channelMap.get_accessor<frantic::graphics::vector3f>(_T("Position"));
      auto radiusAccessor = channelMap.get_accessor<float>(_T("Radius"));
      const bool hasVelocityChannel = channelMap.has_channel(_T("Velocity"));

      std::vector<PackedParticle> inputParticles(particles.size());
      std::vector<float> inputParticleFloats(particles.size() * 4);
      std::vector<float> inputVelocityFloats;
      float maximumParticleRadius = 0.0f;
      float minimumParticleRadius = std::numeric_limits<float>::max();
      for (size_t i = 0; i < particles.size(); ++i) {
        const char *particle = particles.at(i);
        const frantic::graphics::vector3f position =
            positionAccessor.get(particle);
        const float particleRadius = radiusAccessor.get(particle);
        inputParticles[i] = {position.x, position.y, position.z,
                             particleRadius};
        inputParticleFloats[i * 4 + 0] = position.x;
        inputParticleFloats[i * 4 + 1] = position.y;
        inputParticleFloats[i * 4 + 2] = position.z;
        inputParticleFloats[i * 4 + 3] = particleRadius;
        if (particleRadius > 0.0f) {
          maximumParticleRadius =
              std::max(maximumParticleRadius, particleRadius);
          minimumParticleRadius =
              std::min(minimumParticleRadius, particleRadius);
        }
      }
      if (minimumParticleRadius == std::numeric_limits<float>::max()) {
        minimumParticleRadius = 0.0f;
      }

      if (hasVelocityChannel) {
        auto velocityAccessor =
            channelMap.get_accessor<frantic::graphics::vector3f>(_T("Velocity"));
        inputVelocityFloats.resize(particles.size() * 4, 0.0f);
        for (size_t i = 0; i < particles.size(); ++i) {
          const frantic::graphics::vector3f velocity =
              velocityAccessor.get(particles.at(i));
          inputVelocityFloats[i * 4 + 0] = velocity.x;
          inputVelocityFloats[i * 4 + 1] = velocity.y;
          inputVelocityFloats[i * 4 + 2] = velocity.z;
          inputVelocityFloats[i * 4 + 3] = 0.0f;
        }
      }

      const float voxelLength = compute_vulkan_meshing_voxel_length(
          params, maximumParticleRadius, particles.size());
      const float planningRadiusScale = compute_vulkan_effect_radius_scale(
          params, voxelLength, minimumParticleRadius);
      const VulkanParticleComputeSettings computeSettings =
          build_vulkan_compute_settings(params, planningRadiusScale,
                                        voxelLength, maximumParticleRadius);

      std::vector<PackedParticle> outputParticles;
      VulkanParticleComputeResult computeResult;
      std::string computeError;
      bool usedComputeShader = false;
      bool usedExperimentalFieldMesh = false;
      if (run_frost_vulkan_compute_particles(inputParticleFloats.data(),
                                             inputVelocityFloats.empty()
                                                 ? nullptr
                                                 : inputVelocityFloats.data(),
                                             inputParticles.size(),
                                             computeSettings,
                                             computeResult,
                                             computeError)) {
        usedComputeShader = true;
        outputParticles.resize(inputParticles.size());
        for (size_t i = 0; i < outputParticles.size(); ++i) {
          outputParticles[i] = {computeResult.particles[i * 4 + 0],
                                computeResult.particles[i * 4 + 1],
                                computeResult.particles[i * 4 + 2],
                                computeResult.particles[i * 4 + 3]};
        }
        if (options.showDebugLog) {
          std::cout << "[Vulkan] Compute shader dispatch succeeded on "
                    << outputParticles.size() << " particles." << std::endl;
          int32_t minVoxelX = 0;
          int32_t minVoxelY = 0;
          int32_t minVoxelZ = 0;
          int32_t maxVoxelXExclusive = 0;
          int32_t maxVoxelYExclusive = 0;
          int32_t maxVoxelZExclusive = 0;
          if (reduce_vulkan_voxel_bounds(computeResult, outputParticles.size(),
                                         minVoxelX, minVoxelY, minVoxelZ,
                                         maxVoxelXExclusive, maxVoxelYExclusive,
                                         maxVoxelZExclusive)) {
            const int64_t voxelDimX =
                static_cast<int64_t>(maxVoxelXExclusive) - minVoxelX;
            const int64_t voxelDimY =
                static_cast<int64_t>(maxVoxelYExclusive) - minVoxelY;
            const int64_t voxelDimZ =
                static_cast<int64_t>(maxVoxelZExclusive) - minVoxelZ;
            const uint64_t estimatedVoxelCount =
                (voxelDimX > 0 && voxelDimY > 0 && voxelDimZ > 0)
                    ? static_cast<uint64_t>(voxelDimX) *
                          static_cast<uint64_t>(voxelDimY) *
                          static_cast<uint64_t>(voxelDimZ)
                    : 0ull;
            std::cout << "[Vulkan] Planned voxel domain: min(" << minVoxelX
                      << ", " << minVoxelY << ", " << minVoxelZ << "), max("
                      << maxVoxelXExclusive << ", " << maxVoxelYExclusive
                      << ", " << maxVoxelZExclusive << "), dims(" << voxelDimX
                      << " x " << voxelDimY << " x " << voxelDimZ
                      << "), voxel length " << computeResult.voxelLength
                      << ", effect radius scale "
                      << computeResult.planningRadiusScale
                      << ", estimated voxels " << estimatedVoxelCount << "."
                      << std::endl;
            if (computeResult.activeVoxelCount > 0 ||
                !computeResult.voxelCoverageCounts.empty()) {
              std::cout << "[Vulkan] Voxel coverage buffer populated: "
                        << computeResult.activeVoxelCount
                        << " active voxels out of "
                        << estimatedVoxelCount
                        << ", max particle overlaps per voxel "
                        << computeResult.maxVoxelCoverage << "." << std::endl;
              if (computeResult.activeVoxelCount > 0) {
                std::cout << "[Vulkan] Scalar field compaction: "
                          << computeResult.activeVoxelCount
                          << " active voxels dispatched, "
                          << computeResult.coveredParticleVoxelPairs
                          << " voxel-particle candidate pairs evaluated."
                          << std::endl;
                if (!computeResult.activeVoxelParticleIndices.empty() &&
                    !computeResult.activeVoxelParticleOffsets.empty()) {
                  std::cout << "[Vulkan] Compacted voxel-particle table: "
                            << computeResult.activeVoxelParticleIndices.size()
                            << " pairs across "
                            << (computeResult.activeVoxelParticleOffsets.size() - 1)
                            << " active voxels." << std::endl;
                }
              }
            }
            if (!computeResult.voxelScalarField.empty()) {
              float minFieldValue = computeResult.voxelScalarField[0];
              float maxFieldValue = computeResult.voxelScalarField[0];
              std::size_t negativeVoxelCount = 0;
              for (float value : computeResult.voxelScalarField) {
                minFieldValue = std::min(minFieldValue, value);
                maxFieldValue = std::max(maxFieldValue, value);
                if (value < 0.0f) {
                  ++negativeVoxelCount;
                }
              }
              std::cout << "[Vulkan] Scalar field buffer populated: "
                        << computeResult.voxelScalarField.size()
                        << " cells, " << negativeVoxelCount
                        << " negative cells, field range [" << minFieldValue
                        << ", " << maxFieldValue << "], mode "
                        << get_vulkan_scalar_field_mode_name(
                               computeResult.scalarFieldMode)
                        << ", field radius scale "
                        << computeResult.fieldRadiusScale
                        << ", field threshold "
                        << computeResult.fieldThreshold
                        << ", extraction iso " << computeResult.surfaceIsoValue
                        << "." << std::endl;
            }
          }
        }
      } else {
        if (options.showDebugLog && !computeError.empty()) {
          std::cout << "[Vulkan] Compute shader path unavailable, falling back "
                       "to transfer-only round-trip: "
                    << computeError << std::endl;
        }
        if (!roundtrip_particles_vulkan(inputParticles, outputParticles,
                                        outError)) {
          return false;
        }
      }

      if (outputParticles.size() != inputParticles.size()) {
        outError =
            "Vulkan backend returned a particle count mismatch after round-trip.";
        return false;
      }

      if (usedComputeShader && !computeResult.voxelScalarField.empty()) {
        std::string fieldMeshError;
        frantic::geometry::trimesh3 fieldMesh;
        std::vector<uint32_t> gpuSurfaceCellIndices;
        std::vector<uint32_t> gpuSurfaceCellCubeIndices;
        std::vector<uint32_t> candidateCellIndices;
        build_vulkan_candidate_cell_indices(
            computeResult, computeResult.surfaceIsoValue, candidateCellIndices);
        if (options.showDebugLog) {
          std::cout << "[Vulkan] Prepared " << candidateCellIndices.size()
                    << " candidate Marching Cubes cells for GPU classification."
                    << std::endl;
        }
        std::string surfaceClassificationError;
        const bool classifiedSurfaceCells =
            !candidateCellIndices.empty() &&
            run_frost_vulkan_classify_surface_cells(
                computeResult, candidateCellIndices, gpuSurfaceCellIndices,
                gpuSurfaceCellCubeIndices, surfaceClassificationError);
        if (classifiedSurfaceCells && options.showDebugLog) {
          std::size_t cubeMismatchCount = 0;
          const int cellDimX = computeResult.domainDimensions[0] - 1;
          const int cellDimY = computeResult.domainDimensions[1] - 1;
          for (std::size_t i = 0; i < gpuSurfaceCellIndices.size(); ++i) {
            const uint32_t packedCellIndex = gpuSurfaceCellIndices[i];
            const int z = static_cast<int>(
                packedCellIndex / static_cast<uint32_t>(cellDimX * cellDimY));
            const uint32_t rem =
                packedCellIndex -
                static_cast<uint32_t>(z * cellDimX * cellDimY);
            const int y =
                static_cast<int>(rem / static_cast<uint32_t>(cellDimX));
            const int x =
                static_cast<int>(rem - static_cast<uint32_t>(y * cellDimX));
            const uint32_t cpuCubeIndex = static_cast<uint32_t>(
                compute_vulkan_cell_cube_index(computeResult, x, y, z,
                                               computeResult.surfaceIsoValue));
            if (cpuCubeIndex != gpuSurfaceCellCubeIndices[i]) {
              ++cubeMismatchCount;
            }
          }
          std::cout << "[Vulkan] Surface-cell classification cube-index "
                       "mismatches against CPU check: "
                    << cubeMismatchCount << "." << std::endl;
        }
        if (classifiedSurfaceCells && options.showDebugLog) {
          std::cout << "[Vulkan] Surface-cell classification produced "
                    << gpuSurfaceCellIndices.size()
                    << " active Marching Cubes cells from "
                    << candidateCellIndices.size() << " candidates."
                    << std::endl;
        } else if (!classifiedSurfaceCells && options.showDebugLog &&
                   !surfaceClassificationError.empty()) {
          std::cout << "[Vulkan] Surface-cell classification unavailable, "
                       "falling back to CPU-side cell classification: "
                    << surfaceClassificationError << std::endl;
        }
        std::string directFieldMeshError;
        if (classifiedSurfaceCells) {
          uint32_t classifiedTriangleCount = 0u;
          for (uint32_t cubeIndex : gpuSurfaceCellCubeIndices) {
            classifiedTriangleCount += count_marching_cubes_triangles(cubeIndex);
          }

          if (exceeds_resident_raw_surface_budget(
                  computeResult.domainDimensions[0],
                  computeResult.domainDimensions[1],
                  computeResult.domainDimensions[2],
                  static_cast<uint32_t>(gpuSurfaceCellIndices.size()),
                  classifiedTriangleCount)) {
            if (options.showDebugLog) {
              std::cout << "[Vulkan] Skipping direct GPU surface extraction: "
                        << gpuSurfaceCellIndices.size() << " active cells, "
                        << classifiedTriangleCount
                        << " triangles exceed the current safe stability "
                           "budget."
                        << std::endl;
            }
            directFieldMeshError =
                "direct Vulkan surface extraction skipped because the scene "
                "exceeds the current safe stability budget";
          } else {
            VulkanSurfaceMeshResult gpuSurfaceMesh;
            if (run_frost_vulkan_generate_surface_mesh(
                    computeResult, gpuSurfaceCellIndices,
                    gpuSurfaceCellCubeIndices, gpuSurfaceMesh,
                    directFieldMeshError) &&
                extract_vulkan_direct_surface_mesh(
                    computeResult, gpuSurfaceMesh, gpuSurfaceCellIndices,
                    gpuSurfaceCellCubeIndices, fieldMesh,
                    directFieldMeshError)) {
              outMesh.swap(fieldMesh);
              usedExperimentalFieldMesh = true;
              if (options.showDebugLog) {
                std::cout << "[Vulkan] Direct GPU surface extraction "
                             "succeeded with "
                          << outMesh.vertex_count() << " vertices and "
                          << outMesh.face_count() << " faces from "
                          << gpuSurfaceMesh.totalTriangleCount
                          << " emitted triangles." << std::endl;
              }
            }
          }

          if (!usedExperimentalFieldMesh && options.showDebugLog &&
              !directFieldMeshError.empty()) {
            std::cout << "[Vulkan] Direct GPU surface extraction unavailable, "
                         "falling back to CPU-side surface assembly: "
                      << directFieldMeshError << std::endl;
          }
        }
        if (!usedExperimentalFieldMesh &&
            extract_vulkan_scalar_field_mesh(
                computeResult, fieldMesh, fieldMeshError,
                classifiedSurfaceCells ? &gpuSurfaceCellIndices : nullptr,
                classifiedSurfaceCells ? &gpuSurfaceCellCubeIndices : nullptr)) {
          outMesh.swap(fieldMesh);
          usedExperimentalFieldMesh = true;
          if (options.showDebugLog) {
            std::cout << "[Vulkan] Experimental scalar-field surface extraction "
                         "succeeded with "
                      << outMesh.vertex_count() << " vertices and "
                      << outMesh.face_count() << " faces." << std::endl;
          }
        } else if (options.showDebugLog && !fieldMeshError.empty()) {
          std::cout << "[Vulkan] Experimental scalar-field extraction "
                       "unavailable, falling back to Frost CPU mesher: "
                    << fieldMeshError << std::endl;
        }
      }

      if (usedExperimentalFieldMesh) {
        outError.clear();
        return true;
      }

      frantic::particles::particle_array processedParticles;
      processedParticles.reset(channelMap);
      processedParticles.resize(particles.size());

      auto processedPositionAccessor =
          channelMap.get_accessor<frantic::graphics::vector3f>(_T("Position"));
      auto processedRadiusAccessor = channelMap.get_accessor<float>(_T("Radius"));
      const std::size_t particleStride = channelMap.structure_size();

      for (size_t i = 0; i < particles.size(); ++i) {
        char *dstParticle = processedParticles.at(i);
        const char *srcParticle = particles.at(i);
        std::memcpy(dstParticle, srcParticle, particleStride);
        processedPositionAccessor(dstParticle) = frantic::graphics::vector3f(
            outputParticles[i].x, outputParticles[i].y, outputParticles[i].z);
        processedRadiusAccessor(dstParticle) = outputParticles[i].radius;
      }

      frantic::logging::null_progress_logger logger;
      auto stream = boost::make_shared<
          frantic::particles::streams::particle_array_particle_istream>(
          processedParticles);
      std::size_t particleCount = 0;
      frost_parameters paramsCopy = params;
      frost::build_trimesh3(outMesh, particleCount, paramsCopy, stream, logger);

      if (options.showDebugLog) {
        if (usedComputeShader) {
          std::cout << "[Vulkan] Compute shader preprocessed "
                    << outputParticles.size()
                    << " particles before CPU meshing." << std::endl;
        } else {
          std::cout << "[Vulkan] Transfer path round-tripped "
                    << outputParticles.size()
                    << " particles before CPU meshing." << std::endl;
        }
        std::cout << "[Vulkan] Final surface extracted through Frost CPU "
                     "method: "
                  << get_meshing_method_name(paramsCopy.get_meshing_method())
                  << "." << std::endl;
      }

      if (outMesh.face_count() == 0) {
        outError =
            "Vulkan backend completed its particle round-trip, but Frost produced no faces.";
        return false;
      }

      outError.clear();
      return true;
    } catch (const std::exception &e) {
      outError = e.what();
      return false;
    } catch (...) {
      outError = "unknown exception in Vulkan backend";
      return false;
    }
  }
};

#endif

} // namespace

std::unique_ptr<FrostGpuBackend> create_vulkan_gpu_backend() {
#if defined(FROST_ENABLE_VULKAN_PROBE)
  if (get_vulkan_gpu_backend_info().available) {
    return std::make_unique<VulkanTransferGpuBackend>();
  }
#endif
  return nullptr;
}

FrostGpuBackendInfo get_vulkan_gpu_backend_info() {
#if defined(FROST_ENABLE_VULKAN_PROBE)
  static const FrostGpuBackendInfo info = []() {
    const auto bufferInfo = probe_frost_vulkan_storage_buffer();
    if (bufferInfo.storageBufferReady) {
      return FrostGpuBackendInfo{"vulkan-experimental", true};
    }
    return FrostGpuBackendInfo{"none", false};
  }();
  return info;
#else
  return {"none", false};
#endif
}
