#include "frost_interface.hpp"
#include "gpu_backend.hpp"

#include <frantic/channels/channel_propagation_policy.hpp> // Added for evolve_mesh
#include <frantic/geometry/trimesh3.hpp>
#include <frantic/logging/progress_logger.hpp>
#include <frantic/particles/particle_array.hpp>
#include <frantic/particles/streams/particle_array_particle_istream.hpp>
#include <frost/frost.hpp>

#include <algorithm>                       // For std::sort, std::unique
#include <cmath>                           // for std::round
#include <frantic/geometry/relaxation.hpp> // Added for laplacian_smooth
#include <frantic/graphics/boundbox3f.hpp>
#include <frantic/volumetrics/implicitsurface/particle_implicit_surface_policies.hpp>
#include <frantic/volumetrics/voxel_coord_system.hpp>
#include <frost/frost_parameters.hpp>
#include <iostream>
#include <tbb/parallel_for.h> // For parallel mesh copy
#include <unordered_map>

namespace {

bool validate_raw_mesh_buffers(const std::vector<float> &vertices,
                               const std::vector<int> &faces,
                               std::string &outError) {
  outError.clear();

  if (vertices.size() % 3u != 0u) {
    outError = "mesh vertex buffer size is not divisible by 3";
    return false;
  }

  if (faces.size() % 3u != 0u) {
    outError = "mesh face buffer size is not divisible by 3";
    return false;
  }

  const std::size_t vertexCount = vertices.size() / 3u;
  const std::size_t faceCount = faces.size() / 3u;

  if (vertexCount == 0u || faceCount == 0u) {
    if (vertexCount == 0u && faceCount == 0u) {
      return true;
    }
    outError = "mesh buffers contain vertices or faces, but not both";
    return false;
  }

  for (std::size_t i = 0; i < vertices.size(); ++i) {
    const float value = vertices[i];
    if (!std::isfinite(value)) {
      outError = "mesh vertex buffer contains a non-finite coordinate";
      return false;
    }
  }

  for (std::size_t faceIndex = 0; faceIndex < faceCount; ++faceIndex) {
    const std::size_t base = faceIndex * 3u;
    const int i0 = faces[base + 0u];
    const int i1 = faces[base + 1u];
    const int i2 = faces[base + 2u];
    if (i0 < 0 || i1 < 0 || i2 < 0) {
      outError = "mesh face buffer contains a negative vertex index";
      return false;
    }
    if (static_cast<std::size_t>(i0) >= vertexCount ||
        static_cast<std::size_t>(i1) >= vertexCount ||
        static_cast<std::size_t>(i2) >= vertexCount) {
      outError = "mesh face buffer contains an out-of-range vertex index";
      return false;
    }
    if (i0 == i1 || i1 == i2 || i0 == i2) {
      outError = "mesh face buffer contains a degenerate triangle";
      return false;
    }
  }

  return true;
}

// Build edge-use map to identify non-manifold and boundary edges
// Returns true if mesh is fully manifold (no boundary edges)
bool is_manifold_closed(const frantic::geometry::trimesh3 &mesh) {
  // Simple edge use counting
  std::unordered_map<int64_t, int> edgeUseCount;

  auto makeEdgeKey = [](int v0, int v1) -> int64_t {
    if (v0 > v1)
      std::swap(v0, v1);
    return ((int64_t)v0 << 32) | (int64_t)v1;
  };

  for (size_t f = 0; f < mesh.face_count(); ++f) {
    frantic::graphics::vector3 face = mesh.get_face(f);
    int v0 = (int)face.x, v1 = (int)face.y, v2 = (int)face.z;
    edgeUseCount[makeEdgeKey(v0, v1)]++;
    edgeUseCount[makeEdgeKey(v1, v2)]++;
    edgeUseCount[makeEdgeKey(v2, v0)]++;
  }

  // Check for non-manifold or boundary edges
  for (const auto &kv : edgeUseCount) {
    if (kv.second != 2)
      return false; // Not exactly 2 faces sharing edge
  }
  return true;
}

// Safe Laplacian smooth that skips boundary vertices to prevent holes
void safe_laplacian_smooth(frantic::geometry::trimesh3 &mesh, int iterations,
                           float strength) {
  if (iterations <= 0 || mesh.vertex_count() == 0 || mesh.face_count() == 0)
    return;

  // Build edge-use map
  std::unordered_map<int64_t, int> edgeUseCount;
  auto makeEdgeKey = [](int v0, int v1) -> int64_t {
    if (v0 > v1)
      std::swap(v0, v1);
    return ((int64_t)v0 << 32) | (int64_t)v1;
  };

  for (size_t f = 0; f < mesh.face_count(); ++f) {
    frantic::graphics::vector3 face = mesh.get_face(f);
    int v0 = (int)face.x, v1 = (int)face.y, v2 = (int)face.z;
    edgeUseCount[makeEdgeKey(v0, v1)]++;
    edgeUseCount[makeEdgeKey(v1, v2)]++;
    edgeUseCount[makeEdgeKey(v2, v0)]++;
  }

  // Mark boundary vertices (vertices on boundary edges)
  std::vector<bool> isBoundary(mesh.vertex_count(), false);
  for (const auto &kv : edgeUseCount) {
    if (kv.second != 2) { // Boundary or non-manifold edge
      int v0 = (int)(kv.first >> 32);
      int v1 = (int)(kv.first & 0xFFFFFFFF);
      if (v0 >= 0 && v0 < (int)mesh.vertex_count())
        isBoundary[v0] = true;
      if (v1 >= 0 && v1 < (int)mesh.vertex_count())
        isBoundary[v1] = true;
    }
  }

  // If no boundary vertices found, mesh is fully closed - use fast path
  bool hasBoundary = false;
  for (size_t i = 0; i < mesh.vertex_count() && !hasBoundary; ++i) {
    if (isBoundary[i])
      hasBoundary = true;
  }

  if (!hasBoundary) {
    // Mesh is fully closed, use Frantic's optimized version
    frantic::geometry::relaxation::laplacian_smooth(mesh, iterations, strength);
    return;
  }

  // Build vertex adjacency
  std::vector<std::vector<int>> vertexNeighbors(mesh.vertex_count());
  for (size_t f = 0; f < mesh.face_count(); ++f) {
    frantic::graphics::vector3 face = mesh.get_face(f);
    int v0 = (int)face.x, v1 = (int)face.y, v2 = (int)face.z;
    vertexNeighbors[v0].push_back(v1);
    vertexNeighbors[v0].push_back(v2);
    vertexNeighbors[v1].push_back(v0);
    vertexNeighbors[v1].push_back(v2);
    vertexNeighbors[v2].push_back(v0);
    vertexNeighbors[v2].push_back(v1);
  }

  // Remove duplicates from neighbor lists
  for (auto &neighbors : vertexNeighbors) {
    std::sort(neighbors.begin(), neighbors.end());
    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()),
                    neighbors.end());
  }

  // Perform smoothing iterations
  std::vector<frantic::graphics::vector3f> newPositions(mesh.vertex_count());

  for (int iter = 0; iter < iterations; ++iter) {
    // Copy current positions
    for (size_t i = 0; i < mesh.vertex_count(); ++i) {
      newPositions[i] = mesh.get_vertex(i);
    }

    // Smooth only non-boundary vertices, using ALL neighbors for averaging
    tbb::parallel_for(tbb::blocked_range<size_t>(0, mesh.vertex_count()),
                      [&](const tbb::blocked_range<size_t> &r) {
                        for (size_t i = r.begin(); i != r.end(); ++i) {
                          // Skip boundary vertices - they don't move
                          if (isBoundary[i])
                            continue;

                          // Skip vertices with too few neighbors
                          if (vertexNeighbors[i].size() < 3)
                            continue;

                          // Average ALL neighbors (including boundary if any)
                          frantic::graphics::vector3f avg(0, 0, 0);
                          for (int n : vertexNeighbors[i]) {
                            avg += mesh.get_vertex(n);
                          }
                          avg /= (float)vertexNeighbors[i].size();

                          // Blend toward average
                          newPositions[i] =
                              mesh.get_vertex(i) * (1.0f - strength) +
                              avg * strength;
                        }
                      });

    // Apply new positions
    for (size_t i = 0; i < mesh.vertex_count(); ++i) {
      mesh.get_vertex(i) = newPositions[i];
    }
  }
}

void apply_push(frantic::geometry::trimesh3 &mesh, float distance) {
  if (std::abs(distance) < 1e-5f)
    return;

  mesh.build_vertex_normals();

  if (!mesh.has_vertex_channel(_T("Normal")))
    return;

  auto normals = mesh.get_vertex_channel_accessor<frantic::graphics::vector3f>(
      _T("Normal"));

  for (size_t i = 0; i < mesh.vertex_count(); ++i) {
    frantic::graphics::vector3f n = normals[i];
    float magSq = n.get_magnitude_squared();

    // Skip invalid normals
    if (magSq < 1e-12f || std::isnan(magSq) || std::isinf(magSq))
      continue;

    // Normalize the normal vector before applying push
    float mag = std::sqrt(magSq);
    n /= mag;

    // Apply push along normalized normal
    mesh.get_vertex(i) += n * distance;
  }
}

void refine_gpu_mesh_with_zhu_bridson_surface(
    frantic::geometry::trimesh3 &mesh,
    const frantic::particles::particle_array &particles,
    const frost_parameters &params, float gpuSearchRadiusScale,
    int refinementIterations, bool showDebugLog) {
  if (refinementIterations <= 0 || mesh.vertex_count() == 0 ||
      particles.size() == 0) {
    return;
  }

  const auto &channelMap = particles.get_channel_map();
  if (!channelMap.has_channel(_T("Position")) ||
      !channelMap.has_channel(_T("Radius"))) {
    return;
  }

  auto positionAccessor =
      channelMap.get_accessor<frantic::graphics::vector3f>(_T("Position"));
  auto radiusAccessor = channelMap.get_accessor<float>(_T("Radius"));

  float maxParticleRadius = 0.0f;
  frantic::graphics::boundbox3f particleBounds;
  bool hasValidParticle = false;

  for (size_t i = 0; i < particles.size(); ++i) {
    const char *particle = particles[i];
    const frantic::graphics::vector3f pos = positionAccessor.get(particle);
    const float radius = radiusAccessor.get(particle);
    if (!pos.is_finite() || radius <= 0.0f) {
      continue;
    }

    particleBounds += pos;
    maxParticleRadius = std::max(maxParticleRadius, radius);
    hasValidParticle = true;
  }

  if (!hasValidParticle || maxParticleRadius <= 0.0f) {
    return;
  }

  float voxelSize = 0.1f;
  if (params.get_meshing_resolution_mode() == 0) {
    float resolution = params.get_meshing_resolution();
    if (resolution < 0.001f)
      resolution = 0.001f;
    voxelSize = maxParticleRadius / resolution;
  } else {
    voxelSize = params.get_meshing_voxel_length();
  }
  voxelSize = std::max(voxelSize, 0.002f);

  const float effectRadiusScale =
      gpuSearchRadiusScale > 0.0f ? gpuSearchRadiusScale
                                  : params.get_zhu_bridson_blend_radius_scale();
  const float searchRadius = maxParticleRadius * effectRadiusScale;
  if (searchRadius <= 0.0f) {
    return;
  }

  frantic::graphics::vector3f gridMin =
      particleBounds.minimum() - frantic::graphics::vector3f(searchRadius);
  gridMin.x = std::floor(gridMin.x / voxelSize) * voxelSize;
  gridMin.y = std::floor(gridMin.y / voxelSize) * voxelSize;
  gridMin.z = std::floor(gridMin.z / voxelSize) * voxelSize;

  const frantic::graphics::vector3f meshingOrigin =
      gridMin - frantic::graphics::vector3f(voxelSize * 0.5f);
  frantic::volumetrics::voxel_coord_system meshingVCS(meshingOrigin,
                                                      voxelSize);

  const float particleVoxelLength =
      std::max(searchRadius, 0.5f * voxelSize);
  frantic::volumetrics::voxel_coord_system particleVCS(
      frantic::graphics::vector3f(0), particleVoxelLength);

  frantic::particles::particle_grid_tree particleTree;
  particleTree.reset(channelMap, particleVCS);

  for (size_t i = 0; i < particles.size(); ++i) {
    const char *particle = particles[i];
    const frantic::graphics::vector3f pos = positionAccessor.get(particle);
    const float radius = radiusAccessor.get(particle);
    if (!pos.is_finite() || radius <= 0.0f) {
      continue;
    }
    particleTree.insert(particle);
  }

  const bool enableLowDensityTrimming =
      params.get_zhu_bridson_enable_low_density_trimming();
  const float lowDensityTrimmingDensity =
      enableLowDensityTrimming
          ? params.get_zhu_bridson_low_density_trimming_threshold()
          : 0.0f;
  const float lowDensityTrimmingStrength =
      enableLowDensityTrimming
          ? params.get_zhu_bridson_low_density_trimming_strength()
          : 0.0f;

  frantic::volumetrics::implicitsurface::particle_zhu_bridson_is_policy policy(
      particleTree, maxParticleRadius, effectRadiusScale,
      lowDensityTrimmingDensity, lowDensityTrimmingStrength, meshingVCS,
      refinementIterations);

  const float gradientStep = std::max(voxelSize * 0.5f, 1e-4f);
  const float maxStep = std::max(voxelSize, 1e-4f);
  const float convergence = std::max(voxelSize * 1e-3f, 1e-5f);

  for (size_t i = 0; i < mesh.vertex_count(); ++i) {
    frantic::graphics::vector3f v = mesh.get_vertex(i);

    for (int iter = 0; iter < refinementIterations; ++iter) {
      const float density = policy.get_density(v);
      if (!std::isfinite(density) || std::abs(density) <= convergence) {
        break;
      }

      frantic::graphics::vector3f gradient =
          policy.get_gradient(v, gradientStep);
      const float gradMagSq = gradient.get_magnitude_squared();
      if (!std::isfinite(gradMagSq) || gradMagSq < 1e-12f) {
        break;
      }

      frantic::graphics::vector3f step = gradient * (density / gradMagSq);
      const float stepMagSq = step.get_magnitude_squared();
      if (!std::isfinite(stepMagSq) || stepMagSq < 1e-16f) {
        break;
      }

      const float stepMag = std::sqrt(stepMagSq);
      if (stepMag > maxStep) {
        step *= maxStep / stepMag;
      }

      v -= step;
    }

    mesh.get_vertex(i) = v;
  }

  if (showDebugLog) {
    std::cout << "[CUDA] Surface refinement projected " << mesh.vertex_count()
              << " vertices with " << refinementIterations
              << " Zhu-Bridson iteration(s)." << std::endl;
  }
}
} // namespace

struct FrostInterface::Impl {
  frost_parameters params;
  frantic::particles::particle_array particles;
  std::unique_ptr<FrostGpuBackend> gpu_backend;
  std::string last_meshing_backend = "cpu";
  std::string last_meshing_status;
  bool last_meshing_used_fallback = false;

  // Default constructor
  Impl() {
    // Set defaults
    params.set_meshing_resolution(0.1f);
    // Default channel map
    frantic::channels::channel_map map;
    map.define_channel<frantic::graphics::vector3f>(_T("Position"));
    map.define_channel<float>(_T("Radius"));
    map.define_channel<frantic::graphics::vector3f>(_T("Velocity"));
    map.end_channel_definition();
    particles.set_channel_map(map);
    gpu_backend = create_frost_gpu_backend();
  }

  // Custom parameters for manual calls
  int relax_iterations = 0;
  float relax_strength = 0.5f;
  float push_distance = 0.0f;
  bool enable_gpu = false;
  float gpu_search_radius_scale = 2.0f;
  float gpu_voxel_size =
      0.1f; // GPU-specific voxel size (controls mesh resolution)
  int gpu_block_size = 256;
  int gpu_surface_refinement = 0;
  bool show_debug_log = false; // Debug toggle
};

FrostInterface::FrostInterface() : m_impl(std::make_unique<Impl>()) {}
FrostInterface::~FrostInterface() = default;

bool FrostInterface::has_gpu_backend() {
  return get_frost_gpu_backend_info().available;
}

std::string FrostInterface::get_gpu_backend_name() {
  return get_frost_gpu_backend_info().name;
}

std::string FrostInterface::get_last_meshing_backend() const {
  return m_impl->last_meshing_backend;
}

std::string FrostInterface::get_last_meshing_status() const {
  return m_impl->last_meshing_status;
}

bool FrostInterface::get_last_meshing_used_fallback() const {
  return m_impl->last_meshing_used_fallback;
}

void FrostInterface::set_particles(const float *positions, size_t count,
                                   const float *radii,
                                   const float *velocities) {
  bool has_velocity = velocities != nullptr;

  // Define channel map
  frantic::channels::channel_map map;
  map.define_channel<frantic::graphics::vector3f>(_T("Position"));
  map.define_channel<float>(_T("Radius"));
  if (has_velocity) {
    map.define_channel<frantic::graphics::vector3f>(_T("Velocity"));
  }
  map.end_channel_definition();

  m_impl->particles.reset(map);
  m_impl->particles.resize(count); // Use resize instead of reserve+add

  // Accessors
  frantic::channels::channel_accessor<frantic::graphics::vector3f> acc_pos =
      map.get_accessor<frantic::graphics::vector3f>(_T("Position"));
  frantic::channels::channel_accessor<float> acc_rad =
      map.get_accessor<float>(_T("Radius"));

  // Fill basic data - PARALLELIZED with TBB
  tbb::parallel_for(tbb::blocked_range<size_t>(0, count),
                    [&](const tbb::blocked_range<size_t> &r) {
                      for (size_t i = r.begin(); i != r.end(); ++i) {
                        char *p = m_impl->particles.at(i);
                        acc_pos(p) = frantic::graphics::vector3f(
                            positions[i * 3 + 0], positions[i * 3 + 1],
                            positions[i * 3 + 2]);
                        acc_rad(p) = radii[i];
                      }
                    });

  // Fill velocity if present - PARALLELIZED
  if (has_velocity) {
    frantic::channels::channel_accessor<frantic::graphics::vector3f> acc_vel =
        map.get_accessor<frantic::graphics::vector3f>(_T("Velocity"));

    tbb::parallel_for(tbb::blocked_range<size_t>(0, count),
                      [&](const tbb::blocked_range<size_t> &r) {
                        for (size_t i = r.begin(); i != r.end(); ++i) {
                          char *p = m_impl->particles.at(i);
                          acc_vel(p) = frantic::graphics::vector3f(
                              velocities[i * 3 + 0], velocities[i * 3 + 1],
                              velocities[i * 3 + 2]);
                        }
                      });
  }
}

void FrostInterface::set_parameter(const std::string &name, bool value) {
  try {
    auto &p = m_impl->params;

    if (name == "use_gpu")
      m_impl->enable_gpu = value;
    else if (name == "show_debug_log")
      m_impl->show_debug_log = value;
    else if (name == "zhu_bridson_low_density_trimming")
      p.set_zhu_bridson_enable_low_density_trimming(value);
  } catch (const std::exception &) {
  }
}

void FrostInterface::set_parameter(const std::string &name, int value) {
  try {
    auto &p = m_impl->params;

    if (name == "meshing_method")
      p.set_meshing_method(value);
    else if (name == "meshing_resolution_mode")
      p.set_meshing_resolution_mode(value);
    else if (name == "relax_iterations")
      m_impl->relax_iterations = value;
    else if (name == "gpu_block_size")
      m_impl->gpu_block_size = value;
    else if (name == "gpu_surface_refinement")
      m_impl->gpu_surface_refinement = value;
    else if (name == "vert_refinement")
      p.set_vert_refinement_iterations(value);
  } catch (const std::exception &) {
  }
}

void FrostInterface::set_parameter(const std::string &name, float value) {
  try {
    auto &p = m_impl->params;

    if (name == "resolution")
      p.set_meshing_resolution(value);
    else if (name == "voxel_size")
      p.set_meshing_voxel_length(value);
    else if (name == "relax_strength")
      m_impl->relax_strength = value;
    else if (name == "push_distance")
      m_impl->push_distance = value;
    else if (name == "gpu_search_radius_scale")
      m_impl->gpu_search_radius_scale = value;
    else if (name == "gpu_voxel_size")
      m_impl->gpu_voxel_size = value;
    else if (name == "metaball_radius_scale")
      p.set_metaball_radius_scale(value);
    else if (name == "metaball_isosurface_level")
      p.set_metaball_isosurface_level(value);
    else if (name == "plain_marching_cubes_radius_scale")
      p.set_plain_marching_cubes_radius_scale(value);
    else if (name == "plain_marching_cubes_isovalue")
      p.set_plain_marching_cubes_isovalue(value);
    else if (name == "zhu_bridson_blend_radius_scale")
      p.set_zhu_bridson_blend_radius_scale(value);
    else if (name == "zhu_bridson_trimming_threshold")
      p.set_zhu_bridson_low_density_trimming_threshold(value);
    else if (name == "zhu_bridson_trimming_strength")
      p.set_zhu_bridson_low_density_trimming_strength(value);
    else if (name == "anisotropic_radius_scale")
      p.set_anisotropic_radius_scale(value);
    else if (name == "anisotropic_max_anisotropy")
      p.set_anisotropic_max_anisotropy(value);
  } catch (const std::exception &) {
  }
}

#include <boost/make_shared.hpp> // Added for make_shared

void FrostInterface::generate_mesh(std::vector<float> &vertices,
                                   std::vector<int> &faces) {
  frantic::geometry::trimesh3 outMethod;
  bool gpuSuccess = false;
  std::string validationError;
  std::string cpuFallbackReason;
  const bool requiresCpuVertexRefinement =
      m_impl->params.get_vert_refinement_iterations() > 0;
  const bool allowGpuSurfaceMeshing = !requiresCpuVertexRefinement;
  const bool canBypassTriMeshForGpu =
      allowGpuSurfaceMeshing &&
      m_impl->gpu_surface_refinement <= 0 &&
      m_impl->relax_iterations <= 0 &&
      std::abs(m_impl->push_distance) <= 1e-5f;

  m_impl->last_meshing_backend = "cpu";
  m_impl->last_meshing_status.clear();
  m_impl->last_meshing_used_fallback = false;

  if (m_impl->enable_gpu && m_impl->gpu_backend &&
      m_impl->gpu_backend->is_available() && allowGpuSurfaceMeshing) {
    FrostGpuMeshingOptions gpuOptions;
    gpuOptions.searchRadiusScale = m_impl->gpu_search_radius_scale;
    gpuOptions.blockSize = m_impl->gpu_block_size;
    gpuOptions.showDebugLog = m_impl->show_debug_log;

    std::string gpuError;
    if (canBypassTriMeshForGpu &&
        m_impl->gpu_backend->generate_mesh_buffers(
            m_impl->particles, m_impl->params, gpuOptions, vertices, faces,
            gpuError)) {
      if (!validate_raw_mesh_buffers(vertices, faces, validationError)) {
        if (m_impl->show_debug_log) {
          std::cerr << "[" << m_impl->gpu_backend->name()
                    << "] Raw GPU mesh buffers failed validation: "
                    << validationError
                    << ". Falling back to the safer surface path."
                    << std::endl;
        }
        m_impl->last_meshing_status =
            "Raw GPU buffers failed validation; using safer GPU surface path.";
        vertices.clear();
        faces.clear();
      } else {
        m_impl->last_meshing_backend =
            std::string(m_impl->gpu_backend->name()) + "-raw";
        m_impl->last_meshing_status =
            "Direct GPU raw mesh buffers used for the final surface.";
        m_impl->last_meshing_used_fallback = false;
        if (m_impl->show_debug_log) {
          std::cout << "[" << m_impl->gpu_backend->name()
                    << "] Returned raw GPU mesh buffers directly." << std::endl;
        }
        return;
      }
    } else if (canBypassTriMeshForGpu && m_impl->show_debug_log &&
               !gpuError.empty()) {
      if (m_impl->show_debug_log) {
        std::cerr << "[" << m_impl->gpu_backend->name()
                  << "] Raw GPU buffer path unavailable: " << gpuError
                  << std::endl;
      }
      m_impl->last_meshing_status = "Raw GPU path unavailable; trying safer GPU "
                                    "surface extraction.";
    }

    gpuSuccess = m_impl->gpu_backend->generate_mesh(
        m_impl->particles, m_impl->params, gpuOptions, outMethod, gpuError);

    if (gpuSuccess) {
      m_impl->last_meshing_backend = m_impl->gpu_backend->name();
      if (m_impl->last_meshing_status.empty()) {
        m_impl->last_meshing_status =
            "GPU surface extraction completed successfully.";
      }
      m_impl->last_meshing_used_fallback = false;
    }

    if (!gpuSuccess && m_impl->show_debug_log && !gpuError.empty()) {
      std::cerr << "[" << m_impl->gpu_backend->name()
                << "] Falling back to CPU mesher: " << gpuError << std::endl;
    }
    if (!gpuSuccess && !gpuError.empty()) {
      cpuFallbackReason = gpuError;
    }
  } else if (m_impl->enable_gpu && m_impl->gpu_backend &&
             m_impl->gpu_backend->is_available() &&
             requiresCpuVertexRefinement) {
    if (m_impl->show_debug_log) {
      std::cout << "[" << m_impl->gpu_backend->name()
                << "] Vertex refinement is enabled, so GPU surface extraction "
                   "is disabled and Frost CPU meshing is used instead."
                << std::endl;
    }
    cpuFallbackReason =
        "Vertex Refinement requires the CPU surface meshing path.";
  } else if (m_impl->enable_gpu &&
             (!m_impl->gpu_backend || !m_impl->gpu_backend->is_available())) {
    cpuFallbackReason =
        "GPU acceleration was requested, but no GPU backend is available in "
        "the current native build.";
  }

  frantic::logging::null_progress_logger logger;

  // Create stream
  auto stream = boost::make_shared<
      frantic::particles::streams::particle_array_particle_istream>(
      m_impl->particles);

  std::size_t particleCount = 0;

  // Run Frost (parallelized via TBB) - Only if GPU didn't already generate
  // mesh
  if (!gpuSuccess) {
    frost::build_trimesh3(outMethod, particleCount, m_impl->params, stream,
                          logger);
    if (m_impl->enable_gpu) {
      m_impl->last_meshing_backend = "cpu-fallback";
      m_impl->last_meshing_used_fallback = true;
      if (!cpuFallbackReason.empty()) {
        m_impl->last_meshing_status = cpuFallbackReason;
      } else {
        m_impl->last_meshing_status =
            "GPU meshing was unavailable, so Frost used the CPU path.";
      }
    } else {
      m_impl->last_meshing_backend = "cpu";
      m_impl->last_meshing_used_fallback = false;
      m_impl->last_meshing_status = "CPU meshing path.";
    }
  }

  if (gpuSuccess && m_impl->gpu_surface_refinement > 0) {
    refine_gpu_mesh_with_zhu_bridson_surface(
        outMethod, m_impl->particles, m_impl->params,
        m_impl->gpu_search_radius_scale, m_impl->gpu_surface_refinement,
        m_impl->show_debug_log);
  }

  // Post-Processing Pipeline (also parallelized)

  // 1. Push (Inflate/Deflate)
  if (std::abs(m_impl->push_distance) > 1e-5f) {
    apply_push(outMethod, m_impl->push_distance);
  }

  // 2. Relax (Laplacian Smoothing) - uses hybrid approach:
  // - If mesh is fully closed: uses Frantic's optimized version
  // - If mesh has boundary edges: uses safe version that freezes boundary
  if (m_impl->relax_iterations > 0) {
    safe_laplacian_smooth(outMethod, m_impl->relax_iterations,
                          m_impl->relax_strength);
  }

  // Convert output - Optimized for performance
  size_t vertCount = outMethod.vertex_count();
  size_t faceCount = outMethod.face_count();

  vertices.resize(vertCount * 3);
  faces.resize(faceCount * 3);

  float *verts_ptr = vertices.data();
  int *faces_ptr = faces.data();

  // Parallel copy of vertices using TBB
  tbb::parallel_for(tbb::blocked_range<size_t>(0, vertCount),
                    [&](const tbb::blocked_range<size_t> &r) {
                      for (size_t i = r.begin(); i != r.end(); ++i) {
                        frantic::graphics::vector3f v = outMethod.get_vertex(i);
                        size_t idx = i * 3;
                        verts_ptr[idx + 0] = v.x;
                        verts_ptr[idx + 1] = v.y;
                        verts_ptr[idx + 2] = v.z;
                      }
                    });

  // Parallel copy of faces using TBB
  tbb::parallel_for(tbb::blocked_range<size_t>(0, faceCount),
                    [&](const tbb::blocked_range<size_t> &r) {
                      for (size_t i = r.begin(); i != r.end(); ++i) {
                        frantic::graphics::vector3 f = outMethod.get_face(i);
                        size_t idx = i * 3;
                        faces_ptr[idx + 0] = (int)f.x;
                        faces_ptr[idx + 1] = (int)f.y;
                        faces_ptr[idx + 2] = (int)f.z;
                      }
                    });

  if (!validate_raw_mesh_buffers(vertices, faces, validationError)) {
    throw std::runtime_error("generated mesh validation failed: " +
                             validationError);
  }
}
