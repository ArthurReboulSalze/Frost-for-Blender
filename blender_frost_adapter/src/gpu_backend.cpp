#include "gpu_backend.hpp"
#include "vulkan_backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <unordered_map>
#include <vector>

#include <frantic/graphics/boundbox3f.hpp>
#include <tbb/parallel_for.h>

namespace {

struct IntKey {
  int64_t x, y, z;

  bool operator==(const IntKey &other) const {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct IntKeyHash {
  size_t operator()(const IntKey &key) const {
    size_t h = std::hash<int64_t>{}(key.x);
    h ^= std::hash<int64_t>{}(key.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int64_t>{}(key.z) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};

inline IntKey make_int_key(float x, float y, float z, float weldGridScale) {
  return {static_cast<int64_t>(std::round(x * weldGridScale)),
          static_cast<int64_t>(std::round(y * weldGridScale)),
          static_cast<int64_t>(std::round(z * weldGridScale))};
}

class NullGpuBackend : public FrostGpuBackend {
public:
  const char *name() const override { return "none"; }

  bool is_available() const override { return false; }

  bool generate_mesh(const frantic::particles::particle_array &,
                     const frost_parameters &, const FrostGpuMeshingOptions &,
                     frantic::geometry::trimesh3 &, std::string &outError) override {
    outError = "no GPU backend is compiled into this native build";
    return false;
  }
};

#ifdef FROST_ENABLE_CUDA
extern "C" void *FrostGPUManager_Create();
extern "C" void FrostGPUManager_Destroy(void *manager);
extern "C" void FrostGPUManager_UploadParticles(void *manager,
                                                const float *positions,
                                                const float *radii, int count);
extern "C" int FrostGPUManager_FindNeighbors(void *manager, float radius);
extern "C" void FrostGPUManager_ComputeDensity(void *manager);
extern "C" void FrostGPUManager_SetMeshingParameters(
    void *manager, float lowDensityTrimmingDensity,
    float lowDensityTrimmingStrength, int blockSize);
extern "C" void FrostGPUManager_SetScalarFieldParameters(
    void *manager, float minX, float minY, float minZ, float sizeX,
    float sizeY, float sizeZ, float voxelSize);
extern "C" void FrostGPUManager_ComputeScalarField(void *manager);
extern "C" void FrostGPUManager_ComputeMesh(void *manager, float isovalue);
extern "C" int FrostGPUManager_GetTriangleCount(void *manager);
extern "C" void FrostGPUManager_DownloadMesh(void *manager, float *verticesHost,
                                             int maxVerts);
extern "C" void FrostGPUManager_SetDebug(void *manager, bool debug);

class CudaGpuBackend : public FrostGpuBackend {
public:
  CudaGpuBackend() = default;

  ~CudaGpuBackend() override {
    if (m_manager) {
      FrostGPUManager_Destroy(m_manager);
      m_manager = nullptr;
    }
  }

  const char *name() const override { return "cuda"; }

  bool is_available() const override { return true; }

  bool generate_mesh(const frantic::particles::particle_array &particles,
                     const frost_parameters &params,
                     const FrostGpuMeshingOptions &options,
                     frantic::geometry::trimesh3 &outMesh,
                     std::string &outError) override {
    try {
      if (options.showDebugLog) {
        std::cout << "[CUDA] GPU backend enabled" << std::endl;
      }

      if (!m_manager) {
        m_manager = FrostGPUManager_Create();
      }
      if (!m_manager) {
        outError = "failed to create CUDA backend manager";
        return false;
      }

      FrostGPUManager_SetDebug(m_manager, options.showDebugLog);

      const size_t numParticles = particles.size();
      if (numParticles == 0) {
        outError.clear();
        return false;
      }

      float maxParticleRadius = 0.1f;
      bool foundMaxRadius = false;
      std::vector<float> positionsFlat(numParticles * 3);
      std::vector<float> radiiFlat(numParticles, maxParticleRadius);
      frantic::graphics::boundbox3f bounds;

      auto &channelMap = particles.get_channel_map();
      const bool hasRadius = channelMap.has_channel(_T("Radius"));
      frantic::channels::channel_accessor<float> radiusAccessor;
      if (hasRadius) {
        radiusAccessor = channelMap.get_accessor<float>(_T("Radius"));
      }
      auto positionAccessor =
          channelMap.get_accessor<frantic::graphics::vector3f>(_T("Position"));

      for (size_t i = 0; i < numParticles; ++i) {
        char *particle = particles.at(i);
        frantic::graphics::vector3f pos = positionAccessor.get(particle);

        positionsFlat[i * 3 + 0] = pos.x;
        positionsFlat[i * 3 + 1] = pos.y;
        positionsFlat[i * 3 + 2] = pos.z;
        bounds += pos;

        if (hasRadius) {
          const float radius = radiusAccessor.get(particle);
          radiiFlat[i] = radius;
          if (!foundMaxRadius || radius > maxParticleRadius) {
            maxParticleRadius = radius;
            foundMaxRadius = true;
          }
        }
      }

      FrostGPUManager_UploadParticles(m_manager, positionsFlat.data(),
                                      radiiFlat.data(),
                                      static_cast<int>(numParticles));

      float voxelSize = 0.1f;
      const int resolutionMode = params.get_meshing_resolution_mode();
      if (resolutionMode == 0) {
        float resolution = params.get_meshing_resolution();
        if (resolution < 0.001f) {
          resolution = 0.001f;
        }
        voxelSize = maxParticleRadius / resolution;
      } else {
        voxelSize = params.get_meshing_voxel_length();
      }
      voxelSize = std::max(voxelSize, 0.002f);

      const float searchRadius = maxParticleRadius * options.searchRadiusScale;
      if (searchRadius <= 0.0f) {
        outError = "invalid GPU search radius";
        return false;
      }

      const int neighborResult =
          FrostGPUManager_FindNeighbors(m_manager, searchRadius);
      if (neighborResult != 0) {
        outError = "GPU neighbor search failed";
        return false;
      }

      if (options.showDebugLog) {
        std::cout << "[CUDA] Neighbor search completed successfully"
                  << std::endl;
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

      FrostGPUManager_SetMeshingParameters(
          m_manager, lowDensityTrimmingDensity, lowDensityTrimmingStrength,
          options.blockSize);
      FrostGPUManager_ComputeDensity(m_manager);

      const float maxRadius = searchRadius;
      frantic::graphics::vector3f minPt =
          bounds.minimum() - frantic::graphics::vector3f(maxRadius);
      frantic::graphics::vector3f maxPt =
          bounds.maximum() + frantic::graphics::vector3f(maxRadius);

      minPt.x = std::floor(minPt.x / voxelSize) * voxelSize;
      minPt.y = std::floor(minPt.y / voxelSize) * voxelSize;
      minPt.z = std::floor(minPt.z / voxelSize) * voxelSize;
      maxPt.x = std::ceil(maxPt.x / voxelSize) * voxelSize;
      maxPt.y = std::ceil(maxPt.y / voxelSize) * voxelSize;
      maxPt.z = std::ceil(maxPt.z / voxelSize) * voxelSize;

      const frantic::graphics::vector3f size = maxPt - minPt;

      FrostGPUManager_SetScalarFieldParameters(
          m_manager, minPt.x, minPt.y, minPt.z, size.x, size.y, size.z,
          voxelSize);
      FrostGPUManager_ComputeScalarField(m_manager);

      if (options.showDebugLog) {
        std::cout << "[CUDA] Scalar field computed" << std::endl;
      }

      FrostGPUManager_ComputeMesh(m_manager, 0.0f);

      const int triangleCount = FrostGPUManager_GetTriangleCount(m_manager);
      if (triangleCount <= 0) {
        outError = "GPU mesher produced no triangles";
        return false;
      }

      std::vector<float> rawVertices(triangleCount * 9);
      FrostGPUManager_DownloadMesh(m_manager, rawVertices.data(),
                                   static_cast<int>(rawVertices.size()));

      if (options.showDebugLog) {
        std::cout << "[CUDA] Welding vertices..." << std::endl;
      }

      std::unordered_map<IntKey, int, IntKeyHash> vertexMap;
      std::vector<float> weldedVertices;
      std::vector<int> weldedFaces;
      int nextIndex = 0;
      const float weldTolerance = std::max(voxelSize * 0.0025f, 1e-5f);
      const float weldGridScale = 1.0f / weldTolerance;

      weldedVertices.reserve(triangleCount * 9);
      weldedFaces.reserve(triangleCount * 3);

      for (int t = 0; t < triangleCount; ++t) {
        int indices[3];

        for (int j = 0; j < 3; ++j) {
          const float vx = rawVertices[(t * 3 + j) * 3 + 0];
          const float vy = rawVertices[(t * 3 + j) * 3 + 1];
          const float vz = rawVertices[(t * 3 + j) * 3 + 2];

          const IntKey key = make_int_key(vx, vy, vz, weldGridScale);
          auto it = vertexMap.find(key);
          if (it != vertexMap.end()) {
            indices[j] = it->second;
            continue;
          }

          const int vertexIndex = nextIndex++;
          vertexMap[key] = vertexIndex;
          indices[j] = vertexIndex;
          weldedVertices.push_back(vx);
          weldedVertices.push_back(vy);
          weldedVertices.push_back(vz);
        }

        if (indices[0] == indices[1] || indices[1] == indices[2] ||
            indices[0] == indices[2]) {
          continue;
        }

        weldedFaces.push_back(indices[0]);
        weldedFaces.push_back(indices[1]);
        weldedFaces.push_back(indices[2]);
      }

      const int vertexCount = static_cast<int>(weldedVertices.size() / 3);
      const int faceCount = static_cast<int>(weldedFaces.size() / 3);

      if (options.showDebugLog) {
        std::cout << "[CUDA] Welded mesh: " << vertexCount << " vertices, "
                  << faceCount << " faces." << std::endl;
      }

      outMesh.set_vertex_count(vertexCount);
      outMesh.set_face_count(faceCount);

      tbb::parallel_for(tbb::blocked_range<size_t>(0, vertexCount),
                        [&](const tbb::blocked_range<size_t> &r) {
                          for (size_t i = r.begin(); i != r.end(); ++i) {
                            outMesh.get_vertex(i) =
                                frantic::graphics::vector3f(
                                    weldedVertices[i * 3 + 0],
                                    weldedVertices[i * 3 + 1],
                                    weldedVertices[i * 3 + 2]);
                          }
                        });

      tbb::parallel_for(tbb::blocked_range<size_t>(0, faceCount),
                        [&](const tbb::blocked_range<size_t> &r) {
                          for (size_t i = r.begin(); i != r.end(); ++i) {
                            outMesh.get_face(i) = frantic::graphics::vector3(
                                static_cast<float>(weldedFaces[i * 3 + 0]),
                                static_cast<float>(weldedFaces[i * 3 + 1]),
                                static_cast<float>(weldedFaces[i * 3 + 2]));
                          }
                        });

      outError.clear();
      return true;
    } catch (const std::exception &e) {
      outError = e.what();
      return false;
    } catch (...) {
      outError = "unknown exception in CUDA backend";
      return false;
    }
  }

private:
  void *m_manager = nullptr;
};
#endif

} // namespace

std::unique_ptr<FrostGpuBackend> create_frost_gpu_backend() {
#if defined(FROST_ENABLE_VULKAN_PROBE)
  if (auto vulkanBackend = create_vulkan_gpu_backend()) {
    return vulkanBackend;
  }
#endif
#ifdef FROST_ENABLE_CUDA
  return std::make_unique<CudaGpuBackend>();
#else
  return std::make_unique<NullGpuBackend>();
#endif
}

FrostGpuBackendInfo get_frost_gpu_backend_info() {
#if defined(FROST_ENABLE_VULKAN_PROBE)
  const auto vulkanInfo = get_vulkan_gpu_backend_info();
  if (vulkanInfo.available) {
    return vulkanInfo;
  }
#endif
#ifdef FROST_ENABLE_CUDA
  return {"cuda", true};
#else
  return {"none", false};
#endif
}
