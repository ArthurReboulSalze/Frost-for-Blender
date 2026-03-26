#pragma once

#include <memory>
#include <string>

#include <frantic/geometry/trimesh3.hpp>
#include <frantic/particles/particle_array.hpp>
#include <frost/frost_parameters.hpp>

struct FrostGpuBackendInfo {
  std::string name;
  bool available = false;
};

struct FrostGpuMeshingOptions {
  float searchRadiusScale = 2.0f;
  int blockSize = 256;
  bool showDebugLog = false;
};

class FrostGpuBackend {
public:
  virtual ~FrostGpuBackend() = default;

  virtual const char *name() const = 0;
  virtual bool is_available() const = 0;
  virtual bool generate_mesh(const frantic::particles::particle_array &particles,
                             const frost_parameters &params,
                             const FrostGpuMeshingOptions &options,
                             frantic::geometry::trimesh3 &outMesh,
                             std::string &outError) = 0;

  virtual bool generate_mesh_buffers(
      const frantic::particles::particle_array &particles,
      const frost_parameters &params,
      const FrostGpuMeshingOptions &options,
      std::vector<float> &outVertices,
      std::vector<int> &outFaces,
      std::string &outError) {
    outVertices.clear();
    outFaces.clear();
    outError = "raw GPU mesh buffers are unavailable for this backend";
    return false;
  }
};

std::unique_ptr<FrostGpuBackend> create_frost_gpu_backend();
FrostGpuBackendInfo get_frost_gpu_backend_info();
