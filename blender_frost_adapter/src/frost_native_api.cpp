#include "frost_native_api.h"

#include "frost_interface.hpp"
#include "vulkan_buffer_probe.hpp"
#include "vulkan_compute_probe.hpp"
#include "vulkan_probe.hpp"

#include <cstring>
#include <exception>
#include <string>
#include <vector>

namespace {

struct FrostNativeContext {
  FrostInterface frost;
  std::vector<float> vertices;
  std::vector<int> faces;
  std::string lastError;
};

FrostNativeContext *as_context(void *handle) {
  return static_cast<FrostNativeContext *>(handle);
}

void clear_error(FrostNativeContext *ctx) {
  if (ctx) {
    ctx->lastError.clear();
  }
}

int set_error(FrostNativeContext *ctx, const char *message) {
  if (ctx) {
    ctx->lastError = message ? message : "unknown Frost native error";
  }
  return 0;
}

int set_error(FrostNativeContext *ctx, const std::exception &e) {
  return set_error(ctx, e.what());
}

} // namespace

FROST_NATIVE_API void *Frost_Create() {
  try {
    return new FrostNativeContext();
  } catch (...) {
    return nullptr;
  }
}

FROST_NATIVE_API void Frost_Destroy(void *handle) {
  delete as_context(handle);
}

FROST_NATIVE_API int Frost_SetParticles(void *handle, const float *positions,
                                        size_t particleCount,
                                        const float *radii,
                                        const float *velocities) {
  auto *ctx = as_context(handle);
  if (!ctx) {
    return 0;
  }

  if (!positions || !radii) {
    return set_error(ctx, "positions and radii pointers must be valid");
  }

  try {
    clear_error(ctx);
    ctx->frost.set_particles(positions, particleCount, radii, velocities);
    return 1;
  } catch (const std::exception &e) {
    return set_error(ctx, e);
  }
}

FROST_NATIVE_API int Frost_SetParameterBool(void *handle, const char *name,
                                            int value) {
  auto *ctx = as_context(handle);
  if (!ctx || !name) {
    return 0;
  }

  try {
    clear_error(ctx);
    ctx->frost.set_parameter(name, value != 0);
    return 1;
  } catch (const std::exception &e) {
    return set_error(ctx, e);
  }
}

FROST_NATIVE_API int Frost_SetParameterInt(void *handle, const char *name,
                                           int value) {
  auto *ctx = as_context(handle);
  if (!ctx || !name) {
    return 0;
  }

  try {
    clear_error(ctx);
    ctx->frost.set_parameter(name, value);
    return 1;
  } catch (const std::exception &e) {
    return set_error(ctx, e);
  }
}

FROST_NATIVE_API int Frost_SetParameterFloat(void *handle, const char *name,
                                             float value) {
  auto *ctx = as_context(handle);
  if (!ctx || !name) {
    return 0;
  }

  try {
    clear_error(ctx);
    ctx->frost.set_parameter(name, value);
    return 1;
  } catch (const std::exception &e) {
    return set_error(ctx, e);
  }
}

FROST_NATIVE_API int Frost_GenerateMesh(void *handle) {
  auto *ctx = as_context(handle);
  if (!ctx) {
    return 0;
  }

  try {
    clear_error(ctx);
    ctx->vertices.clear();
    ctx->faces.clear();
    ctx->frost.generate_mesh(ctx->vertices, ctx->faces);
    return 1;
  } catch (const std::exception &e) {
    ctx->vertices.clear();
    ctx->faces.clear();
    return set_error(ctx, e);
  }
}

FROST_NATIVE_API int Frost_GetVertexCount(void *handle) {
  auto *ctx = as_context(handle);
  if (!ctx) {
    return 0;
  }
  return (int)(ctx->vertices.size() / 3);
}

FROST_NATIVE_API int Frost_GetFaceCount(void *handle) {
  auto *ctx = as_context(handle);
  if (!ctx) {
    return 0;
  }
  return (int)(ctx->faces.size() / 3);
}

FROST_NATIVE_API int Frost_CopyVertices(void *handle, float *outVertices,
                                        size_t floatCount) {
  auto *ctx = as_context(handle);
  if (!ctx || (!ctx->vertices.empty() && !outVertices)) {
    return 0;
  }

  if (floatCount < ctx->vertices.size()) {
    return set_error(ctx, "vertex buffer is too small");
  }

  if (!ctx->vertices.empty()) {
    std::memcpy(outVertices, ctx->vertices.data(),
                ctx->vertices.size() * sizeof(float));
  }
  return 1;
}

FROST_NATIVE_API int Frost_CopyFaces(void *handle, int *outFaces,
                                     size_t intCount) {
  auto *ctx = as_context(handle);
  if (!ctx || (!ctx->faces.empty() && !outFaces)) {
    return 0;
  }

  if (intCount < ctx->faces.size()) {
    return set_error(ctx, "face buffer is too small");
  }

  if (!ctx->faces.empty()) {
    std::memcpy(outFaces, ctx->faces.data(),
                ctx->faces.size() * sizeof(int));
  }
  return 1;
}

FROST_NATIVE_API const char *Frost_GetLastError(void *handle) {
  auto *ctx = as_context(handle);
  if (!ctx || ctx->lastError.empty()) {
    return "";
  }
  return ctx->lastError.c_str();
}

FROST_NATIVE_API int Frost_HasGpuBackend() {
  return FrostInterface::has_gpu_backend() ? 1 : 0;
}

FROST_NATIVE_API int Frost_HasCudaBackend() {
  return FrostInterface::get_gpu_backend_name() == "cuda" ? 1 : 0;
}

FROST_NATIVE_API const char *Frost_GetGpuBackendName() {
  static std::string backendName = FrostInterface::get_gpu_backend_name();
  backendName = FrostInterface::get_gpu_backend_name();
  return backendName.c_str();
}

FROST_NATIVE_API int Frost_HasVulkanRuntime() {
  return probe_frost_vulkan_runtime().hasPhysicalDevice ? 1 : 0;
}

FROST_NATIVE_API const char *Frost_GetVulkanRuntimeStatus() {
  static std::string statusMessage;
  statusMessage = probe_frost_vulkan_runtime().statusMessage;
  return statusMessage.c_str();
}

FROST_NATIVE_API int Frost_HasVulkanCompute() {
  const auto info = probe_frost_vulkan_compute();
  return (info.deviceCreated && info.hasComputeQueue) ? 1 : 0;
}

FROST_NATIVE_API const char *Frost_GetVulkanComputeStatus() {
  static std::string statusMessage;
  statusMessage = probe_frost_vulkan_compute().statusMessage;
  return statusMessage.c_str();
}

FROST_NATIVE_API int Frost_HasVulkanStorageBuffer() {
  return probe_frost_vulkan_storage_buffer().storageBufferReady ? 1 : 0;
}

FROST_NATIVE_API const char *Frost_GetVulkanStorageBufferStatus() {
  static std::string statusMessage;
  statusMessage = probe_frost_vulkan_storage_buffer().statusMessage;
  return statusMessage.c_str();
}
