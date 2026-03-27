#pragma once

#include <stddef.h>

#ifdef _WIN32
#define FROST_NATIVE_API extern "C" __declspec(dllexport)
#else
#define FROST_NATIVE_API extern "C"
#endif

FROST_NATIVE_API void *Frost_Create();
FROST_NATIVE_API void Frost_Destroy(void *handle);

FROST_NATIVE_API int Frost_SetParticles(void *handle, const float *positions,
                                        size_t particleCount,
                                        const float *radii,
                                        const float *velocities);

FROST_NATIVE_API int Frost_SetParameterBool(void *handle, const char *name,
                                            int value);
FROST_NATIVE_API int Frost_SetParameterInt(void *handle, const char *name,
                                           int value);
FROST_NATIVE_API int Frost_SetParameterFloat(void *handle, const char *name,
                                             float value);

FROST_NATIVE_API int Frost_GenerateMesh(void *handle);
FROST_NATIVE_API int Frost_GetVertexCount(void *handle);
FROST_NATIVE_API int Frost_GetFaceCount(void *handle);
FROST_NATIVE_API int Frost_CopyVertices(void *handle, float *outVertices,
                                        size_t floatCount);
FROST_NATIVE_API int Frost_CopyFaces(void *handle, int *outFaces,
                                     size_t intCount);

FROST_NATIVE_API const char *Frost_GetLastError(void *handle);
FROST_NATIVE_API const char *Frost_GetLastMeshingBackend(void *handle);
FROST_NATIVE_API const char *Frost_GetLastMeshingStatus(void *handle);
FROST_NATIVE_API int Frost_GetLastMeshingUsedFallback(void *handle);
FROST_NATIVE_API int Frost_HasGpuBackend();
FROST_NATIVE_API int Frost_HasCudaBackend();
FROST_NATIVE_API const char *Frost_GetGpuBackendName();
FROST_NATIVE_API int Frost_HasVulkanRuntime();
FROST_NATIVE_API const char *Frost_GetVulkanRuntimeStatus();
FROST_NATIVE_API int Frost_HasVulkanCompute();
FROST_NATIVE_API const char *Frost_GetVulkanComputeStatus();
FROST_NATIVE_API int Frost_HasVulkanStorageBuffer();
FROST_NATIVE_API const char *Frost_GetVulkanStorageBufferStatus();
