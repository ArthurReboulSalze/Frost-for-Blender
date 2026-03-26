#pragma once

#include <memory>

#include "gpu_backend.hpp"

std::unique_ptr<FrostGpuBackend> create_vulkan_gpu_backend();
FrostGpuBackendInfo get_vulkan_gpu_backend_info();
