#pragma once

#include <cstdint>
#include <string>

struct FrostVulkanBufferInfo {
  bool loaderPresent = false;
  bool instanceCreated = false;
  bool deviceCreated = false;
  bool storageBufferReady = false;
  uint32_t apiVersion = 0;
  uint32_t physicalDeviceCount = 0;
  uint32_t computeQueueFamilyIndex = 0xFFFFFFFFu;
  uint64_t allocatedBytes = 0;
  std::string statusMessage;
};

FrostVulkanBufferInfo probe_frost_vulkan_storage_buffer();
