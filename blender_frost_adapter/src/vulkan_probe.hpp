#pragma once

#include <cstdint>
#include <string>

struct FrostVulkanRuntimeInfo {
  bool loaderPresent = false;
  bool instanceCreated = false;
  bool hasPhysicalDevice = false;
  uint32_t apiVersion = 0;
  uint32_t physicalDeviceCount = 0;
  std::string statusMessage;
};

FrostVulkanRuntimeInfo probe_frost_vulkan_runtime();
