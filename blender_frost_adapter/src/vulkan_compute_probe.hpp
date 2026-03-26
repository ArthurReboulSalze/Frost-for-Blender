#pragma once

#include <cstdint>
#include <string>

struct FrostVulkanComputeInfo {
  bool loaderPresent = false;
  bool instanceCreated = false;
  bool deviceCreated = false;
  bool hasComputeQueue = false;
  uint32_t apiVersion = 0;
  uint32_t physicalDeviceCount = 0;
  uint32_t computeQueueFamilyIndex = 0xFFFFFFFFu;
  std::string statusMessage;
};

FrostVulkanComputeInfo probe_frost_vulkan_compute();
