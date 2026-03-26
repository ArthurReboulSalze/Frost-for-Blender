#include "vulkan_compute_probe.hpp"

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

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

#if defined(FROST_ENABLE_VULKAN_PROBE)

struct VkInstance_T;
struct VkPhysicalDevice_T;
struct VkDevice_T;
struct VkQueue_T;
struct VkAllocationCallbacks;

using VkInstance = VkInstance_T *;
using VkPhysicalDevice = VkPhysicalDevice_T *;
using VkDevice = VkDevice_T *;
using VkQueue = VkQueue_T *;
using VkFlags = uint32_t;
using VkQueueFlags = VkFlags;
using VkStructureType = int32_t;
using VkResult = int32_t;

constexpr VkResult VK_SUCCESS = 0;
constexpr VkResult VK_INCOMPLETE = 5;

constexpr VkStructureType VK_STRUCTURE_TYPE_APPLICATION_INFO = 0;
constexpr VkStructureType VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1;
constexpr VkStructureType VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = 2;
constexpr VkStructureType VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 3;

constexpr VkQueueFlags VK_QUEUE_COMPUTE_BIT = 0x00000002;

constexpr uint32_t make_vk_api_version(uint32_t variant, uint32_t major,
                                       uint32_t minor, uint32_t patch) {
  return (variant << 29) | (major << 22) | (minor << 12) | patch;
}

constexpr uint32_t VK_API_VERSION_1_0 = make_vk_api_version(0, 1, 0, 0);

constexpr uint32_t vk_api_version_major(uint32_t version) {
  return (version >> 22) & 0x7F;
}

constexpr uint32_t vk_api_version_minor(uint32_t version) {
  return (version >> 12) & 0x3FF;
}

constexpr uint32_t vk_api_version_patch(uint32_t version) {
  return version & 0xFFF;
}

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

std::string format_api_version(uint32_t version) {
  std::ostringstream stream;
  stream << vk_api_version_major(version) << "."
         << vk_api_version_minor(version) << "."
         << vk_api_version_patch(version);
  return stream.str();
}

#endif

} // namespace

FrostVulkanComputeInfo probe_frost_vulkan_compute() {
  FrostVulkanComputeInfo info;

#if !defined(FROST_ENABLE_VULKAN_PROBE)
  info.statusMessage = "Vulkan compute probe is disabled in this native build.";
  return info;
#endif

  LibraryHandle library = load_vulkan_library();
  if (!library) {
    info.statusMessage =
        "Vulkan runtime not found on this system (vulkan-1.dll / libvulkan.so.1 missing).";
    return info;
  }

  info.loaderPresent = true;

  auto vkGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
      load_symbol(library, "vkGetInstanceProcAddr"));
  if (!vkGetInstanceProcAddr) {
    info.statusMessage =
        "Vulkan loader found, but vkGetInstanceProcAddr is unavailable.";
    close_vulkan_library(library);
    return info;
  }

  auto vkEnumerateInstanceVersion =
      reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
          vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceVersion"));
  info.apiVersion = VK_API_VERSION_1_0;
  if (vkEnumerateInstanceVersion) {
    uint32_t apiVersion = info.apiVersion;
    if (vkEnumerateInstanceVersion(&apiVersion) == VK_SUCCESS) {
      info.apiVersion = apiVersion;
    }
  }

  auto vkCreateInstance = reinterpret_cast<PFN_vkCreateInstance>(
      vkGetInstanceProcAddr(nullptr, "vkCreateInstance"));
  if (!vkCreateInstance) {
    info.statusMessage =
        "Vulkan loader found, but vkCreateInstance is unavailable.";
    close_vulkan_library(library);
    return info;
  }

  VkApplicationInfo appInfo{};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "Frost for Blender";
  appInfo.applicationVersion = 1;
  appInfo.pEngineName = "Frost";
  appInfo.engineVersion = 1;
  appInfo.apiVersion = info.apiVersion;

  VkInstanceCreateInfo instanceInfo{};
  instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instanceInfo.pApplicationInfo = &appInfo;

  VkInstance instance = nullptr;
  const VkResult createInstanceResult =
      vkCreateInstance(&instanceInfo, nullptr, &instance);
  if (createInstanceResult != VK_SUCCESS || !instance) {
    std::ostringstream stream;
    stream << "Vulkan runtime found, but instance creation failed (VkResult "
           << createInstanceResult << ").";
    info.statusMessage = stream.str();
    close_vulkan_library(library);
    return info;
  }

  info.instanceCreated = true;

  auto vkDestroyInstance = reinterpret_cast<PFN_vkDestroyInstance>(
      vkGetInstanceProcAddr(instance, "vkDestroyInstance"));
  auto vkEnumeratePhysicalDevices =
      reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(
          vkGetInstanceProcAddr(instance, "vkEnumeratePhysicalDevices"));
  auto vkGetPhysicalDeviceQueueFamilyProperties =
      reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
          vkGetInstanceProcAddr(instance,
                                "vkGetPhysicalDeviceQueueFamilyProperties"));
  auto vkCreateDevice = reinterpret_cast<PFN_vkCreateDevice>(
      vkGetInstanceProcAddr(instance, "vkCreateDevice"));
  auto vkGetDeviceProcAddr = reinterpret_cast<PFN_vkGetDeviceProcAddr>(
      vkGetInstanceProcAddr(instance, "vkGetDeviceProcAddr"));

  if (!vkDestroyInstance || !vkEnumeratePhysicalDevices ||
      !vkGetPhysicalDeviceQueueFamilyProperties || !vkCreateDevice ||
      !vkGetDeviceProcAddr) {
    info.statusMessage =
        "Vulkan instance created, but required device creation functions are missing.";
    if (vkDestroyInstance) {
      vkDestroyInstance(instance, nullptr);
    }
    close_vulkan_library(library);
    return info;
  }

  uint32_t physicalDeviceCount = 0;
  VkResult enumerateResult =
      vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, nullptr);
  if (enumerateResult != VK_SUCCESS && enumerateResult != VK_INCOMPLETE) {
    std::ostringstream stream;
    stream << "Vulkan instance created, but physical device enumeration failed"
           << " (VkResult " << enumerateResult << ").";
    info.statusMessage = stream.str();
    vkDestroyInstance(instance, nullptr);
    close_vulkan_library(library);
    return info;
  }

  info.physicalDeviceCount = physicalDeviceCount;
  if (physicalDeviceCount == 0) {
    info.statusMessage =
        "Vulkan runtime detected, but no physical device was enumerated.";
    vkDestroyInstance(instance, nullptr);
    close_vulkan_library(library);
    return info;
  }

  std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
  enumerateResult = vkEnumeratePhysicalDevices(instance, &physicalDeviceCount,
                                               physicalDevices.data());
  if (enumerateResult != VK_SUCCESS && enumerateResult != VK_INCOMPLETE) {
    std::ostringstream stream;
    stream << "Vulkan physical device listing failed (VkResult "
           << enumerateResult << ").";
    info.statusMessage = stream.str();
    vkDestroyInstance(instance, nullptr);
    close_vulkan_library(library);
    return info;
  }

  VkPhysicalDevice selectedPhysicalDevice = nullptr;
  uint32_t selectedQueueFamilyIndex = 0xFFFFFFFFu;

  for (VkPhysicalDevice physicalDevice : physicalDevices) {
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount,
                                             nullptr);
    if (queueFamilyCount == 0) {
      continue;
    }

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount,
                                             queueFamilies.data());

    for (uint32_t i = 0; i < queueFamilyCount; ++i) {
      if (queueFamilies[i].queueCount > 0 &&
          (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
        selectedPhysicalDevice = physicalDevice;
        selectedQueueFamilyIndex = i;
        break;
      }
    }

    if (selectedPhysicalDevice) {
      break;
    }
  }

  if (!selectedPhysicalDevice || selectedQueueFamilyIndex == 0xFFFFFFFFu) {
    info.statusMessage =
        "Vulkan runtime detected, but no compute-capable queue family was found.";
    vkDestroyInstance(instance, nullptr);
    close_vulkan_library(library);
    return info;
  }

  info.computeQueueFamilyIndex = selectedQueueFamilyIndex;
  info.hasComputeQueue = true;

  float queuePriority = 1.0f;
  VkDeviceQueueCreateInfo queueCreateInfo{};
  queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueCreateInfo.queueFamilyIndex = selectedQueueFamilyIndex;
  queueCreateInfo.queueCount = 1;
  queueCreateInfo.pQueuePriorities = &queuePriority;

  VkDeviceCreateInfo deviceCreateInfo{};
  deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  deviceCreateInfo.queueCreateInfoCount = 1;
  deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;

  VkDevice device = nullptr;
  const VkResult createDeviceResult =
      vkCreateDevice(selectedPhysicalDevice, &deviceCreateInfo, nullptr,
                     &device);
  if (createDeviceResult != VK_SUCCESS || !device) {
    std::ostringstream stream;
    stream << "Vulkan compute queue was found, but logical device creation "
              "failed (VkResult "
           << createDeviceResult << ").";
    info.statusMessage = stream.str();
    vkDestroyInstance(instance, nullptr);
    close_vulkan_library(library);
    return info;
  }

  info.deviceCreated = true;

  auto vkDestroyDevice = reinterpret_cast<PFN_vkDestroyDevice>(
      vkGetDeviceProcAddr(device, "vkDestroyDevice"));
  auto vkGetDeviceQueue = reinterpret_cast<PFN_vkGetDeviceQueue>(
      vkGetDeviceProcAddr(device, "vkGetDeviceQueue"));

  if (!vkDestroyDevice || !vkGetDeviceQueue) {
    info.statusMessage =
        "Vulkan logical device created, but queue retrieval functions are missing.";
    if (vkDestroyDevice) {
      vkDestroyDevice(device, nullptr);
    }
    vkDestroyInstance(instance, nullptr);
    close_vulkan_library(library);
    return info;
  }

  VkQueue queue = nullptr;
  vkGetDeviceQueue(device, selectedQueueFamilyIndex, 0, &queue);
  if (!queue) {
    info.statusMessage =
        "Vulkan logical device created, but no compute queue handle was returned.";
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    close_vulkan_library(library);
    return info;
  }

  std::ostringstream stream;
  stream << "Vulkan compute context is ready (API "
         << format_api_version(info.apiVersion) << ", "
         << info.physicalDeviceCount << " physical device(s), queue family "
         << info.computeQueueFamilyIndex
         << "). The Frost experimental Vulkan backend can already create "
            "devices and queues, and now supports an initial compute shader "
            "dispatch.";
  info.statusMessage = stream.str();

  vkDestroyDevice(device, nullptr);
  vkDestroyInstance(instance, nullptr);
  close_vulkan_library(library);
  return info;
}
