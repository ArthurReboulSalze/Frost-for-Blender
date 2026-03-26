#include "vulkan_buffer_probe.hpp"

#include <cstdint>
#include <cstring>
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
struct VkAllocationCallbacks;
struct VkBuffer_T;
struct VkDeviceMemory_T;

using VkInstance = VkInstance_T *;
using VkPhysicalDevice = VkPhysicalDevice_T *;
using VkDevice = VkDevice_T *;
using VkBuffer = VkBuffer_T *;
using VkDeviceMemory = VkDeviceMemory_T *;
using VkDeviceSize = uint64_t;
using VkFlags = uint32_t;
using VkQueueFlags = VkFlags;
using VkBufferUsageFlags = VkFlags;
using VkMemoryPropertyFlags = VkFlags;
using VkStructureType = int32_t;
using VkResult = int32_t;
using VkSharingMode = int32_t;

constexpr VkResult VK_SUCCESS = 0;
constexpr VkResult VK_INCOMPLETE = 5;

constexpr VkStructureType VK_STRUCTURE_TYPE_APPLICATION_INFO = 0;
constexpr VkStructureType VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1;
constexpr VkStructureType VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = 2;
constexpr VkStructureType VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 3;
constexpr VkStructureType VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = 5;
constexpr VkStructureType VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO = 12;

constexpr VkQueueFlags VK_QUEUE_COMPUTE_BIT = 0x00000002;
constexpr VkBufferUsageFlags VK_BUFFER_USAGE_STORAGE_BUFFER_BIT = 0x00000020;
constexpr VkMemoryPropertyFlags VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT =
    0x00000002;
constexpr VkMemoryPropertyFlags VK_MEMORY_PROPERTY_HOST_COHERENT_BIT =
    0x00000004;
constexpr VkSharingMode VK_SHARING_MODE_EXCLUSIVE = 0;
constexpr uint32_t VK_MAX_MEMORY_TYPES = 32;
constexpr uint32_t VK_MAX_MEMORY_HEAPS = 16;

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

struct VkBufferCreateInfo {
  VkStructureType sType;
  const void *pNext;
  VkFlags flags;
  VkDeviceSize size;
  VkBufferUsageFlags usage;
  VkSharingMode sharingMode;
  uint32_t queueFamilyIndexCount;
  const uint32_t *pQueueFamilyIndices;
};

struct VkMemoryRequirements {
  VkDeviceSize size;
  VkDeviceSize alignment;
  uint32_t memoryTypeBits;
};

struct VkMemoryAllocateInfo {
  VkStructureType sType;
  const void *pNext;
  VkDeviceSize allocationSize;
  uint32_t memoryTypeIndex;
};

struct VkMemoryType {
  VkMemoryPropertyFlags propertyFlags;
  uint32_t heapIndex;
};

struct VkMemoryHeap {
  VkDeviceSize size;
  VkFlags flags;
};

struct VkPhysicalDeviceMemoryProperties {
  uint32_t memoryTypeCount;
  VkMemoryType memoryTypes[VK_MAX_MEMORY_TYPES];
  uint32_t memoryHeapCount;
  VkMemoryHeap memoryHeaps[VK_MAX_MEMORY_HEAPS];
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
using PFN_vkGetPhysicalDeviceMemoryProperties =
    void (*)(VkPhysicalDevice physicalDevice,
             VkPhysicalDeviceMemoryProperties *pMemoryProperties);
using PFN_vkCreateDevice =
    VkResult (*)(VkPhysicalDevice physicalDevice,
                 const VkDeviceCreateInfo *pCreateInfo,
                 const VkAllocationCallbacks *pAllocator, VkDevice *pDevice);
using PFN_vkGetDeviceProcAddr = PFN_vkVoidFunction (*)(VkDevice device,
                                                       const char *pName);
using PFN_vkDestroyDevice = void (*)(VkDevice device,
                                     const VkAllocationCallbacks *pAllocator);
using PFN_vkCreateBuffer =
    VkResult (*)(VkDevice device, const VkBufferCreateInfo *pCreateInfo,
                 const VkAllocationCallbacks *pAllocator, VkBuffer *pBuffer);
using PFN_vkDestroyBuffer = void (*)(VkDevice device, VkBuffer buffer,
                                     const VkAllocationCallbacks *pAllocator);
using PFN_vkGetBufferMemoryRequirements =
    void (*)(VkDevice device, VkBuffer buffer,
             VkMemoryRequirements *pMemoryRequirements);
using PFN_vkAllocateMemory =
    VkResult (*)(VkDevice device, const VkMemoryAllocateInfo *pAllocateInfo,
                 const VkAllocationCallbacks *pAllocator,
                 VkDeviceMemory *pMemory);
using PFN_vkFreeMemory = void (*)(VkDevice device, VkDeviceMemory memory,
                                  const VkAllocationCallbacks *pAllocator);
using PFN_vkBindBufferMemory =
    VkResult (*)(VkDevice device, VkBuffer buffer, VkDeviceMemory memory,
                 VkDeviceSize memoryOffset);
using PFN_vkMapMemory =
    VkResult (*)(VkDevice device, VkDeviceMemory memory, VkDeviceSize offset,
                 VkDeviceSize size, VkFlags flags, void **ppData);
using PFN_vkUnmapMemory = void (*)(VkDevice device, VkDeviceMemory memory);

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

bool find_memory_type_index(const VkPhysicalDeviceMemoryProperties &properties,
                            uint32_t memoryTypeBits,
                            VkMemoryPropertyFlags requiredFlags,
                            uint32_t &outIndex) {
  for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
    const bool supportedByBuffer = (memoryTypeBits & (1u << i)) != 0;
    const bool hasRequiredFlags =
        (properties.memoryTypes[i].propertyFlags & requiredFlags) ==
        requiredFlags;
    if (supportedByBuffer && hasRequiredFlags) {
      outIndex = i;
      return true;
    }
  }
  return false;
}

#endif

} // namespace

FrostVulkanBufferInfo probe_frost_vulkan_storage_buffer() {
  FrostVulkanBufferInfo info;

#if !defined(FROST_ENABLE_VULKAN_PROBE)
  info.statusMessage = "Vulkan buffer probe is disabled in this native build.";
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
  auto vkGetPhysicalDeviceMemoryProperties =
      reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(
          vkGetInstanceProcAddr(instance,
                                "vkGetPhysicalDeviceMemoryProperties"));
  auto vkCreateDevice = reinterpret_cast<PFN_vkCreateDevice>(
      vkGetInstanceProcAddr(instance, "vkCreateDevice"));
  auto vkGetDeviceProcAddr = reinterpret_cast<PFN_vkGetDeviceProcAddr>(
      vkGetInstanceProcAddr(instance, "vkGetDeviceProcAddr"));

  if (!vkDestroyInstance || !vkEnumeratePhysicalDevices ||
      !vkGetPhysicalDeviceQueueFamilyProperties ||
      !vkGetPhysicalDeviceMemoryProperties || !vkCreateDevice ||
      !vkGetDeviceProcAddr) {
    info.statusMessage =
        "Vulkan instance created, but required buffer setup functions are missing.";
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
    stream << "Vulkan physical device enumeration failed (VkResult "
           << enumerateResult << ").";
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
    stream << "Vulkan device creation failed (VkResult " << createDeviceResult
           << ").";
    info.statusMessage = stream.str();
    vkDestroyInstance(instance, nullptr);
    close_vulkan_library(library);
    return info;
  }

  info.deviceCreated = true;

  auto vkDestroyDevice = reinterpret_cast<PFN_vkDestroyDevice>(
      vkGetDeviceProcAddr(device, "vkDestroyDevice"));
  auto vkCreateBuffer = reinterpret_cast<PFN_vkCreateBuffer>(
      vkGetDeviceProcAddr(device, "vkCreateBuffer"));
  auto vkDestroyBuffer = reinterpret_cast<PFN_vkDestroyBuffer>(
      vkGetDeviceProcAddr(device, "vkDestroyBuffer"));
  auto vkGetBufferMemoryRequirements =
      reinterpret_cast<PFN_vkGetBufferMemoryRequirements>(
          vkGetDeviceProcAddr(device, "vkGetBufferMemoryRequirements"));
  auto vkAllocateMemory = reinterpret_cast<PFN_vkAllocateMemory>(
      vkGetDeviceProcAddr(device, "vkAllocateMemory"));
  auto vkFreeMemory = reinterpret_cast<PFN_vkFreeMemory>(
      vkGetDeviceProcAddr(device, "vkFreeMemory"));
  auto vkBindBufferMemory = reinterpret_cast<PFN_vkBindBufferMemory>(
      vkGetDeviceProcAddr(device, "vkBindBufferMemory"));
  auto vkMapMemory = reinterpret_cast<PFN_vkMapMemory>(
      vkGetDeviceProcAddr(device, "vkMapMemory"));
  auto vkUnmapMemory = reinterpret_cast<PFN_vkUnmapMemory>(
      vkGetDeviceProcAddr(device, "vkUnmapMemory"));

  if (!vkDestroyDevice || !vkCreateBuffer || !vkDestroyBuffer ||
      !vkGetBufferMemoryRequirements || !vkAllocateMemory || !vkFreeMemory ||
      !vkBindBufferMemory || !vkMapMemory || !vkUnmapMemory) {
    info.statusMessage =
        "Vulkan device created, but required buffer functions are missing.";
    if (vkDestroyDevice) {
      vkDestroyDevice(device, nullptr);
    }
    vkDestroyInstance(instance, nullptr);
    close_vulkan_library(library);
    return info;
  }

  VkBufferCreateInfo bufferCreateInfo{};
  bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferCreateInfo.size = 4096;
  bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VkBuffer buffer = nullptr;
  const VkResult createBufferResult =
      vkCreateBuffer(device, &bufferCreateInfo, nullptr, &buffer);
  if (createBufferResult != VK_SUCCESS || !buffer) {
    std::ostringstream stream;
    stream << "Vulkan storage buffer creation failed (VkResult "
           << createBufferResult << ").";
    info.statusMessage = stream.str();
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    close_vulkan_library(library);
    return info;
  }

  VkMemoryRequirements memoryRequirements{};
  vkGetBufferMemoryRequirements(device, buffer, &memoryRequirements);

  VkPhysicalDeviceMemoryProperties memoryProperties{};
  vkGetPhysicalDeviceMemoryProperties(selectedPhysicalDevice, &memoryProperties);

  uint32_t memoryTypeIndex = 0;
  const VkMemoryPropertyFlags requiredFlags =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  if (!find_memory_type_index(memoryProperties, memoryRequirements.memoryTypeBits,
                              requiredFlags, memoryTypeIndex)) {
    info.statusMessage =
        "Vulkan storage buffer created, but no host-visible coherent memory type was found.";
    vkDestroyBuffer(device, buffer, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    close_vulkan_library(library);
    return info;
  }

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memoryRequirements.size;
  allocInfo.memoryTypeIndex = memoryTypeIndex;

  VkDeviceMemory memory = nullptr;
  const VkResult allocateMemoryResult =
      vkAllocateMemory(device, &allocInfo, nullptr, &memory);
  if (allocateMemoryResult != VK_SUCCESS || !memory) {
    std::ostringstream stream;
    stream << "Vulkan buffer memory allocation failed (VkResult "
           << allocateMemoryResult << ").";
    info.statusMessage = stream.str();
    vkDestroyBuffer(device, buffer, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    close_vulkan_library(library);
    return info;
  }

  const VkResult bindMemoryResult =
      vkBindBufferMemory(device, buffer, memory, 0);
  if (bindMemoryResult != VK_SUCCESS) {
    std::ostringstream stream;
    stream << "Vulkan buffer memory bind failed (VkResult " << bindMemoryResult
           << ").";
    info.statusMessage = stream.str();
    vkFreeMemory(device, memory, nullptr);
    vkDestroyBuffer(device, buffer, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    close_vulkan_library(library);
    return info;
  }

  void *mappedPtr = nullptr;
  const VkResult mapResult = vkMapMemory(device, memory, 0, allocInfo.allocationSize,
                                         0, &mappedPtr);
  if (mapResult != VK_SUCCESS || !mappedPtr) {
    std::ostringstream stream;
    stream << "Vulkan buffer map failed (VkResult " << mapResult << ").";
    info.statusMessage = stream.str();
    vkFreeMemory(device, memory, nullptr);
    vkDestroyBuffer(device, buffer, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    close_vulkan_library(library);
    return info;
  }

  unsigned char pattern[16];
  for (size_t i = 0; i < sizeof(pattern); ++i) {
    pattern[i] = static_cast<unsigned char>(i);
  }
  std::memcpy(mappedPtr, pattern, sizeof(pattern));
  const bool roundTripOk = std::memcmp(mappedPtr, pattern, sizeof(pattern)) == 0;
  vkUnmapMemory(device, memory);

  if (!roundTripOk) {
    info.statusMessage =
        "Vulkan storage buffer was allocated and mapped, but the host round-trip check failed.";
    vkFreeMemory(device, memory, nullptr);
    vkDestroyBuffer(device, buffer, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    close_vulkan_library(library);
    return info;
  }

  info.storageBufferReady = true;
  info.allocatedBytes = allocInfo.allocationSize;

  std::ostringstream stream;
  stream << "Vulkan storage buffer path is ready (API "
         << format_api_version(info.apiVersion) << ", "
         << info.physicalDeviceCount << " physical device(s), queue family "
         << info.computeQueueFamilyIndex << ", "
         << info.allocatedBytes
         << " bytes allocated). The Frost experimental Vulkan backend can "
            "already feed an initial compute shader with particle buffers; "
            "full field evaluation is still pending.";
  info.statusMessage = stream.str();

  vkFreeMemory(device, memory, nullptr);
  vkDestroyBuffer(device, buffer, nullptr);
  vkDestroyDevice(device, nullptr);
  vkDestroyInstance(instance, nullptr);
  close_vulkan_library(library);
  return info;
}
