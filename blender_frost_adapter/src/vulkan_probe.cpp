#include "vulkan_probe.hpp"

#include <cstdint>
#include <sstream>
#include <string>

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
struct VkAllocationCallbacks;

using VkInstance = VkInstance_T *;
using VkPhysicalDevice = VkPhysicalDevice_T *;
using VkFlags = uint32_t;
using VkBool32 = uint32_t;
using VkStructureType = int32_t;
using VkResult = int32_t;

constexpr VkResult VK_SUCCESS = 0;
constexpr VkResult VK_INCOMPLETE = 5;

constexpr VkStructureType VK_STRUCTURE_TYPE_APPLICATION_INFO = 0;
constexpr VkStructureType VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1;

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

FrostVulkanRuntimeInfo probe_frost_vulkan_runtime() {
  FrostVulkanRuntimeInfo info;

#if !defined(FROST_ENABLE_VULKAN_PROBE)
  info.statusMessage = "Vulkan runtime probe is disabled in this native build.";
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
  const VkResult createResult =
      vkCreateInstance(&instanceInfo, nullptr, &instance);
  if (createResult != VK_SUCCESS || !instance) {
    std::ostringstream stream;
    stream << "Vulkan runtime found, but instance creation failed (VkResult "
           << createResult << ").";
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

  if (!vkDestroyInstance || !vkEnumeratePhysicalDevices) {
    info.statusMessage =
        "Vulkan instance created, but required instance functions are missing.";
    if (vkDestroyInstance) {
      vkDestroyInstance(instance, nullptr);
    }
    close_vulkan_library(library);
    return info;
  }

  uint32_t physicalDeviceCount = 0;
  const VkResult enumerateResult =
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
  info.hasPhysicalDevice = physicalDeviceCount > 0;

  std::ostringstream stream;
  if (info.hasPhysicalDevice) {
    stream << "Vulkan runtime detected (API "
           << format_api_version(info.apiVersion) << ", "
           << info.physicalDeviceCount
           << " physical device(s)). The Frost experimental Vulkan backend can "
              "use this runtime and now reaches a first compute-shader stage; "
              "full field meshing is still in progress.";
  } else {
    stream << "Vulkan runtime detected (API "
           << format_api_version(info.apiVersion)
           << "), but no physical device was enumerated.";
  }
  info.statusMessage = stream.str();

  vkDestroyInstance(instance, nullptr);
  close_vulkan_library(library);
  return info;
}
