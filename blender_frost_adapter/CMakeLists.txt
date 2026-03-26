cmake_minimum_required(VERSION 3.15)
project(blender_frost_adapter)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(FROST_ENABLE_CUDA "Build the CUDA backend" OFF)
option(FROST_ENABLE_VULKAN_PROBE "Build Vulkan support and runtime probes" ON)
option(FROST_BUILD_PYTHON_MODULE "Build the legacy pybind11 module" OFF)

# Global settings that are safe/required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add Thinkbox CM Library to module path
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/../deps/thinkbox-cm-library)

find_package(Boost REQUIRED)
find_package(TBB REQUIRED)
find_package(OpenEXR REQUIRED)
find_package(tinyxml2 REQUIRED)
find_package(ZLIB REQUIRED)
find_package(xxHash REQUIRED)
find_package(glslang CONFIG REQUIRED)

if(FROST_BUILD_PYTHON_MODULE)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
endif()

# Include directories will be added to target

# Disable unit tests for subdirectories
set(BUILD_UNIT_TESTS OFF CACHE BOOL "" FORCE)
# Disable unnecessary IO dependencies
set(BUILD_WITH_E57 OFF CACHE BOOL "" FORCE)
set(BUILD_WITH_LZ4 OFF CACHE BOOL "" FORCE)
set(BUILD_WITH_TBB ON CACHE BOOL "" FORCE)

if(TARGET Boost::boost AND NOT TARGET Boost::Boost)
    add_library(Boost::Boost ALIAS Boost::boost)
endif()

if(TARGET xxHash::xxhash AND NOT TARGET xxHash::xxHash)
    add_library(xxHash::xxHash ALIAS xxHash::xxhash)
endif()

# Add subdirectories
# Note: We assume these are available in the parent directory
add_subdirectory(${CMAKE_SOURCE_DIR}/../deps/thinkbox-library thinkbox-library_build)
add_subdirectory(${CMAKE_SOURCE_DIR}/../thinkbox-frost thinkbox-frost_build)

if(FROST_ENABLE_CUDA)
    # Add CUDA Subdirectory (Isolates toolchain)
    add_subdirectory(src/cuda)

    # Add cuNSearch GPU Neighbor Search
    set(BUILD_DEMO OFF CACHE BOOL "" FORCE)
    set(CUNSEARCH_USE_DOUBLE_PRECISION OFF CACHE BOOL "" FORCE) # Use float for Frost
    set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "" FORCE) # Fix empty arch warning
    # Override C++14 to C++17 for CUDA 12.9 compatibility
    set(CMAKE_CXX_STANDARD 17 CACHE STRING "" FORCE)
    add_subdirectory(${CMAKE_SOURCE_DIR}/../deps/cuNSearch cuNSearch_build)
endif()

# Add source files
set(FROST_NATIVE_SOURCES
    src/frost_interface.cpp
    src/gpu_backend.cpp
    src/frost_native_api.cpp
    src/vulkan_backend.cpp
    src/vulkan_compute_shader.cpp
    src/vulkan_probe.cpp
    src/vulkan_compute_probe.cpp
    src/vulkan_buffer_probe.cpp
)

# Create stable native library (usable from ctypes across Blender Python versions)
add_library(frost_native SHARED ${FROST_NATIVE_SOURCES})
set_target_properties(frost_native PROPERTIES OUTPUT_NAME "frost_native")

# Link libraries
target_link_libraries(frost_native PRIVATE
    frost
    thinkboxlibrary
    Boost::boost
    glslang::glslang
    glslang::SPIRV
    glslang::OSDependent
    glslang::MachineIndependent
    glslang::GenericCodeGen
    glslang::glslang-default-resource-limits
    OpenEXR::OpenEXR
    TBB::tbb
    tinyxml2::tinyxml2
    ZLIB::ZLIB
)

if(FROST_ENABLE_CUDA)
    target_link_libraries(frost_native PRIVATE
        frost_cuda
        cuNSearch
    )
endif()

# Apply Forced Include only to this target (and its dependencies that need it)
target_compile_options(frost_native PRIVATE /FI"${CMAKE_SOURCE_DIR}/src/compat_headers.hpp")
# Apply general compile options
target_compile_options(frost_native PRIVATE /wd4834)
# Apply definitions
target_compile_definitions(frost_native PRIVATE 
    BOOST_BIND_GLOBAL_PLACEHOLDERS
    UNICODE _UNICODE NOMINMAX
)

if(FROST_ENABLE_CUDA)
    target_compile_definitions(frost_native PRIVATE FROST_ENABLE_CUDA)
endif()

if(FROST_ENABLE_VULKAN_PROBE)
    target_compile_definitions(frost_native PRIVATE FROST_ENABLE_VULKAN_PROBE)
endif()

# Include directories (Moved from global)
target_include_directories(frost_native PRIVATE
    ${CMAKE_SOURCE_DIR}/inc_compat
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/../deps/thinkbox-library/include
    ${CMAKE_SOURCE_DIR}/../thinkbox-frost
    ${CMAKE_SOURCE_DIR}/../deps/Vulkan-Headers/include
    ${CMAKE_SOURCE_DIR}/../deps/vcpkg/installed/x64-windows/include
    ${CMAKE_SOURCE_DIR}/../deps/vcpkg/installed/x64-windows/include/eigen3
    ${CMAKE_SOURCE_DIR}/../deps/vcpkg/installed/x64-windows/include/Imath
    ${CMAKE_SOURCE_DIR}/../deps/vcpkg/installed/x64-windows/include/OpenEXR
)

if(FROST_BUILD_PYTHON_MODULE)
    pybind11_add_module(blender_frost_adapter src/adapter.cpp)
    target_link_libraries(blender_frost_adapter PRIVATE
        frost_native
        TBB::tbb
    )
    target_compile_options(blender_frost_adapter PRIVATE /wd4834)
    target_compile_definitions(blender_frost_adapter PRIVATE
        BOOST_BIND_GLOBAL_PLACEHOLDERS
        UNICODE _UNICODE NOMINMAX
    )
    target_include_directories(blender_frost_adapter PRIVATE
        ${CMAKE_SOURCE_DIR}/src
    )
endif()

# Dependencies also need the shim for Boost/Imath compatibility and warning suppression
if(TARGET frost)
    target_compile_options(frost PRIVATE /FI"${CMAKE_SOURCE_DIR}/src/compat_headers.hpp")
    target_compile_options(frost PRIVATE /wd4834)
    # Frost needs Imath/OpenEXR includes (previously inherited from global scope)
    target_include_directories(frost PUBLIC
        ${CMAKE_SOURCE_DIR}/inc_compat
        ${CMAKE_SOURCE_DIR}/../deps/vcpkg/installed/x64-windows/include
        ${CMAKE_SOURCE_DIR}/../deps/vcpkg/installed/x64-windows/include/Imath
        ${CMAKE_SOURCE_DIR}/../deps/vcpkg/installed/x64-windows/include/OpenEXR
    )
endif()
if(TARGET thinkboxlibrary)
    target_compile_options(thinkboxlibrary PRIVATE /FI"${CMAKE_SOURCE_DIR}/src/compat_headers.hpp")
    target_compile_options(thinkboxlibrary PRIVATE /wd4834)
endif()

# Fix includes for thinkboxlibrary (Frost already handled by reverting to original)
target_include_directories(thinkboxlibrary PUBLIC
    ${CMAKE_SOURCE_DIR}/inc_compat
    ${CMAKE_SOURCE_DIR}/../deps/utf8cpp/source
    ${CMAKE_SOURCE_DIR}/../deps/vcpkg/installed/x64-windows/include
    ${CMAKE_SOURCE_DIR}/../deps/vcpkg/installed/x64-windows/include/eigen3
    ${CMAKE_SOURCE_DIR}/../deps/vcpkg/installed/x64-windows/include/Imath
    ${CMAKE_SOURCE_DIR}/../deps/vcpkg/installed/x64-windows/include/OpenEXR
)

# Copy to addon folder matching compilation output to Blender version
# (Handled by install or post-build)
