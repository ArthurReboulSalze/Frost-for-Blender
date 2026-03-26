#include "frost_interface.hpp"
#include "vulkan_buffer_probe.hpp"
#include "vulkan_compute_probe.hpp"
#include "vulkan_probe.hpp"
#include <cstring>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <tbb/global_control.h>
#include <thread>

namespace py = pybind11;

// Force TBB to use all available hardware threads
// This persistent object ensures TBB uses all 24 threads for the entire module
// lifetime
static std::unique_ptr<tbb::global_control> g_tbb_thread_control;

namespace {

void set_particles_wrapper(FrostInterface &frost,
                           py::array_t<float, py::array::c_style |
                                                  py::array::forcecast>
                               positions,
                           py::array_t<float, py::array::c_style |
                                                  py::array::forcecast>
                               radii,
                           py::object velocitiesObj) {
  if (positions.ndim() != 2 || positions.shape(1) != 3) {
    throw std::runtime_error("positions must be a float32 array shaped [N, 3]");
  }

  if (radii.ndim() != 1 || radii.shape(0) != positions.shape(0)) {
    throw std::runtime_error("radii must be a float32 array shaped [N]");
  }

  const float *velocityPtr = nullptr;
  py::array_t<float, py::array::c_style | py::array::forcecast> velocities;
  if (!velocitiesObj.is_none()) {
    velocities =
        velocitiesObj.cast<py::array_t<float, py::array::c_style |
                                                  py::array::forcecast>>();
    if (velocities.ndim() != 2 || velocities.shape(0) != positions.shape(0) ||
        velocities.shape(1) != 3) {
      throw std::runtime_error(
          "velocities must be a float32 array shaped [N, 3]");
    }
    velocityPtr = velocities.data();
  }

  frost.set_particles(positions.data(), (size_t)positions.shape(0), radii.data(),
                      velocityPtr);
}

void set_parameter_wrapper(FrostInterface &frost, const std::string &name,
                           py::object value) {
  if (py::isinstance<py::bool_>(value)) {
    frost.set_parameter(name, value.cast<bool>());
    return;
  }

  if (py::isinstance<py::int_>(value)) {
    frost.set_parameter(name, value.cast<int>());
    return;
  }

  if (py::isinstance<py::float_>(value)) {
    frost.set_parameter(name, value.cast<float>());
    return;
  }

  throw std::runtime_error("unsupported Frost parameter type");
}

void set_parameters_wrapper(FrostInterface &frost, py::dict params) {
  for (auto item : params) {
    std::string key = py::str(item.first);
    py::object value = py::reinterpret_borrow<py::object>(item.second);
    set_parameter_wrapper(frost, key, value);
  }
}

py::tuple generate_mesh_wrapper(FrostInterface &frost) {
  std::vector<float> vertices;
  std::vector<int> faces;

  {
    py::gil_scoped_release release;
    frost.generate_mesh(vertices, faces);
  }

  const py::ssize_t vertexCount = (py::ssize_t)(vertices.size() / 3);
  const py::ssize_t faceCount = (py::ssize_t)(faces.size() / 3);

  py::array_t<float> vertexArray({vertexCount, (py::ssize_t)3});
  py::array_t<int> faceArray({faceCount, (py::ssize_t)3});

  if (!vertices.empty()) {
    std::memcpy(vertexArray.mutable_data(), vertices.data(),
                vertices.size() * sizeof(float));
  }

  if (!faces.empty()) {
    std::memcpy(faceArray.mutable_data(), faces.data(),
                faces.size() * sizeof(int));
  }

  return py::make_tuple(vertexArray, faceArray);
}

} // namespace

PYBIND11_MODULE(blender_frost_adapter, m) {
  m.doc() = "Frost Blender Adapter";

  // Initialize TBB to use ALL available hardware threads
  // Without this, TBB defaults to a conservative thread count
  unsigned int num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0)
    num_threads = 24; // Fallback for user's 12c/24t CPU

  g_tbb_thread_control = std::make_unique<tbb::global_control>(
      tbb::global_control::max_allowed_parallelism, num_threads);

  m.attr("tbb_thread_count") = num_threads; // Expose for debugging
  const std::string gpuBackendName = FrostInterface::get_gpu_backend_name();
  const auto vulkanBuffer = probe_frost_vulkan_storage_buffer();
  const auto vulkanRuntime = probe_frost_vulkan_runtime();
  const auto vulkanCompute = probe_frost_vulkan_compute();
  m.attr("HAS_CUDA_BACKEND") = gpuBackendName == "cuda";
  m.attr("HAS_GPU_BACKEND") = FrostInterface::has_gpu_backend();
  m.attr("GPU_BACKEND_NAME") = gpuBackendName;
  m.attr("HAS_VULKAN_RUNTIME") = vulkanRuntime.hasPhysicalDevice;
  m.attr("VULKAN_RUNTIME_STATUS") = vulkanRuntime.statusMessage;
  m.attr("HAS_VULKAN_COMPUTE") =
      vulkanCompute.deviceCreated && vulkanCompute.hasComputeQueue;
  m.attr("VULKAN_COMPUTE_STATUS") = vulkanCompute.statusMessage;
  m.attr("HAS_VULKAN_STORAGE_BUFFER") = vulkanBuffer.storageBufferReady;
  m.attr("VULKAN_STORAGE_BUFFER_STATUS") = vulkanBuffer.statusMessage;

  py::class_<FrostInterface>(m, "FrostInterface")
      .def(py::init<>())
      .def("set_particles", &set_particles_wrapper, "Set particle data",
           py::arg("positions"), py::arg("radii"),
           py::arg("velocities") = py::none())
      .def("set_parameter", &set_parameter_wrapper,
           "Set a single parameter")
      .def("set_parameters", &set_parameters_wrapper,
           "Set multiple parameters")
      .def("generate_mesh", &generate_mesh_wrapper,
           "Generate mesh from particles");
}
