"""
Stable native bridge for Frost.

This module loads the version-agnostic native DLL through ctypes so the addon
does not depend on Blender's embedded Python ABI.
"""

import ctypes
import os

import numpy as np


_addon_dir = os.path.dirname(__file__)
_dll_path = os.path.join(_addon_dir, "frost_native.dll")

if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(_addon_dir)

if not os.path.exists(_dll_path):
    raise ImportError(f"frost_native.dll not found in addon folder: {_dll_path}")

try:
    _lib = ctypes.CDLL(_dll_path)
except OSError as exc:
    raise ImportError(f"Failed to load frost_native.dll: {exc}") from exc


_lib.Frost_Create.restype = ctypes.c_void_p

_lib.Frost_Destroy.argtypes = [ctypes.c_void_p]
_lib.Frost_Destroy.restype = None

_lib.Frost_SetParticles.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
]
_lib.Frost_SetParticles.restype = ctypes.c_int

_lib.Frost_SetParameterBool.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
_lib.Frost_SetParameterBool.restype = ctypes.c_int

_lib.Frost_SetParameterInt.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
_lib.Frost_SetParameterInt.restype = ctypes.c_int

_lib.Frost_SetParameterFloat.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_float]
_lib.Frost_SetParameterFloat.restype = ctypes.c_int

_lib.Frost_GenerateMesh.argtypes = [ctypes.c_void_p]
_lib.Frost_GenerateMesh.restype = ctypes.c_int

_lib.Frost_GetVertexCount.argtypes = [ctypes.c_void_p]
_lib.Frost_GetVertexCount.restype = ctypes.c_int

_lib.Frost_GetFaceCount.argtypes = [ctypes.c_void_p]
_lib.Frost_GetFaceCount.restype = ctypes.c_int

_lib.Frost_CopyVertices.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
]
_lib.Frost_CopyVertices.restype = ctypes.c_int

_lib.Frost_CopyFaces.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_size_t,
]
_lib.Frost_CopyFaces.restype = ctypes.c_int

_lib.Frost_GetLastError.argtypes = [ctypes.c_void_p]
_lib.Frost_GetLastError.restype = ctypes.c_char_p

_lib.Frost_GetLastMeshingBackend.argtypes = [ctypes.c_void_p]
_lib.Frost_GetLastMeshingBackend.restype = ctypes.c_char_p

_lib.Frost_GetLastMeshingStatus.argtypes = [ctypes.c_void_p]
_lib.Frost_GetLastMeshingStatus.restype = ctypes.c_char_p

_lib.Frost_GetLastMeshingUsedFallback.argtypes = [ctypes.c_void_p]
_lib.Frost_GetLastMeshingUsedFallback.restype = ctypes.c_int

_lib.Frost_HasGpuBackend.argtypes = []
_lib.Frost_HasGpuBackend.restype = ctypes.c_int

_lib.Frost_HasCudaBackend.argtypes = []
_lib.Frost_HasCudaBackend.restype = ctypes.c_int

_lib.Frost_GetGpuBackendName.argtypes = []
_lib.Frost_GetGpuBackendName.restype = ctypes.c_char_p

_lib.Frost_HasVulkanRuntime.argtypes = []
_lib.Frost_HasVulkanRuntime.restype = ctypes.c_int

_lib.Frost_GetVulkanRuntimeStatus.argtypes = []
_lib.Frost_GetVulkanRuntimeStatus.restype = ctypes.c_char_p

_lib.Frost_HasVulkanCompute.argtypes = []
_lib.Frost_HasVulkanCompute.restype = ctypes.c_int

_lib.Frost_GetVulkanComputeStatus.argtypes = []
_lib.Frost_GetVulkanComputeStatus.restype = ctypes.c_char_p

_lib.Frost_HasVulkanStorageBuffer.argtypes = []
_lib.Frost_HasVulkanStorageBuffer.restype = ctypes.c_int

_lib.Frost_GetVulkanStorageBufferStatus.argtypes = []
_lib.Frost_GetVulkanStorageBufferStatus.restype = ctypes.c_char_p

GPU_BACKEND_NAME = (_lib.Frost_GetGpuBackendName() or b"none").decode("utf-8", errors="replace")
HAS_GPU_BACKEND = bool(_lib.Frost_HasGpuBackend())
HAS_CUDA_BACKEND = bool(_lib.Frost_HasCudaBackend())
HAS_VULKAN_RUNTIME = bool(_lib.Frost_HasVulkanRuntime())
VULKAN_RUNTIME_STATUS = (_lib.Frost_GetVulkanRuntimeStatus() or b"").decode(
    "utf-8", errors="replace"
)
HAS_VULKAN_COMPUTE = bool(_lib.Frost_HasVulkanCompute())
VULKAN_COMPUTE_STATUS = (_lib.Frost_GetVulkanComputeStatus() or b"").decode(
    "utf-8", errors="replace"
)
HAS_VULKAN_STORAGE_BUFFER = bool(_lib.Frost_HasVulkanStorageBuffer())
VULKAN_STORAGE_BUFFER_STATUS = (
    (_lib.Frost_GetVulkanStorageBufferStatus() or b"").decode(
        "utf-8", errors="replace"
    )
)


def _as_float_pointer(array):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _as_int_pointer(array):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))


class FrostInterface:
    """Python wrapper matching the old pybind11-facing API."""

    def __init__(self):
        handle = _lib.Frost_Create()
        if not handle:
            raise RuntimeError("Failed to create Frost native context")
        self._handle = handle

    def __del__(self):
        handle = getattr(self, "_handle", None)
        if handle:
            try:
                _lib.Frost_Destroy(handle)
            except Exception:
                pass
            self._handle = None

    def _get_error(self):
        message = _lib.Frost_GetLastError(self._handle)
        if not message:
            return "Unknown Frost native error"
        return message.decode("utf-8", errors="replace")

    def get_last_meshing_info(self):
        backend = _lib.Frost_GetLastMeshingBackend(self._handle)
        status = _lib.Frost_GetLastMeshingStatus(self._handle)
        return {
            "backend": (backend or b"").decode("utf-8", errors="replace"),
            "status": (status or b"").decode("utf-8", errors="replace"),
            "used_fallback": bool(_lib.Frost_GetLastMeshingUsedFallback(self._handle)),
        }

    def _require_ok(self, result, action):
        if result:
            return
        raise RuntimeError(f"{action} failed: {self._get_error()}")

    def set_particles(self, positions, radii, velocities=None):
        positions = np.ascontiguousarray(positions, dtype=np.float32)
        radii = np.ascontiguousarray(radii, dtype=np.float32)

        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape [N, 3]")
        if radii.ndim != 1 or radii.shape[0] != positions.shape[0]:
            raise ValueError("radii must have shape [N]")

        velocity_ptr = None
        if velocities is not None:
            velocities = np.ascontiguousarray(velocities, dtype=np.float32)
            if velocities.ndim != 2 or velocities.shape != positions.shape:
                raise ValueError("velocities must have shape [N, 3]")
            velocity_ptr = _as_float_pointer(velocities)

        self._require_ok(
            _lib.Frost_SetParticles(
                self._handle,
                _as_float_pointer(positions),
                positions.shape[0],
                _as_float_pointer(radii),
                velocity_ptr,
            ),
            "set_particles",
        )

    def set_parameter(self, name, value):
        if isinstance(value, (bool, np.bool_)):
            result = _lib.Frost_SetParameterBool(
                self._handle, name.encode("utf-8"), int(bool(value))
            )
        elif isinstance(value, (int, np.integer)):
            result = _lib.Frost_SetParameterInt(
                self._handle, name.encode("utf-8"), int(value)
            )
        elif isinstance(value, (float, np.floating)):
            result = _lib.Frost_SetParameterFloat(
                self._handle, name.encode("utf-8"), float(value)
            )
        else:
            raise TypeError(f"Unsupported Frost parameter type for {name}: {type(value)!r}")

        self._require_ok(result, f"set_parameter({name})")

    def set_parameters(self, params):
        for name, value in params.items():
            self.set_parameter(name, value)

    def generate_mesh(self):
        self._require_ok(_lib.Frost_GenerateMesh(self._handle), "generate_mesh")

        vertex_count = _lib.Frost_GetVertexCount(self._handle)
        face_count = _lib.Frost_GetFaceCount(self._handle)

        vertices = np.empty((vertex_count, 3), dtype=np.float32)
        faces = np.empty((face_count, 3), dtype=np.int32)

        self._require_ok(
            _lib.Frost_CopyVertices(self._handle, _as_float_pointer(vertices), vertices.size),
            "copy_vertices",
        )
        self._require_ok(
            _lib.Frost_CopyFaces(self._handle, _as_int_pointer(faces), faces.size),
            "copy_faces",
        )

        return vertices, faces
