"""
Frost Particle Meshing Operator

Main operator for generating meshes from particles using the C++ Frost Core.
"""

import bpy
import time
import numpy as np
import os
import sys
import traceback

from .particle_extractor import extract_particles

# Optional: Import C++ adapter with DLL path handling
adapter_load_error = None
blender_frost_adapter = None

try:
    # Add addon directory to DLL search path for dependencies (TBB, Boost, etc.)
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(os.path.dirname(__file__))

    try:
        from . import native_bridge as blender_frost_adapter
    except ImportError as native_exc:
        from . import blender_frost_adapter
        adapter_load_error = f"Native bridge unavailable: {native_exc}"
except ImportError as e:
    blender_frost_adapter = None
    adapter_load_error = f"{str(e)} | Py: {sys.version.split()[0]}"
except Exception as e:
    blender_frost_adapter = None
    adapter_load_error = f"Unexpected error: {str(e)} | Py: {sys.version.split()[0]}"

import gc

_updates_in_progress = set()


def has_native_backend():
    return blender_frost_adapter is not None


def get_gpu_backend_name():
    if blender_frost_adapter is None:
        return "none"
    return str(getattr(blender_frost_adapter, "GPU_BACKEND_NAME", "cuda" if getattr(blender_frost_adapter, "HAS_CUDA_BACKEND", False) else "none"))


def has_gpu_backend():
    if blender_frost_adapter is None:
        return False
    if hasattr(blender_frost_adapter, "HAS_GPU_BACKEND"):
        return bool(getattr(blender_frost_adapter, "HAS_GPU_BACKEND"))
    return bool(getattr(blender_frost_adapter, "HAS_CUDA_BACKEND", False))


def has_vulkan_runtime():
    if blender_frost_adapter is None:
        return False
    return bool(getattr(blender_frost_adapter, "HAS_VULKAN_RUNTIME", False))


def get_vulkan_runtime_status():
    if blender_frost_adapter is None:
        return ""
    return str(getattr(blender_frost_adapter, "VULKAN_RUNTIME_STATUS", ""))


def has_vulkan_compute():
    if blender_frost_adapter is None:
        return False
    return bool(getattr(blender_frost_adapter, "HAS_VULKAN_COMPUTE", False))


def get_vulkan_compute_status():
    if blender_frost_adapter is None:
        return ""
    return str(getattr(blender_frost_adapter, "VULKAN_COMPUTE_STATUS", ""))


def has_vulkan_storage_buffer():
    if blender_frost_adapter is None:
        return False
    return bool(getattr(blender_frost_adapter, "HAS_VULKAN_STORAGE_BUFFER", False))


def get_vulkan_storage_buffer_status():
    if blender_frost_adapter is None:
        return ""
    return str(getattr(blender_frost_adapter, "VULKAN_STORAGE_BUFFER_STATUS", ""))

# Performance Optimization: Cache FrostInterface per object to avoid recreation overhead
_frost_cache = {}

def get_cached_frost(obj_id):
    """Get or create a cached FrostInterface for an object."""
    global _frost_cache
    if obj_id not in _frost_cache:
        _frost_cache[obj_id] = blender_frost_adapter.FrostInterface()
    return _frost_cache[obj_id]

def clear_frost_cache(obj_id=None):
    """Clear cached FrostInterface instances."""
    global _frost_cache
    if obj_id is not None and obj_id in _frost_cache:
        del _frost_cache[obj_id]
    else:
        _frost_cache.clear()


def _trim_status_message(message, max_chars=160):
    message = (message or "").strip()
    if len(message) <= max_chars:
        return message
    return message[: max_chars - 3].rstrip() + "..."


def record_meshing_status(frost_props, gpu_requested, meshing_info):
    backend = str(meshing_info.get("backend", "") or "")
    status = _trim_status_message(meshing_info.get("status", "") or "")
    used_fallback = bool(meshing_info.get("used_fallback", False))

    if not backend:
        backend = "cpu-fallback" if used_fallback else "cpu"
    if not status:
        if used_fallback:
            status = "GPU meshing was requested, but Frost used the CPU fallback path."
        elif backend.startswith("vulkan") or backend.startswith("cuda"):
            status = "GPU meshing path used successfully."
        else:
            status = "CPU meshing path used."

    frost_props.last_meshing_backend = backend
    frost_props.last_meshing_status = status
    frost_props.last_meshing_used_fallback = used_fallback
    frost_props.last_meshing_had_gpu_request = bool(gpu_requested)
    frost_props.last_meshing_timestamp = time.time()

def unload_adapter():
    """Explicitly release the C++ adapter to prevent shutdown crashes."""
    global blender_frost_adapter, _frost_cache
    _frost_cache.clear()
    blender_frost_adapter = None
    gc.collect()


def validate_mesh_arrays(vertices, faces):
    """Validate mesh buffers before they are sent to Blender."""
    if vertices is None or faces is None:
        return False, "missing mesh buffers"

    vertices = np.asarray(vertices)
    faces = np.asarray(faces)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        return False, f"invalid vertex array shape {vertices.shape}"
    if faces.ndim != 2 or faces.shape[1] != 3:
        return False, f"invalid face array shape {faces.shape}"

    if len(vertices) == 0 and len(faces) == 0:
        return True, ""
    if len(vertices) == 0 or len(faces) == 0:
        return False, "mesh buffers contain vertices or faces, but not both"

    if not np.isfinite(vertices).all():
        return False, "vertex array contains non-finite coordinates"
    if np.issubdtype(faces.dtype, np.integer) is False:
        return False, f"invalid face dtype {faces.dtype}"
    if (faces < 0).any():
        return False, "face array contains negative vertex indices"
    max_face_index = int(np.max(faces)) if faces.size else -1
    if max_face_index >= len(vertices):
        return False, "face array contains out-of-range vertex indices"
    if np.any(faces[:, 0] == faces[:, 1]) or np.any(faces[:, 1] == faces[:, 2]) or np.any(faces[:, 0] == faces[:, 2]):
        return False, "face array contains degenerate triangles"

    return True, ""


def get_particle_data(obj, context, frost_props):
    """Gather particle data from all enabled sources."""
    
    all_positions = []
    all_radii = []
    all_velocities = []
    
    total_particles = 0
    
    # Iterate over sources
    # 1. Main Source
    if frost_props.source_object:
         try:
            p, r, v = extract_particles(
                frost_props.source_object,
                source_type=frost_props.source_type, # Main Type
                particle_system_index=frost_props.particle_system_index, # Main Index
                default_radius=frost_props.default_radius
            )
            if len(p) > 0:
                all_positions.append(p)
                all_radii.append(r)
                if v is not None and len(v) > 0: all_velocities.append(v)
                else: all_velocities.append(np.zeros_like(p))
                total_particles += len(p)
         except Exception as e:
            print(f"Frost Warning: Failed to extract from {frost_props.source_object.name}: {e}")

    # 2. Additional Sources
    sources = frost_props.sources
    
    for item in sources:
        if not item.enabled or not item.object:
            continue
            
        source_obj = item.object
        
        try:
            p, r, v = extract_particles(
                source_obj,
                source_type=item.source_type,
                particle_system_index=item.system_index,
                default_radius=frost_props.default_radius
            )
            
            if len(p) > 0:
                all_positions.append(p)
                all_radii.append(r)
                if v is not None and len(v) > 0:
                    all_velocities.append(v)
                else:
                    # Pad zero velocity if mixed sources have/have not velocity
                    all_velocities.append(np.zeros_like(p))
                    
                total_particles += len(p)
                
        except Exception as e:
            print(f"Frost Warning: Failed to extract from {source_obj.name}: {e}")
            
    if total_particles == 0:
        return np.array([]), np.array([]), None
        
    # Concatenate
    positions = np.concatenate(all_positions)
    radii = np.concatenate(all_radii)
    velocities = np.concatenate(all_velocities)
    
    return positions, radii, velocities

def update_frost_mesh(obj, context):
    """
    Core function to generate Frost mesh data and update the given object.
    Updates obj.data in-place.
    """
    if blender_frost_adapter is None:
        print("Frost C++ adapter not loaded.")
        return

    obj_id = id(obj)
    if obj_id in _updates_in_progress:
        return

    # Smart Optimization: Skip updates if hidden in viewport
    if obj.hide_viewport:
        return

    frost_props = obj.frost_properties
    
    _updates_in_progress.add(obj_id)
    try:
        start_time = time.time()
        
        # Extract particles
        try:
            positions, radii, velocities = get_particle_data(obj, context, frost_props)
        except Exception as e:
            print(f"Frost Warning: {e}")
            return

        if len(positions) == 0:
            return # Nothing to mesh
        
        # Safety Check: Particle Count Limit
        # Very large particle counts combined with fine voxel resolution can crash TBB allocator
        MAX_SAFE_PARTICLES = 500000  # Conservative limit to prevent memory crashes
        particle_count = len(positions)
        
        if particle_count > MAX_SAFE_PARTICLES:
            print(f"Frost Warning: Skipping meshing - {particle_count} particles exceeds safe limit of {MAX_SAFE_PARTICLES}.")
            print(f"  -> Reduce particle count or increase Voxel Length to prevent crashes.")
            return
        
        # Safety Check: Estimate Voxel Grid Size
        # Grid size ~ (bounding_box / voxel_size)^3
        # If this is too large, skip to avoid memory exhaustion
        if particle_count > 10000:  # Only check for larger counts
            min_bound = np.min(positions, axis=0)
            max_bound = np.max(positions, axis=0)
            bbox_size = max_bound - min_bound
            
            voxel_size = frost_props.meshing_voxel_length
            if voxel_size < 0.01:  # Very fine resolution
                estimated_voxels = np.prod(bbox_size / max(voxel_size, 0.001))
                if estimated_voxels > 500_000_000:  # 500M voxels is dangerous
                    print(f"Frost Warning: Skipping meshing - Estimated voxel count ({estimated_voxels:.0f}) too large.")
                    print(f"  -> Increase Voxel Length to prevent crashes.")
                    return

        # Initialize Frost - use cached instance for performance
        try:
            gpu_backend_available = has_gpu_backend()
            gpu_backend_name = get_gpu_backend_name()
            if frost_props.use_gpu and not gpu_backend_available and frost_props.show_debug_log:
                print(f"Frost: no GPU backend is available in the current native build ({gpu_backend_name}), falling back to CPU.")
            
            # Transform World Space Particles -> Local Space of Frost Object
            # Extracted particles are in World Space.
            # But we are writing to obj.data (Mesh). 
            # Blender applies obj.matrix_world to the mesh.
            # So: FinalPos = ObjMatrix * VertexPos
            # We want FinalPos == WorldSpaceParticle
            # Therefore: VertexPos = Inverse(ObjMatrix) * WorldSpaceParticle
            
            matrix_inv = np.array(obj.matrix_world.inverted()) # 4x4
            
            # Homogeneous transform
            ones = np.ones((len(positions), 1), dtype=np.float32)
            pos_homo = np.hstack((positions, ones)) # Nx4
            
            # Apply Inverse Matrix
            positions_local_homo = pos_homo @ matrix_inv.T
            positions = positions_local_homo[:, :3].astype(np.float32)

            # Set particles
            positions = np.ascontiguousarray(positions, dtype=np.float32)
            radii = np.ascontiguousarray(radii, dtype=np.float32)
            if velocities is not None:
                velocities = np.ascontiguousarray(velocities, dtype=np.float32)
            
            # Apply Union of Spheres radius scaling
            if frost_props.meshing_method == '0': # Union of Spheres
                radii = radii * frost_props.union_of_spheres_radius_scale
            
            if len(radii) > 0:
                epsilon_radius = 1e-4
                radii = np.maximum(radii, epsilon_radius)

            if len(positions) > 0:
                min_bound = np.min(positions, axis=0)
                max_bound = np.max(positions, axis=0)
                size = max_bound - min_bound
                
                epsilon_bounds = 1e-4
                if np.any(size < epsilon_bounds):
                    if size[0] < epsilon_bounds: positions[0, 0] += epsilon_bounds
                    if size[1] < epsilon_bounds: positions[0, 1] += epsilon_bounds
                    if size[2] < epsilon_bounds: positions[0, 2] += epsilon_bounds

            params = {
                "meshing_method": int(frost_props.meshing_method),
                "meshing_resolution_mode": int(frost_props.meshing_resolution_mode),
                "resolution": frost_props.meshing_resolution,
                "voxel_size": frost_props.meshing_voxel_length,
                
                # Adaptive Resolution
                "vert_refinement": frost_props.vert_refinement,
                
                # Post Processing
                "relax_iterations": frost_props.relax_iterations,
                "relax_strength": frost_props.relax_strength,
                "push_distance": frost_props.push_distance,

                # Union of Spheres
                "union_of_spheres_radius_scale": frost_props.union_of_spheres_radius_scale,

                # Metaball
                "metaball_radius_scale": frost_props.metaball_radius_scale,
                "metaball_isosurface_level": frost_props.metaball_isosurface_level,
                
                # Plain Marching Cubes
                "plain_marching_cubes_radius_scale": frost_props.plain_marching_cubes_radius_scale,
                "plain_marching_cubes_isovalue": frost_props.plain_marching_cubes_isovalue,
                
                # Zhu-Bridson
                "zhu_bridson_blend_radius_scale": frost_props.zhu_bridson_blend_radius_scale,
                "zhu_bridson_low_density_trimming": False,
                "zhu_bridson_trimming_threshold": 0.0,
                "zhu_bridson_trimming_strength": 0.0,
                
                # Anisotropic
                "anisotropic_radius_scale": frost_props.anisotropic_radius_scale,
                "anisotropic_isosurface_level": frost_props.anisotropic_isosurface_level,
                "anisotropic_max_anisotropy": frost_props.anisotropic_max_anisotropy,
                "anisotropic_min_neighbor_count": frost_props.anisotropic_min_neighbor_count,
                "anisotropic_position_smoothing_weight": frost_props.anisotropic_position_smoothing_weight,
                
                # GPU settings
                "use_gpu": bool(frost_props.use_gpu and gpu_backend_available),
                "gpu_search_radius_scale": frost_props.gpu_search_radius_scale,
                "gpu_voxel_size": frost_props.gpu_voxel_size,
                "gpu_block_size": 256,
                "gpu_surface_refinement": 0,
                
                # Debug
                "show_debug_log": frost_props.show_debug_log,
            }

            def run_mesher(use_gpu_flag):
                frost = get_cached_frost(obj_id)
                frost.set_particles(positions, radii, velocities)
                params_to_apply = dict(params)
                params_to_apply["use_gpu"] = bool(use_gpu_flag and gpu_backend_available)
                frost.set_parameters(params_to_apply)
                vertices_local, faces_local = frost.generate_mesh()
                if hasattr(frost, "get_last_meshing_info"):
                    meshing_info_local = frost.get_last_meshing_info()
                else:
                    meshing_info_local = {
                        "backend": "cpu",
                        "status": "Meshing completed.",
                        "used_fallback": False,
                    }
                return vertices_local, faces_local, meshing_info_local

            vertices, faces, meshing_info = run_mesher(frost_props.use_gpu)
            is_valid, validation_error = validate_mesh_arrays(vertices, faces)

            if not is_valid and frost_props.use_gpu:
                print(f"Frost Warning: GPU mesh validation failed for {obj.name}: {validation_error}")
                clear_frost_cache(obj_id)
                vertices, faces, meshing_info = run_mesher(False)
                meshing_info = {
                    "backend": "cpu-fallback",
                    "status": f"GPU mesh rejected during validation: {validation_error}",
                    "used_fallback": True,
                }
                is_valid, validation_error = validate_mesh_arrays(vertices, faces)

            if not is_valid:
                raise RuntimeError(f"Unsafe mesh buffers rejected: {validation_error}")

            record_meshing_status(frost_props, frost_props.use_gpu, meshing_info)
            
        except Exception as e:
            traceback.print_exc()
            print(f"Meshing failed: {str(e)}")
            frost_props.last_meshing_backend = "error"
            frost_props.last_meshing_status = _trim_status_message(str(e))
            frost_props.last_meshing_used_fallback = bool(frost_props.use_gpu)
            frost_props.last_meshing_had_gpu_request = bool(frost_props.use_gpu)
            frost_props.last_meshing_timestamp = time.time()
            clear_frost_cache(obj_id)
            return
        
        if len(vertices) == 0:
            return
        
        # Update Blender mesh data - OPTIMIZED
        try:
            mesh_data = obj.data
            mesh_data.clear_geometry() # Clear existing data
            
            num_verts = len(vertices)
            num_faces = len(faces)
            
            mesh_data.vertices.add(num_verts)
            mesh_data.loops.add(num_faces * 3)
            mesh_data.polygons.add(num_faces)
            
            verts_flat = np.ascontiguousarray(vertices.ravel(), dtype=np.float32)
            faces_flat = np.ascontiguousarray(faces.ravel(), dtype=np.int32)
            
            mesh_data.vertices.foreach_set("co", verts_flat)
            mesh_data.polygons.foreach_set("loop_start", np.arange(0, num_faces * 3, 3, dtype=np.int32))
            mesh_data.polygons.foreach_set("loop_total", np.full(num_faces, 3, dtype=np.int32))
            mesh_data.loops.foreach_set("vertex_index", faces_flat)
            
            mesh_data.update()
            
            if frost_props.use_smooth_shading:
                mesh_data.polygons.foreach_set("use_smooth", np.ones(num_faces, dtype=bool))
            else:
                mesh_data.polygons.foreach_set("use_smooth", np.zeros(num_faces, dtype=bool))
            
        except Exception as e:
            print(f"Failed to update Blender mesh data: {str(e)}")
        
        elapsed = time.time() - start_time
        # print(f"Frost Update: {len(vertices)} verts in {elapsed:.3f}s")
    finally:
        _updates_in_progress.discard(obj_id)


class FROST_OT_generate_mesh(bpy.types.Operator):
    """Generate independent Frost Mesh from active object"""
    bl_idname = "frost.generate_mesh"
    bl_label = "Generate Frost Mesh"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        return context.active_object is not None
    
    def execute(self, context):
        if blender_frost_adapter is None:
            self.report({'ERROR'}, "Frost C++ adapter not loaded.")
            return {'CANCELLED'}

        src_obj = context.active_object
        
        # Create new Mesh Object
        mesh_name = f"{src_obj.name}_Frost"
        mesh_data = bpy.data.meshes.new(mesh_name)
        new_obj = bpy.data.objects.new(mesh_name, mesh_data)
        context.collection.objects.link(new_obj)
        
        # Link source
        new_obj.frost_properties.source_object = src_obj
        
        # Copy properties from source to new object
        src_props = src_obj.frost_properties
        new_props = new_obj.frost_properties
        
        # Core Settings
        new_props.use_gpu = src_props.use_gpu
        new_props.default_radius = src_props.default_radius
        
        # GPU Settings
        new_props.gpu_meshing_method = src_props.gpu_meshing_method
        new_props.gpu_search_radius_scale = src_props.gpu_search_radius_scale
        new_props.gpu_voxel_size = src_props.gpu_voxel_size
        new_props.gpu_block_size = src_props.gpu_block_size
        
        # CPU Settings
        new_props.meshing_method = src_props.meshing_method
        new_props.meshing_resolution_mode = src_props.meshing_resolution_mode
        new_props.meshing_resolution = src_props.meshing_resolution
        new_props.meshing_voxel_length = src_props.meshing_voxel_length
        new_props.vert_refinement = src_props.vert_refinement
        
        # Method Specific Settings
        new_props.union_of_spheres_radius_scale = src_props.union_of_spheres_radius_scale
        
        new_props.metaball_radius_scale = src_props.metaball_radius_scale
        new_props.metaball_isosurface_level = src_props.metaball_isosurface_level
        
        new_props.plain_marching_cubes_radius_scale = src_props.plain_marching_cubes_radius_scale
        new_props.plain_marching_cubes_isovalue = src_props.plain_marching_cubes_isovalue
        
        new_props.zhu_bridson_blend_radius_scale = src_props.zhu_bridson_blend_radius_scale
        new_props.zhu_bridson_enable_low_density_trimming = False
        new_props.zhu_bridson_low_density_trimming_threshold = 0.0
        new_props.zhu_bridson_low_density_trimming_strength = 0.0
        
        new_props.anisotropic_radius_scale = src_props.anisotropic_radius_scale
        new_props.anisotropic_isosurface_level = src_props.anisotropic_isosurface_level
        new_props.anisotropic_max_anisotropy = src_props.anisotropic_max_anisotropy
        new_props.anisotropic_min_neighbor_count = src_props.anisotropic_min_neighbor_count
        new_props.anisotropic_position_smoothing_weight = src_props.anisotropic_position_smoothing_weight
        
        # Post Processing
        new_props.push_distance = src_props.push_distance
        new_props.relax_iterations = src_props.relax_iterations
        new_props.relax_strength = src_props.relax_strength
        new_props.use_smooth_shading = src_props.use_smooth_shading
        
        # Perform initial update
        context.view_layer.objects.active = new_obj
        update_frost_mesh(new_obj, context)
        
        return {'FINISHED'}

class FROST_OT_bake_to_alembic(bpy.types.Operator):
    """Bake Frost animation to Alembic (.abc) sequence"""
    bl_idname = "frost.bake_alembic"
    bl_label = "Bake to Alembic"
    bl_options = {'REGISTER'}

    # File browser properties
    filepath: bpy.props.StringProperty(
        name="File Path",
        description="Output Alembic file path (frames will be saved as name_0001.abc, etc.)",
        subtype='FILE_PATH',
        default="frost_export.abc"
    )

    filter_glob: bpy.props.StringProperty(
        default="*.abc",
        options={'HIDDEN'}
    )

    def invoke(self, context, event):
        obj = context.active_object
        if obj:
            self.filepath = f"{obj.name}_frost.abc"
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        obj = context.active_object

        # Validation
        if not obj:
            self.report({'ERROR'}, "No active object.")
            return {'CANCELLED'}

        has_source = obj.frost_properties.source_object is not None or len(obj.frost_properties.sources) > 0
        if not has_source:
            self.report({'ERROR'}, "Active object is not a valid Frost mesh (no source).")
            return {'CANCELLED'}

        # Ensure object mode
        if bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode='OBJECT')

        # Select ONLY the Frost object
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        context.view_layer.objects.active = obj

        scene = context.scene
        frame_start = scene.frame_start
        frame_end = scene.frame_end
        original_frame = scene.frame_current

        # Build output directory from filepath
        base_path = os.path.splitext(self.filepath)[0]
        base_dir = os.path.dirname(self.filepath)
        base_name = os.path.splitext(os.path.basename(self.filepath))[0]

        # Create output directory for the sequence
        seq_dir = os.path.join(base_dir, base_name + "_seq")
        os.makedirs(seq_dir, exist_ok=True)

        total_frames = frame_end - frame_start + 1
        exported = 0

        print(f"[Frost Bake] Exporting frames {frame_start}-{frame_end} to {seq_dir}")

        try:
            for frame in range(frame_start, frame_end + 1):
                # 1. Set frame — triggers depsgraph update
                scene.frame_set(frame)

                # 2. Force Frost mesh regeneration
                update_frost_mesh(obj, context)

                # 3. Export this single frame to its own .abc file
                frame_path = os.path.join(seq_dir, f"{base_name}_{frame:04d}.abc")

                bpy.ops.wm.alembic_export(
                    filepath=frame_path,
                    start=frame,
                    end=frame,
                    selected=True,
                    flatten=False,
                    export_hair=False,
                    export_particles=False,
                    as_background_job=False,
                )

                exported += 1
                # Progress feedback
                if exported % 10 == 0 or frame == frame_end:
                    print(f"[Frost Bake] {exported}/{total_frames} frames exported")

        except Exception as e:
            self.report({'ERROR'}, f"Bake failed at frame {frame}: {e}")
            traceback.print_exc()
        finally:
            # Restore original frame
            scene.frame_set(original_frame)

        self.report({'INFO'}, f"Frost Alembic sequence: {exported} frames → {seq_dir}")
        return {'FINISHED'}



def register():
    bpy.utils.register_class(FROST_OT_generate_mesh)
    bpy.utils.register_class(FROST_OT_bake_to_alembic)

def unregister():
    bpy.utils.unregister_class(FROST_OT_bake_to_alembic)
    bpy.utils.unregister_class(FROST_OT_generate_mesh)
