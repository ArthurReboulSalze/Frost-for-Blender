"""
Frost UI Panel and Properties

User interface for Frost particle meshing in the 3D Viewport.
"""

import bpy
from bpy.props import EnumProperty, FloatProperty, IntProperty, BoolProperty, PointerProperty, StringProperty
from bpy.types import PropertyGroup, Panel
import time

# Simple debounce mechanism to prevent cascade updates
_last_update_time = 0
_update_pending = False


def format_meshing_backend_label(backend_name):
    backend_name = (backend_name or "").strip()
    if not backend_name:
        return "Unknown"

    parts = backend_name.replace("_", " ").replace("-", " ").split()
    formatted_parts = []
    for part in parts:
        lower = part.lower()
        if lower == "cpu":
            formatted_parts.append("CPU")
        elif lower == "gpu":
            formatted_parts.append("GPU")
        elif lower == "raw":
            formatted_parts.append("Raw")
        elif lower == "fallback":
            formatted_parts.append("Fallback")
        elif lower == "vulkan":
            formatted_parts.append("Vulkan")
        elif lower == "cuda":
            formatted_parts.append("CUDA")
        else:
            formatted_parts.append(part.capitalize())
    return " ".join(formatted_parts)

def update_mesh_callback(self, context):
    """Trigger mesh update if auto_update is enabled"""
    global _last_update_time, _update_pending
    
    if not self.auto_update:
        return
    
    # Debounce: skip if last update was less than 50ms ago
    current_time = time.time()
    if current_time - _last_update_time < 0.05:
        return
    
    _last_update_time = current_time
        
    # Import locally to avoid circular import during registration
    from . import operator
    
    # self is FrostProperties or FrostSourceItem
    # For FrostSourceItem, id_data is still the Object
    obj = self.id_data
    if obj and obj.type == 'MESH':
        operator.update_frost_mesh(obj, context)


class FrostSourceItem(PropertyGroup):
    """A single source contributing particles to the Frost mesh."""
    
    # We can't easily use "update" callback on PointerProperty to trigger main update
    # because 'self' is the Item, not the main FrostProperties.
    # We'll rely on the main "Update" button or Auto-Update looping/checking?
    # Actually, we can look up the ID data.
    
    object: PointerProperty(
        name="Object",
        type=bpy.types.Object,
        description="Source Object",
        update=update_mesh_callback
    )
    
    system_index: IntProperty(
        name="System Index",
        default=0,
        min=0,
        update=update_mesh_callback
    )
    
    source_type: EnumProperty(
        name="Type",
        items=[
            ('AUTO', "Auto", "Automatically detect"),
            ('PARTICLE_SYSTEM', "Particle System", ""),
            ('POINT_CLOUD', "Point Cloud", ""),
            ('MESH', "Mesh Vertices", ""),
        ],
        default='AUTO',
        update=update_mesh_callback
    )
    
    enabled: BoolProperty(
        name="Enabled",
        default=True,
        update=update_mesh_callback
    )


class FROST_UL_source_list(bpy.types.UIList):
    """UI List for managing Frost Sources."""
    
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        # item is FrostSourceItem
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            row.prop(item, "enabled", text="")
            row.prop(item, "object", text="", icon='PARTICLES')
            
            if item.object:
                # Contextual info
                if item.source_type == 'PARTICLE_SYSTEM':
                     row.prop(item, "system_index", text="Sys")
                elif item.source_type == 'AUTO':
                     row.label(text="Auto")
            
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text="", icon='PARTICLES')


class FROST_OT_add_source(bpy.types.Operator):
    """Add a new source to the Frost mesh"""
    bl_idname = "frost.add_source"
    bl_label = "Add Source"
    
    def execute(self, context):
        from . import operator # Fix NameError
        obj = context.active_object
        if not obj: return {'CANCELLED'}
        
        props = obj.frost_properties
        item = props.sources.add()
        props.active_source_index = len(props.sources) - 1
        
        return {'FINISHED'}

class FROST_OT_remove_source(bpy.types.Operator):
    """Remove the selected source from the Frost mesh"""
    bl_idname = "frost.remove_source"
    bl_label = "Remove Source"
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.frost_properties.sources
        
    def execute(self, context):
        from . import operator # Fix NameError
        obj = context.active_object
        props = obj.frost_properties
        idx = props.active_source_index
        
        props.sources.remove(idx)
        
        if props.active_source_index >= len(props.sources):
            props.active_source_index = max(0, len(props.sources) - 1)
            
        # Trigger update
        operator.update_frost_mesh(obj, context)
            
        return {'FINISHED'}


class FrostProperties(PropertyGroup):
    """Properties for Frost meshing."""
    
    # Logic Settings
    auto_update: BoolProperty(
        name="Auto Update",
        description="Update mesh automatically when parameters change",
        default=True,
        update=update_mesh_callback
    )

    show_debug_log: BoolProperty(
        name="Show Debug Log",
        description="Print verbose debug info to System Console",
        default=False,
        update=update_mesh_callback
    )

    last_meshing_backend: StringProperty(
        name="Last Meshing Backend",
        default="",
        options={'HIDDEN'}
    )

    last_meshing_status: StringProperty(
        name="Last Meshing Status",
        default="",
        options={'HIDDEN'}
    )

    last_meshing_used_fallback: BoolProperty(
        name="Last Meshing Used Fallback",
        default=False,
        options={'HIDDEN'}
    )

    last_meshing_had_gpu_request: BoolProperty(
        name="Last Meshing Had GPU Request",
        default=False,
        options={'HIDDEN'}
    )

    last_meshing_timestamp: FloatProperty(
        name="Last Meshing Timestamp",
        default=0.0,
        options={'HIDDEN'}
    )
    
    # Primary Source
    source_object: PointerProperty(
        name="Source Object",
        type=bpy.types.Object,
        description="Main object to emit particles from",
        update=update_mesh_callback
    )
    
    # Multi-Source Support (Additional)
    sources: bpy.props.CollectionProperty(type=FrostSourceItem)
    active_source_index: IntProperty()

    
    source_type: EnumProperty(
        name="Source",
        description="Particle source type",
        items=[
            ('AUTO', "Auto", "Automatically detect source type"),
            ('PARTICLE_SYSTEM', "Particle System", "Use particle system"),
            ('POINT_CLOUD', "Point Cloud", "Use point cloud from Geometry Nodes"),
            ('MESH', "Mesh Vertices", "Use mesh vertices as particles"),
        ],
        default='AUTO',
        update=update_mesh_callback
    )
    
    particle_system_index: IntProperty(
        name="Particle System",
        description="Index of particle system to use",
        default=0,
        min=0,
        update=update_mesh_callback
    )
    
    default_radius: FloatProperty(
        name="Default Radius",
        description="Default particle radius (for mesh vertices)",
        default=0.1,
        min=0.001,
        max=10.0,
        update=update_mesh_callback
    )
    
    # Core Mesh Settings
    use_gpu: BoolProperty(
        name="GPU Acceleration",
        description="Enable the available GPU backend when the current native build supports one",
        default=False,
        update=update_mesh_callback
    )
    
    # GPU-Specific Settings
    gpu_search_radius_scale: FloatProperty(
        name="Search Radius Scale",
        description="Search Radius = Particle Radius * Scale",
        default=1.5,
        min=0.1,
        max=4.0,
        update=update_mesh_callback
    )
    
    gpu_voxel_size: FloatProperty(
        name="GPU Voxel Size",
        description="GPU mesh resolution (smaller = more detailed)",
        default=0.1,
        min=0.002,
        max=1.0,
        update=update_mesh_callback
    )
    
    gpu_block_size: IntProperty(
        name="Block Size",
        description="Internal GPU workgroup size",
        default=256,
        min=64,
        max=1024,
        update=update_mesh_callback
    )
    
    gpu_meshing_method: EnumProperty(
        name="GPU Method",
        description="Legacy dedicated GPU meshing algorithm",
        items=[
            ('ZHU_BRIDSON_GPU', "Zhu-Bridson GPU", "Fluid meshing with the active GPU backend"),
            # ('METABALL_GPU', "Metaball GPU", "Smooth blobby surface with GPU acceleration"), # Not implemented yet
            # ('ANISOTROPIC_GPU', "Anisotropic GPU", "Stretched metaballs with GPU neighbor search"), # Not implemented yet
        ],
        default='ZHU_BRIDSON_GPU',
        update=update_mesh_callback
    )

    meshing_method: EnumProperty(
        name="Method",
        description="Meshing algorithm",
        items=[
            ('0', "Union of Spheres", "Simple sphere union"),
            ('1', "Metaball", "Smooth blobby surface"),
            ('3', "Zhu-Bridson", "Fluid meshing"),
            ('4', "Anisotropic", "Stretched metaballs based on velocity"),
        ],
        default='1',
        update=update_mesh_callback
    )
    
    meshing_resolution_mode: EnumProperty(
        name="Resolution Mode",
        description="How to determine voxel size",
        items=[
            ('0', "Subdivide Max Radius", "Voxel size based on max particle radius"),
            ('1', "Fixed Voxel Length", "Explicit voxel size"),
        ],
        default='0',
        update=update_mesh_callback
    )
    
    meshing_resolution: FloatProperty(
        name="Resolution",
        description="Subdivisions per max radius",
        default=0.5,
        min=0.1,
        max=10.0,
        update=update_mesh_callback
    )
    
    meshing_voxel_length: FloatProperty(
        name="Voxel Length",
        description="Size of each voxel unit",
        default=0.1,
        min=0.001,
        soft_min=0.005,
        max=10.0,
        precision=3,
        update=update_mesh_callback
    )
    
    # Adaptive Resolution
    vert_refinement: IntProperty(
        name="Vertex Refinement",
        description="Iterations to refine vertices toward implicit surface (higher = more accurate)",
        default=3,
        min=0,
        max=10,
        update=update_mesh_callback
    )
    
    # Post Processing
    relax_iterations: IntProperty(
        name="Relax Iterations",
        description="Number of Laplacian smoothing passes",
        default=0,
        min=0,
        max=50,
        update=update_mesh_callback
    )
    
    relax_strength: FloatProperty(
        name="Relax Strength",
        description="Strength of each smoothing pass",
        default=0.5,
        min=0.0,
        max=1.0,
        update=update_mesh_callback
    )

    push_distance: FloatProperty(
        name="Push Distance",
        description="Inflate (positive) or Deflate (negative) the mesh",
        default=0.0,
        soft_min=-1.0,
        soft_max=1.0,
        unit='LENGTH',
        update=update_mesh_callback
    )
    
    use_smooth_shading: BoolProperty(
        name="Smooth Shading",
        description="Apply smooth shading to the generated mesh",
        default=True,
        update=update_mesh_callback
    )

    
    # Union of Spheres specific
    union_of_spheres_radius_scale: FloatProperty(
        name="Radius Scale",
        description="Multiplier for particle radius in Union of Spheres mode",
        default=1.5,
        min=0.1,
        max=10.0,
        update=update_mesh_callback
    )

    # Metaball specific
    metaball_radius_scale: FloatProperty(
        name="Radius Scale",
        description="Multiplier for particle radius",
        default=1.5,
        min=0.1,
        max=50.0,
        update=update_mesh_callback
    )
    
    metaball_isosurface_level: FloatProperty(
        name="Isovalue",
        description="Threshold for surface extraction",
        default=0.5,
        min=0.01,
        max=2.0,
        update=update_mesh_callback
    )
    
    # Plain Marching Cubes specific
    plain_marching_cubes_radius_scale: FloatProperty(
        name="Radius Scale",
        description="Multiplier for particle radius",
        default=2.0,
        min=0.1,
        max=50.0,
        update=update_mesh_callback
    )
    
    plain_marching_cubes_isovalue: FloatProperty(
        name="Isovalue",
        description="Surface threshold for marching cubes",
        default=0.5,
        min=0.01,
        max=2.0,
        update=update_mesh_callback
    )

    # Zhu-Bridson specific
    zhu_bridson_blend_radius_scale: FloatProperty(
        name="Blend Radius Scale",
        description="Radius multiplier for density blending",
        default=2.0,
        min=1.0,
        max=50.0,
        update=update_mesh_callback
    )
    
    zhu_bridson_enable_low_density_trimming: BoolProperty(
        name="Low Density Trimming",
        description="Remove mesh in areas of low particle density",
        default=False,
        update=update_mesh_callback
    )
    
    zhu_bridson_low_density_trimming_threshold: FloatProperty(
        name="Trim Threshold",
        description="Density threshold for trimming",
        default=1.0,
        min=0.0,
        max=10.0,
        update=update_mesh_callback
    )
    
    zhu_bridson_low_density_trimming_strength: FloatProperty(
        name="Trim Strength",
        description="Hardness of the trim edge",
        default=2.0,
        min=0.0,
        max=10.0,
        update=update_mesh_callback
    )
    
    # Anisotropic specific
    anisotropic_radius_scale: FloatProperty(
        name="Radius Scale",
        description="Base radius multiplier",
        default=1.0,
        min=0.1,
        max=50.0,
        update=update_mesh_callback
    )
    
    anisotropic_isosurface_level: FloatProperty(
        name="Isovalue",
        description="Surface threshold",
        default=0.5,
        min=0.01,
        max=2.0,
        update=update_mesh_callback
    )
    
    anisotropic_max_anisotropy: FloatProperty(
        name="Max Anisotropy",
        description="Maximum stretching factor (eigenvalue ratio)",
        default=4.0,
        min=1.0,
        max=20.0,
        update=update_mesh_callback
    )
    
    anisotropic_min_neighbor_count: IntProperty(
        name="Min Neighbors",
        description="Minimum neighbors required to compute anisotropy",
        default=6,
        min=1,
        max=50,
        update=update_mesh_callback
    )
    
    anisotropic_position_smoothing_weight: FloatProperty(
        name="Pos Smoothing",
        description="Weight for position smoothing",
        default=0.9,
        min=0.0,
        max=1.0,
        update=update_mesh_callback
    )


class FROST_PT_main_panel(Panel):
    """Main Frost panel in the 3D Viewport."""
    bl_label = "Frost Particle Meshing"
    bl_idname = "FROST_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Frost'
    
    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        
        if not obj:
            return

        # Check if we are checking a Frost object or a Source object
        frost_props = obj.frost_properties
        
        # Header / Status
        from . import operator
        gpu_backend_available = operator.has_gpu_backend()
        gpu_backend_name = operator.get_gpu_backend_name().lower()
        is_vulkan_backend = gpu_backend_name.startswith("vulkan")
        if operator.blender_frost_adapter is None and operator.adapter_load_error:
             box = layout.box()
             box.alert = True
             box.label(text="Core Load Error", icon='ERROR')
             box.label(text=operator.adapter_load_error)
        elif operator.adapter_load_error:
             box = layout.box()
             box.label(text="Native Bridge Note", icon='INFO')
             box.label(text=operator.adapter_load_error)
        
        # Top Row: Auto Update toggle
        row = layout.row()
        row.prop(frost_props, "auto_update", icon="FILE_REFRESH")
        gpu_toggle = row.row()
        gpu_toggle.enabled = gpu_backend_available
        gpu_toggle.prop(frost_props, "use_gpu", toggle=True, icon='SHADING_RENDERED')

        if frost_props.last_meshing_backend and (frost_props.use_gpu or frost_props.last_meshing_had_gpu_request):
            status_box = layout.box()
            status_row = status_box.row()
            status_row.alert = frost_props.last_meshing_used_fallback
            status_icon = 'ERROR' if frost_props.last_meshing_used_fallback else 'CHECKMARK'
            status_row.label(
                text=f"Last Meshing: {format_meshing_backend_label(frost_props.last_meshing_backend)}",
                icon=status_icon,
            )
            if frost_props.last_meshing_status:
                detail_row = status_box.row()
                detail_row.label(text=frost_props.last_meshing_status, icon='INFO')
        
        # Main Generate/Bake Button
        # Show Create button if no source, otherwise show Bake
        has_sources = frost_props.source_object is not None or len(frost_props.sources) > 0
        
        if not has_sources:
             layout.operator("frost.generate_mesh", text="Create Frost Mesh", icon='MESH_DATA')
        else:
             # Only Bake to Alembic button - mesh updates automatically via Auto Update
             layout.operator("frost.bake_alembic", text="Bake to Alembic", icon='EXPORT')
        
        # GPU Settings are now shown in the main Settings box below when GPU mode is active
        # Source settings
        box = layout.box()
        box.label(text="Particle Sources:", icon='PARTICLES')
        
        # Main Source
        box.prop(frost_props, "source_object", text="Main Source")
        
        if frost_props.source_object:
            col = box.column(align=True)
            col.prop(frost_props, "source_type")
            if frost_props.source_type == 'PARTICLE_SYSTEM':
                col.prop(frost_props, "particle_system_index")
            elif frost_props.source_type == 'MESH':
                # box.prop(frost_props, "default_radius") 
                # Keep Default Radius global for now 
                pass
        
        # Additional Sources
        box.label(text="Additional Sources:", icon='ADD')
        row = box.row()
        row.template_list("FROST_UL_source_list", "", frost_props, "sources", frost_props, "active_source_index", rows=3)
        
        col = row.column(align=True)
        col.operator("frost.add_source", icon='ADD', text="")
        col.operator("frost.remove_source", icon='REMOVE', text="")
        
        # Display settings for active source
        if frost_props.sources and frost_props.active_source_index >= 0 and len(frost_props.sources) > frost_props.active_source_index:
            active_item = frost_props.sources[frost_props.active_source_index]
            
            box_item = box.box()
            box_item.prop(active_item, "source_type")
            
            if active_item.source_type == 'PARTICLE_SYSTEM':
                 box_item.prop(active_item, "system_index")
                 
            # Global Default Radius (could be per item if needed, but currently global)
            # Actually, per-item radius override would be cool, but sticking to global for now
            # unless user requested per-object control?
            # "The user wants to add another object"
            # It's safer to keep "Default Radius" as a global fall-back or move it to item?
            # Let's keep Default Radius global for now to minimize complexity, 
            # or move it to "Meshing" section if it applies to Mesh sources.
            pass
            
        # Default Radius removed as per user request
        # box.prop(frost_props, "default_radius")
        
        # Meshing settings
        box = layout.box()
        box.label(text="Meshing:", icon='MESH_ICOSPHERE')
        
        if frost_props.use_gpu and gpu_backend_available and not is_vulkan_backend:
            # === GPU MODE ===
            # Method
            box.prop(frost_props, "gpu_meshing_method", text="")
             
        if frost_props.use_gpu and gpu_backend_available and not is_vulkan_backend:
            # GPU Mode: Unified controls
            res_box = box.box()
            res_box.label(text="GPU Resolution:", icon='PREFERENCES')
            
            # 1. Resolution Mode
            res_box.prop(frost_props, "meshing_resolution_mode")
            if frost_props.meshing_resolution_mode == '0':
                res_box.prop(frost_props, "meshing_resolution", text="Subdivisions")
            else:
                res_box.prop(frost_props, "meshing_voxel_length", text="Voxel Length")
                if frost_props.meshing_voxel_length < 0.002:
                     res_box.label(text="Limit: 0.002", icon='INFO')

            res_box.separator()
            # 2. Search Radius Scale (renamed from Radius Scale)
            # Default 1.5-2.0 is standard. Range 0-4.
            res_box.prop(frost_props, "gpu_search_radius_scale", text="Search Radius Scale")
            
            info_row = box.row()
            backend_name = operator.get_gpu_backend_name().upper()
            info_row.label(text=f"GPU Backend: {backend_name}", icon='EXPERIMENTAL')
            if frost_props.show_debug_log:
                 info_row.label(text="Debug Logs On", icon='CONSOLE')

        else:
            # === CPU MODE ===

            if frost_props.use_gpu and not gpu_backend_available:
                warn_row = box.row()
                warn_row.label(text="Current build has no GPU backend, using CPU settings", icon='INFO')
            box.prop(frost_props, "meshing_method", text="")
        
            # CPU Resolution Settings
            res_box = box.box()
            res_box.prop(frost_props, "meshing_resolution_mode")
            if frost_props.meshing_resolution_mode == '0':
                res_box.prop(frost_props, "meshing_resolution")
            else:
                res_box.prop(frost_props, "meshing_voxel_length")
                if frost_props.meshing_voxel_length < 0.015:
                    res_box.label(text="Warning: High Resolution!", icon='ERROR')
                    res_box.label(text="May cause freeze/crash.", icon='info')
                    
            # Adaptive Quality Control (CPU Only)
            res_box.separator(factor=0.5)
            res_box.prop(frost_props, "vert_refinement", slider=True)


        # Method-specific Settings Section
        method = frost_props.meshing_method
        method_box = layout.box()
        method_box.label(text="Settings", icon='SETTINGS')
        
        if frost_props.use_gpu and gpu_backend_available and not is_vulkan_backend:
            method_box.label(text="Zhu-Bridson GPU", icon='MOD_FLUIDSIM')
            method_box.label(text="GPU settings are handled above", icon='INFO')
        else:
            # CPU Mode: Show method-specific settings
            if method == '0': # Union of Spheres
                method_box.prop(frost_props, "union_of_spheres_radius_scale")

            elif method == '1': # Metaball
                method_box.prop(frost_props, "metaball_radius_scale")
                method_box.prop(frost_props, "metaball_isosurface_level")
                
            elif method == '2': # Plain Marching Cubes
                method_box.prop(frost_props, "plain_marching_cubes_radius_scale")
                method_box.prop(frost_props, "plain_marching_cubes_isovalue")
                
            elif method == '3': # Zhu-Bridson
                method_box.prop(frost_props, "zhu_bridson_blend_radius_scale")
                    
            elif method == '4': # Anisotropic
                method_box.prop(frost_props, "anisotropic_radius_scale")
                method_box.prop(frost_props, "anisotropic_isosurface_level")
                method_box.separator(factor=0.5)
                method_box.prop(frost_props, "anisotropic_max_anisotropy")
                method_box.prop(frost_props, "anisotropic_min_neighbor_count")
                method_box.prop(frost_props, "anisotropic_position_smoothing_weight")

        # Post Processing Section
        post_box = layout.box()
        post_box.label(text="Post Processing", icon='MODIFIER')
        
        # Basic adjustments (always visible)
        post_box.prop(frost_props, "push_distance")
        post_box.prop(frost_props, "use_smooth_shading")
        
        # Geometric Flow (Laplacian Smoothing) - collapsible
        flow_header = post_box.row()
        flow_header.prop(frost_props, "relax_iterations", text="Geometric Flow Passes")
        
        if frost_props.relax_iterations > 0:
            flow_col = post_box.column(align=True)
            flow_col.prop(frost_props, "relax_strength", text="Smoothing Strength", slider=True)
            flow_col.separator(factor=0.5)


class FROST_PT_info_panel(Panel):
    """Info panel for Frost."""
    bl_label = "Info"
    bl_idname = "FROST_PT_info_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Frost'
    bl_parent_id = "FROST_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        from . import bl_info as addon_info
        version = ".".join(str(part) for part in addon_info["version"])
        
        layout.label(text=f"Frost v{version}", icon='INFO')
        layout.label(text="Creator: Arthur Reboul Salze")
        layout.label(text="Core by AWS Thinkbox")
        
        if obj:
            frost_props = obj.frost_properties
            from . import operator
            backend_name = operator.get_gpu_backend_name().upper()
            layout.label(text=f"GPU Backend: {backend_name}", icon='SHADING_RENDERED' if operator.has_gpu_backend() else 'INFO')
            layout.label(
                text=f"Vulkan Runtime: {'Detected' if operator.has_vulkan_runtime() else 'Not detected'}",
                icon='CHECKMARK' if operator.has_vulkan_runtime() else 'INFO'
            )
            layout.label(
                text=f"Vulkan Compute: {'Ready' if operator.has_vulkan_compute() else 'Not ready'}",
                icon='CHECKMARK' if operator.has_vulkan_compute() else 'INFO'
            )
            layout.label(
                text=f"Vulkan Buffers: {'Ready' if operator.has_vulkan_storage_buffer() else 'Not ready'}",
                icon='CHECKMARK' if operator.has_vulkan_storage_buffer() else 'INFO'
            )
            if frost_props.last_meshing_backend:
                row = layout.row()
                row.alert = frost_props.last_meshing_used_fallback
                row.label(
                    text=f"Last Meshing: {format_meshing_backend_label(frost_props.last_meshing_backend)}",
                    icon='ERROR' if frost_props.last_meshing_used_fallback else 'CHECKMARK'
                )
                if frost_props.last_meshing_status:
                    layout.label(text=frost_props.last_meshing_status, icon='INFO')
            if frost_props.use_gpu and operator.has_gpu_backend():
                 row = layout.row()
                 row.label(text="GPU Acceleration: Enabled", icon='SHADING_RENDERED')
            elif operator.has_gpu_backend():
                 layout.label(text="GPU Acceleration: Disabled", icon='X')
            else:
                 layout.label(text="GPU Acceleration: Not available in current build", icon='INFO')

        # Debug Toggle in Info Panel as well (or keep it here if main panel is crowded)
        if obj:
            layout.prop(obj.frost_properties, "show_debug_log")


def register():
    bpy.utils.register_class(FrostSourceItem)
    bpy.utils.register_class(FROST_UL_source_list)
    bpy.utils.register_class(FROST_OT_add_source)
    bpy.utils.register_class(FROST_OT_remove_source)
    
    bpy.utils.register_class(FrostProperties)
    bpy.utils.register_class(FROST_PT_main_panel)
    bpy.utils.register_class(FROST_PT_info_panel)
    bpy.types.Object.frost_properties = PointerProperty(type=FrostProperties)


def unregister():
    del bpy.types.Object.frost_properties
    bpy.utils.unregister_class(FROST_PT_info_panel)
    bpy.utils.unregister_class(FROST_PT_main_panel)
    bpy.utils.unregister_class(FrostProperties)

    bpy.utils.unregister_class(FROST_OT_remove_source)
    bpy.utils.unregister_class(FROST_OT_add_source)
    bpy.utils.unregister_class(FROST_UL_source_list)
    bpy.utils.unregister_class(FrostSourceItem)
