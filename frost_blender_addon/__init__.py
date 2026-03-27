"""
Frost - Particle Meshing for Blender

A Blender addon for converting particles into meshes using various algorithms
inspired by Thinkbox Frost.

Supported Methods:
- Union of Spheres: Fast preview meshing
- Metaball: Smooth blobby surfaces
- Zhu-Bridson: Advanced fluid meshing

Author: Arthur Reboul Salze
Version: 1.25.0
Blender: 5+
"""

bl_info = {
    "name": "Frost Particle Meshing",
    "author": "Arthur Reboul Salze",
    "version": (1, 25, 0),
    "blender": (5, 0, 0),
    "location": "View3D > Sidebar > Frost",
    "description": "Particle meshing plugin with Thinkbox CPU and the current Vulkan GPU backend work.",
    "warning": "",
    "doc_url": "https://github.com/ArthurReboulSalze/Frost-for-Blender",
    "category": "Mesh",
}


import bpy
import sys
import os
from bpy.app.handlers import persistent

# Add addon directory to Python path for imports
addon_dir = os.path.dirname(__file__)
if addon_dir not in sys.path:
    sys.path.insert(0, addon_dir)

# Import addon modules
from . import operator
from . import ui


@persistent
def frost_frame_change_handler(scene):
    """Update Frost meshes on frame change"""
    # Avoid errors if C++ not loaded or module unloaded
    if operator is None or not operator.blender_frost_adapter:
        return

    # Iterate over objects in the current scene
    try:
        for obj in scene.objects:
            if obj.type == 'MESH':
                # Check if it's a configured Frost object
                props = getattr(obj, "frost_properties", None)
                
                # Check for Main Source OR Additional Sources
                has_source = props and (props.source_object or len(props.sources) > 0)
                
                if has_source and props.auto_update:
                    try:
                        # Context might be restricted in handlers, pass None or constructed context
                        # update_frost_mesh only needs context for potential lookups (which we might not need if we have the obj)
                        operator.update_frost_mesh(obj, bpy.context)
                    except Exception as e:
                        print(f"Frost Animation Error on {obj.name}: {e}")
    except ReferenceError:
        pass # Object already deleted during shutdown
    except Exception:
        pass # General shutdown noise


# Visibility Detection Handler
# Tracks visibility state to trigger update on Rising Edge (Hidden -> Visible)
_visibility_cache = {}

@persistent
def frost_visibility_handler(scene, depsgraph):
    """
    Check for objects transitioning from Hidden -> Visible.
    Triggers an immediate mesh update if Auto Update is enabled.
    """
    # Avoid errors if operator/adapter not loaded
    if operator is None or not operator.blender_frost_adapter:
        return

    try:
        # Check all objects in scene (robust) 
        # depsgraph updates might not contain all info for property changes
        for obj in scene.objects:
            if obj.type == 'MESH':
                # Quick check if it's a Frost object
                props = getattr(obj, "frost_properties", None)
                if not props: continue # Not a Frost object logic

                # Determine current state
                is_visible = not obj.hide_viewport
                
                # Retrieve previous state (Default True to avoid spike on initial load)
                was_visible = _visibility_cache.get(obj.name, True)
                
                # Logic: If becoming visible NOW, and was previously hidden
                # AND Auto Update is ON -> Trigger Update
                if is_visible and not was_visible:
                    if props.auto_update:
                        try:
                            # We must re-check if source exists
                            if props.source_object or len(props.sources) > 0:
                                operator.update_frost_mesh(obj, bpy.context)
                        except Exception as e:
                            print(f"Frost Visibility Update Error: {e}")

                # Update cache
                _visibility_cache[obj.name] = is_visible
                
    except Exception:
        pass


# Registration
def register():
    """Register addon classes and properties."""
    ui.register()
    operator.register()
    
    if frost_frame_change_handler not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(frost_frame_change_handler)

    if frost_visibility_handler not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(frost_visibility_handler)
    
    print("Frost Particle Meshing addon registered")


def unregister():
    """Unregister addon classes and properties."""
    if frost_frame_change_handler in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(frost_frame_change_handler)

    if frost_visibility_handler in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(frost_visibility_handler)
        
    operator.unregister()
    ui.unregister()
    
    # Explicitly unload adapter to prevent TBB shutdown crashes
    if operator:
        operator.unload_adapter()
    
    print("Frost Particle Meshing addon unregistered")


if __name__ == "__main__":
    register()
