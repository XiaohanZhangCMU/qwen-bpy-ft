# prompt_id : eval_003
# sample_idx: 3
# exit_code : 0
# prompt    : Create a night market food stall lane with three vendor stalls, hanging paper lanterns, folding tables, and plastic stools.
# ========================================================================

import bpy

# Clear default scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Add a camera
camera_data = bpy.data.cameras.new(name="Camera")
camera_object = bpy.data.objects.new("Camera", camera_data)
bpy.context.collection.objects.link(camera_object)
camera_object.location = (0, -10, 5)
camera_object.rotation_euler = (1.1, 0, 0)

# Add a light source
light_data = bpy.data.lights.new(name="Light", type='POINT')
light_object = bpy.data.objects.new("Light", light_data)
bpy.context.collection.objects.link(light_object)
light_object.location = (0, 0, 10)
light_data.energy = 1000

# Function to add a vendor stall
def add_vendor_stall(location):
    # Table
    bpy.ops.mesh.primitive_cube_add(size=1, location=(location[0], location[1], 0))
    table = bpy.context.object
    table.scale = (1, 1, 0.1)
    
    # Stool
    bpy.ops.mesh.primitive_cylinder_add(radius=0.1, depth=0.5, location=(location[0] + 0.5, location[1], 0.25))
    stool_leg = bpy.context.object
    stool_leg.scale = (0.1, 0.1, 0.5)
    
    # Paper lantern
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.2, location=(location[0], location[1] + 1, 0.5))
    lantern = bpy.context.object
    lantern.scale = (0.5, 0.5, 0.5)
    lantern.data.materials.append(bpy.data.materials.new(name="LanternMaterial"))
    lantern_material = lantern.data.materials[-1]
    lantern_material.diffuse_color = (1, 1, 0, 1)  # Yellow color for lanterns

# Add vendor stalls
add_vendor_stall((-2, 0, 0))
add_vendor_stall((0, 0, 0))
add_vendor_stall((2, 0, 0))

# Set active view to camera
bpy.context.scene.camera = camera_object