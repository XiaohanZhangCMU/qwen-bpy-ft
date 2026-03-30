# prompt_id : eval_005
# sample_idx: 2
# exit_code : 0
# prompt    : Create a café interior with four small round tables, matching chairs, a serving counter, and pendant lights overhead.
# ========================================================================

import bpy

# Clear existing data
bpy.ops.wm.read_factory_settings(use_empty=True)

# Add a camera
camera_data = bpy.data.cameras.new(name="Camera")
camera_object = bpy.data.objects.new("Camera", camera_data)
bpy.context.collection.objects.link(camera_object)
camera_object.location = (0, -10, 5)
camera_object.rotation_euler = (1.1, 0, 0)

# Add a light
light_data = bpy.data.lights.new(name="Light", type='POINT')
light_object = bpy.data.objects.new("Light", light_data)
bpy.context.collection.objects.link(light_object)
light_object.location = (0, 0, 5)
light_object.data.energy = 1000

# Function to add a table and chairs
def add_table_and_chairs(x, y):
    # Table
    bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=0.1, location=(x, y, 0))
    table = bpy.context.object
    table.name = "Table"
    
    # Chairs
    for i in range(4):
        chair_location = (x + (-0.5 if i % 2 == 0 else 0.5), y + (-0.5 if i < 2 else 0.5), 0)
        bpy.ops.mesh.primitive_cone_add(vertices=4, radius1=0.2, radius2=0, depth=0.5, location=chair_location)
        chair = bpy.context.object
        chair.name = f"Chair_{i}"
        chair.rotation_euler = (1.57, 0, 0)

# Add tables and chairs
add_table_and_chairs(-2, 0)
add_table_and_chairs(2, 0)
add_table_and_chairs(-2, -3)
add_table_and_chairs(2, -3)

# Serving Counter
bpy.ops.mesh.primitive_cube_add(size=2, location=(-2, 3, 0.5))
counter = bpy.context.object
counter.name = "Counter"

# Pendant Lights
for i in range(4):
    light_location = (-2 + i * 2, 3, 2)
    bpy.ops.object.light_add(type='SPOT', location=light_location)
    pendant_light = bpy.context.object
    pendant_light.data.energy = 500
    pendant_light.rotation_euler = (1.57, 0, 0)

# Final adjustments
bpy.context.scene.camera = camera_object
bpy.context.view_layer.objects.active = camera_object