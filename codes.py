
import bpy
import math
from mathutils import Vector

print ('starting')

def get_the_current_scene_objects():
    
    scene = bpy.context.scene
    objects_in_scene = scene.objects

    for obj in objects_in_scene:
        print(obj.name)



def make_object_mirrory(object_name): # if something is wrong check the hard-coded positions
    
    obj = bpy.data.objects.get(object_name)

    if obj is not None:
        # Create a new material
        material = bpy.data.materials.new(name="GlossyMaterial")
        obj.data.materials.append(material)

        # Enable 'Use Nodes' for the material
        material.use_nodes = True
        tree = material.node_tree
        nodes = tree.nodes

        # Clear existing nodes
        for node in nodes:
            nodes.remove(node)

        # Add Glossy BSDF node for reflection
        glossy_node = nodes.new(type='ShaderNodeBsdfGlossy')
        glossy_node.location = 200, 300
        
        
        # Set the roughness to zero
        glossy_node.inputs[1].default_value = 0.0

        
        # Add Material Output node
        output_node = nodes.new(type='ShaderNodeOutputMaterial')
        output_node.location = 400, 300
        
        # Connect Glossy BSDF to Material Output
        links = tree.links
        links.new(glossy_node.outputs[0], output_node.inputs[0])
    else:
        print("Object not found:", object_name)


def adjust_camera(camera_name, location, rotation):
    """
    Adjusts the location and rotation of the specified camera.

    Parameters:
    - camera_name: The name of the camera object.
    - location: A tuple (x, y, z) representing the new location of the camera.
    - rotation: A tuple (rx, ry, rz) representing the new rotation of the camera in degrees.
    """

    camera = bpy.data.objects.get(camera_name)

    if camera is not None and camera.type == 'CAMERA':
        camera.location = location

        # Convert degrees to radians for rotation
        rotation_radians = [math.radians(angle) for angle in rotation]
        # Set the rotation of the camera
        camera.rotation_euler = rotation_radians

        print("Camera '{}' adjusted successfully.".format(camera_name))
    else:
        print("Camera '{}' not found or is not a camera object.".format(camera_name))

def frame_object_in_camera(object_name, camera_name, scale_factor=2.8):
    # Get the object to focus on
    obj = bpy.data.objects[object_name]

    # Get the camera
    camera = bpy.data.objects.get(camera_name)
    if camera is None or camera.type != 'CAMERA':
        print("Error: Camera not found or not selected.")
        return

    # Get the object's mesh
    mesh = obj.data

    # Get the world matrix of the object
    obj_matrix = obj.matrix_world

    # Calculate bounding box
    bbox_corners = [obj_matrix @ v.co for v in mesh.vertices]
    min_x = min([v[0] for v in bbox_corners])
    max_x = max([v[0] for v in bbox_corners])
    min_y = min([v[1] for v in bbox_corners])
    max_y = max([v[1] for v in bbox_corners])
    min_z = min([v[2] for v in bbox_corners])
    max_z = max([v[2] for v in bbox_corners])

    # Calculate object center
    obj_center = ((max_x + min_x) / 2, (max_y + min_y) / 2, (max_z + min_z) / 2)

    # Calculate camera location (as Vector)
    distance = max(max_x - min_x, max_y - min_y, max_z - min_z) * scale_factor
    camera_location = Vector((obj_center[0] + distance, obj_center[1] - distance, obj_center[2] ))

    # Point camera to object center
    look_at = Vector(obj_center)
    direction = camera_location - look_at
    rot_quat = direction.to_track_quat('Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()

    # Set camera location
    camera.location = camera_location

    # Set camera view parameters
    bpy.context.scene.camera = camera
#    bpy.context.scene.render.resolution_x = 1920
#    bpy.context.scene.render.resolution_y = 1080
#    bpy.context.scene.render.pixel_aspect_x = 1
#    bpy.context.scene.render.pixel_aspect_y = 1
    bpy.context.scene.render.resolution_percentage = 100


# Sample_USES:________________________

# make_object_mirrory('Sphere') 
# adjust_camera("Camera", (5.0, 3.0, 4.0), (45.0, 0.0, 0.0))
frame_object_in_camera("Sphere", "Camera")

print ('done')

