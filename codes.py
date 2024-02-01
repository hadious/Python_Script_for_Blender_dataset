
import bpy
import math
from mathutils import Vector
import numpy as np
import os
from mathutils import Matrix
#import cv2

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
 
 


def generate_depth_map(camera_name, object_name):
    # Get the camera and object by name
    camera = bpy.data.objects.get(camera_name)
    obj = bpy.data.objects.get(object_name)

    if camera is None:
        print("Error: Camera not found.")
        return None
    if obj is None:
        print("Error: Object not found.")
        return None

    # Set the scene camera
    bpy.context.scene.camera = camera

    # Set up rendering settings
    output_directory = "/home/hadi/Desktop/GF/DataSet/MyDataset/"  # Change this to your desired directory
    output_filename = "depth_map"

    bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
    bpy.context.scene.render.image_settings.color_depth = '32'
    bpy.context.scene.render.filepath = output_directory + output_filename
    bpy.context.scene.render.use_compositing = True

    # Set up nodes for rendering the depth map
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    # Clear default nodes
    for node in tree.nodes:
        tree.nodes.remove(node)

    # Create nodes for rendering the depth map
    render_layers_node = tree.nodes.new('CompositorNodeRLayers')
    normalize_node = tree.nodes.new('CompositorNodeNormalize')

    # Link nodes
    links.new(render_layers_node.outputs[2], normalize_node.inputs[0])
    links.new(normalize_node.outputs[0], tree.nodes.new('CompositorNodeOutputFile').inputs[0])

    # Render the scene
    bpy.ops.render.render(write_still=True)

    # Load the depth map and return it as a numpy array
    depth_map_path = bpy.path.abspath(output_directory + output_filename)
    depth_map = bpy.data.images.load(depth_map_path)

    # Convert the depth map to a numpy array
    depth_map_array = np.array(depth_map.pixels[:])
    width, height = depth_map.size[:]

    # Reshape the array to a 2D array
    depth_map_array = depth_map_array.reshape((height, width))

    # Flip the depth map vertically
    depth_map_array = np.flipud(depth_map_array)

    return depth_map_array

def save_depth_map_as_image(depth_map_array, output_filename):
    # Normalize depth map values to [0, 1]
    normalized_depth_map = (depth_map_array - np.min(depth_map_array)) / (np.max(depth_map_array) - np.min(depth_map_array))

    # Create a new image
    depth_map_image = bpy.data.images.new("Depth Map", width=len(depth_map_array[0]), height=len(depth_map_array))

    # Flatten the depth map array
    depth_map_pixels = normalized_depth_map.flatten()

    # Set the image pixels
    depth_map_image.pixels = depth_map_pixels

    # Save the image to a file
    depth_map_image.file_format = 'PNG'  # Change the format if needed
    depth_map_image.filepath_raw = output_filename
    depth_map_image.save()

def get_depth():
    """Obtains depth map from Blender render.
    :return: The depth map of the rendered camera view as a numpy array of size (H,W).
    """
    z = bpy.data.images['Viewer Node']
    w, h = z.size
    dmap = np.array(z.pixels[:], dtype=np.float32) # convert to numpy array
    dmap = np.reshape(dmap, (h, w, 4))[:,:,0]
    dmap = np.rot90(dmap, k=2)
    dmap = np.fliplr(dmap)
    return dmap

def dmap2norm(dmap):
    """Computes surface normals from a depth map.
    :param dmap: A grayscale depth map image as a numpy array of size (H,W).
    :return: The corresponding surface normals map as numpy array of size (H,W,3).
    """
    zx = cv2.Sobel(dmap, cv2.CV_64F, 1, 0, ksize=5)
    zy = cv2.Sobel(dmap, cv2.CV_64F, 0, 1, ksize=5)

    # convert to unit vectors
    normals = np.dstack((-zx, -zy, np.ones_like(dmap)))
    length = np.linalg.norm(normals, axis=2)
    normals[:, :, :] /= length

    # offset and rescale values to be in 0-1
    normals += 1
    normals /= 2
    return normals[:, :, ::-1].astype(np.float32)

def modify_pattern_plate(plate_name, position, rotation_angles, pattern_image_path):
    # Get the plate object by name
    plate_object = bpy.data.objects.get(plate_name)
    if not plate_object:
        print(f"Plate object with name '{plate_name}' not found.")
        return
    
    # Set the position
    plate_object.location = position
    
    # Set the rotation (in radians)
    plate_object.rotation_euler = rotation_angles
    
    # Apply the pattern texture
    # Create a new material
    material = bpy.data.materials.new(name="PatternMaterial")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    
    # Clear default nodes
    for node in nodes:
        nodes.remove(node)
    
    # Add image texture node
    image_texture_node = nodes.new(type='ShaderNodeTexImage')
    image_texture_node.image = bpy.data.images.load(pattern_image_path)
    
    # Add diffuse BSDF node
    diffuse_node = nodes.new(type='ShaderNodeBsdfDiffuse')
    
    # Add output node
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    
    # Link nodes
    links.new(image_texture_node.outputs[0], diffuse_node.inputs[0])
    links.new(diffuse_node.outputs[0], output_node.inputs[0])
    
    # Assign the material to the plate object
    if plate_object.data.materials:
        plate_object.data.materials[0] = material
    else:
        plate_object.data.materials.append(material)

def write_depth_map_into_numpy(fname, DMAP):
    with open(fname, "wb") as f:
        np.savez(f, DMAP)

def write_frame_into_image_file(camera_name):
    
    
    
# Sample_USE:________________________

#make_object_mirrory('Sphere') 
#adjust_camera("Camera", (5.0, 3.0, 4.0), (45.0, 0.0, 0.0))
#frame_object_in_camera("Cube", "Camera",scale_factor=2.9)
#depth_map = generate_depth_map("Camera", "Cube")
#print (depth_map.shape)
#save_depth_map_as_image(depth_map, "depth_map.png")
#DMAP = get_depth()
#DMAP = dmap2norm(DMAP)

#fname = "depth.npz" 
#with open(fname, "wb") as f:
#    np.savez(f, DMAP)
    
#modify_pattern_plate("Plane", (1.0, 2.0, 0.0), (0.0, 5.0, 1.57), "pattern_1.png")

###############################################################################################################################


    
    

def target_objects_name(object_input_import_from_file, import_path):
    
    return object_names # is an array of the object names, if from blender then just the names, if from the any imported file path 
                            # return the path to that file

def import_object_and_get_the_name(object_path):
    # todo: import the object from the path into blender and then return the equivalent name in the blender
    
    return object_name

def perform_capturing_2D_depthMap(obj_name, camera_name, camera_position_angle_change_matrix): 
    # camera_position_angle_change_matrix or number of samples
    # to do: use adjust_camera => frame_object_in_camera => generate_depth_map?? or  get_depth => save_depth_map_as_image???? 
    
    return (2D_images, DepthMap)

def perform_pattern_plate_change(pattern_position_angle_change_matrix, i):
# to do     change the pattern plate using modify_pattern_plate, manage the images from file or fixed patterns !!!
    return 


def write_into_files(2D_images, DepthMap):
    # to do : write_depth_map_into_numpy and write_frame_into_image_file


def main():
    # Camera Settings
    camera_name = 'Camera'
    camera_position_angle_change_matrix = ?
    ##################################################################
    # Patten setting
    pattern_position_angle_change_matrix = ?
    number_of_pattern_plate_changes = ?
    ##################################################################
    
    object_input_import_from_file = False # True if objects are imported from a path e.x. .stl files
    import_path = ''
    if object_input_import_from_file:
        import_path = 'SOME_PATH'
    #################################################################
    
    target_objects_name = target_objects_name(object_input_import_from_file,import_path)
    
    for obj_name in target_objects_name:
        if object_input_import_from_file:
            obj_name = import_object_and_get_the_name(obj)
            
        make_object_mirrory(obj_name)
        for i in range(number_of_pattern_plate_changes):
            2D_images, DepthMap = perform_capturing_2D_depthMap (obj_name, camera_name, camera_position_angle_change_matrix)
            write_into_files(2D_images, DepthMap)
            perform_pattern_plate_change(pattern_position_angle_change_matrix, i)
            
    
    
    
main()




print ('done')


