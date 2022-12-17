'''
Author: Xingchen Li, Ruihang Jiang, Haochen Su
Date: 2022-12-17 16:04:59
LastEditTime: 2022-12-17 16:35:36
LastEditors: Ruihang Jiang
Description: Generate the dataset for ML model
'''
import logging
import kubric as kb
import numpy as np
from kubric.renderer.blender import Blender as KubricRenderer
import json
import bpy
import os
import shutil
import random

from kubric.simulator import PyBullet
from kubric.renderer import Blender

logging.basicConfig(level="INFO")


# --- CLI arguments
parser = kb.ArgumentParser()

# Configuration the camera position
parser.add_argument("--r_interval", type=int, default=3,
                    help="value of distance change between different r")

parser.add_argument("--r_change_number", type=int, default=1,
                    help="change time of parameter r")
parser.add_argument("--phi_change_number", type=int, default=5,
                    help="change time of parameter phi")
parser.add_argument("--theta_change_number", type=int, default=5,
                    help="change time of parameter theta")

# Configuration for the source of the assets
parser.add_argument("--kubasic_assets", type=str,
                    default="gs://kubric-public/assets/KuBasic/KuBasic.json")
parser.add_argument("--hdri_assets", type=str,
                    default="gs://kubric-public/assets/HDRI_haven/HDRI_haven.json")

# Configuration for the saving of blender
parser.add_argument("--save_state", dest="save_state", action="store_true")

# Configuration for the background setting
parser.add_argument("--bg_change_number", type=int, default=50,
                    help="change time of background")
# Configuration for the friction and floor_restituion attributes of the background
parser.add_argument("--floor_friction", type=float, default=0.3)
parser.add_argument("--floor_restitution", type=float, default=0.5)
parser.add_argument("--backgrounds_split", choices=["train", "test"],
                    default="train")

FLAGS = parser.parse_args()

'''
description: Calculate the new position of camera in each frame
param {*} frame: the index of the frame
param {*} r_interval: the difference of distance r in Spherical Coordinates
param {*} phi_change_num: the difference of phi in Spherical Coordinates
param {*} theta_change_num: the difference of theta in Spherical Coordinates
return {*} (x, y, z): the new position of camera in World Coordinates
use: 
'''
def camera_position_cal(frame, r_interval, phi_change_num, theta_change_num):

    phi_change = np.pi / 2 / (phi_change_num)
    theta_change = (2 * np.pi) / theta_change_num

    i = frame / (phi_change_num*theta_change_num)
    j = frame / theta_change_num % phi_change_num
    k = frame % theta_change_num

    r_new = r + i * r_interval + np.random.normal(loc=0, scale=1.5, size=1)[0]
    phi_new = np.pi / 2 - j * phi_change + np.random.uniform(-5, 5, 1)[0] / 180 * np.pi
    theta_new = k * theta_change + theta + np.random.uniform(-5, 5, 1)[0] / 180 * np.pi

    phi_new = np.maximum(phi_new, 0)
    phi_new = np.minimum(phi_new, np.pi / 2)

    theta_new = np.maximum(theta_new, 0)
    theta_new = np.minimum(theta_new, 2*np.pi)

    # These values of (x, y, z) will lie on the same sphere as the original camera.
    x = r_new * np.cos(theta_new) * np.sin(phi_new)
    y = r_new * np.sin(theta_new) * np.sin(phi_new)
    z = r_new * np.cos(phi_new)
    z += 3.5
    return x, y, z


'''
description: Generate a random direction of camera
return {*} (x_look[0], y_look[0], z_look[0]): random direction of camera in World Coordinates
'''
def camera_lookat_cal():
    x_look = np.random.normal(loc=0, scale=1, size=1)
    y_look = np.random.normal(loc=0, scale=1, size=1)
    z_look = np.random.normal(loc=3.5, scale=1, size=1)
    return x_look[0], y_look[0], z_look[0] 


'''
description: Generate a random rotation of the probe. The rotation is decribed by a quaternion
return {*} (qua_w, qua_x, qua_y, qua_z): the rotation quaternion of the probe
'''
def quaternion_cal():
    rotation_cita = np.random.uniform(0, 2*np.pi, 1)[0]

    rotation_axis_r = 1
    rotation_axis_phi = np.random.uniform(0, np.pi, 1)[0]
    rotation_axis_theta = np.random.uniform(0, 2*np.pi, 1)[0]

    rotation_axis_x = rotation_axis_r * np.sin(rotation_axis_phi) * np.cos(rotation_axis_theta)
    rotation_axis_y = rotation_axis_r * np.sin(rotation_axis_phi) * np.sin(rotation_axis_theta)
    rotation_axis_z = rotation_axis_r * np.cos(rotation_axis_phi)
    # print(rotation_cita/(2*np.pi)*180, rotation_axis_x, rotation_axis_y, rotation_axis_z)
    qua_w = np.cos(rotation_cita/2)
    qua_x = np.sin(rotation_cita/2) * rotation_axis_x
    qua_y = np.sin(rotation_cita/2) * rotation_axis_y
    qua_z = np.sin(rotation_cita/2) * rotation_axis_z

    return qua_w, qua_x, qua_y, qua_z


'''
description: Calculate the rotation matrix according to the quaternion
param {*} qua_w: the first element in quaternion
param {*} qua_x: the second element in quaternion
param {*} qua_y: the third element in quaternion
param {*} qua_z: the fourth element in quaternion
return {*} probe_rotation_matrix: the rotation matrix according to the quaternion
use: 
'''
def probe_rotation_cal(qua_w, qua_x, qua_y, qua_z):
    probe_rotation_matrix = np.eye(4)
    # First row of the rotation matrix
    r00 = 2 * (qua_w * qua_w + qua_x * qua_x) - 1
    r01 = 2 * (qua_x * qua_y - qua_w * qua_z)
    r02 = 2 * (qua_x * qua_z + qua_w * qua_y)
     
    # Second row of the rotation matrix
    r10 = 2 * (qua_x * qua_y + qua_w * qua_z)
    r11 = 2 * (qua_w * qua_w + qua_y * qua_y) - 1
    r12 = 2 * (qua_y * qua_z - qua_w * qua_x)
     
    # Third row of the rotation matrix
    r20 = 2 * (qua_x * qua_z - qua_w * qua_y)
    r21 = 2 * (qua_y * qua_z + qua_w * qua_x)
    r22 = 2 * (qua_w * qua_w + qua_z * qua_z) - 1

    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])

    probe_rotation_matrix[0:3, 0:3] = rot_matrix

    return probe_rotation_matrix


# Get the change number of the background
bg_change_num = FLAGS.bg_change_number
# For each selected background, generate the corresponding probe images
for bg_index in range(bg_change_num):
    # Configuration for the output file path
    FLAGS.job_dir = './ML_Project/output/dataset/bg' + str(bg_index).zfill(4)
    output_path = FLAGS.job_dir
    output_txt_path = './ML_Project/output/dataset/'

    # Get the configuration for the parameters of camera position
    r_interval = FLAGS.r_interval
    r_change_num = FLAGS.r_change_number
    phi_change_num = FLAGS.phi_change_number
    theta_change_num = FLAGS.theta_change_number

    print("r_interval: {}, r_change_num: {}, phi_change_num: {}, theta_change_num: {}".format(
        r_interval, r_change_num, phi_change_num, theta_change_num))

    # Generate blank folders for output
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    os.makedirs(os.path.join(output_path, 'matrix'))
    os.makedirs(os.path.join(output_path, 'rgba'))

    # Generate scene, random generator and setting the file path
    scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)
    # Setting the resource
    kubasic = kb.AssetSource.from_manifest(FLAGS.kubasic_assets)
    hdri_source = kb.AssetSource.from_manifest(FLAGS.hdri_assets)
    # Split the background resources and we will only use the training parts in our project
    train_backgrounds, test_backgrounds = hdri_source.get_test_split(fraction=0.1)

    # Print the basic information of background
    if bg_index == 0:
        logging.info("Training backgrounds number: " + str(len(train_backgrounds)))
        logging.info(train_backgrounds)
        logging.info("Testing backgrounds number: " + str(len(test_backgrounds)))
        logging.info(test_backgrounds)

        # Record the background information
        train_bg_txt = os.path.join(output_txt_path, 'train_bg.txt')
        with open(train_bg_txt, 'w') as f:
            f.write("The number of backgrounds in training dataset is " + str(len(train_backgrounds)))
            f.write("\n")
            i = 0
            for single_bg in train_backgrounds:
                f.write("bg" + str(i) + ": " + str(single_bg))
                f.write("\n")
                i += 1
        test_bg_txt = os.path.join(output_txt_path, 'test_bg.txt')
        with open(test_bg_txt, 'w') as f:
            f.write("The number of backgrounds in testing dataset is " + str(len(test_backgrounds)))
            f.write("\n")
            for single_bg in test_backgrounds:
                f.write("bg" + str(i) + ": " + str(single_bg))
                f.write("\n")
                i += 1

    # Select one background from the background dataset
    if FLAGS.backgrounds_split == "train":
        hdri_id = train_backgrounds[bg_index]
    else:
        hdri_id = test_backgrounds[bg_index]
    # Create the HDRI according to the selected background
    background_hdri = hdri_source.create(asset_id=hdri_id)
    logging.info("Using background %s", hdri_id)

    # Generate images of probe with different angles and distance in the selected background
    for frame in range(0, r_change_num*phi_change_num*theta_change_num):
        # Redefine the scene
        scene = kb.Scene(resolution=(256, 256), frame_start=0, frame_end=1)
        # Generate the blender object
        renderer = Blender(scene, scratch_dir, samples_per_pixel=64)
        
        # Add the selected background in the scene and blender
        scene.metadata["background"] = hdri_id
        renderer._set_ambient_light_hdri(background_hdri.filename)

        # Create the Dome object and add the dome in scene and blender
        dome = kubasic.create(asset_id="dome", name="dome",
                            friction=FLAGS.floor_friction,
                            restitution=FLAGS.floor_restitution,
                            static=True, background=True)
        assert isinstance(dome, kb.FileBasedObject)
        scene += dome
        dome_blender = dome.linked_objects[renderer]
        # Load the texture of background in scene
        texture_node = dome_blender.data.materials[0].node_tree.nodes["Image Texture"]
        texture_node.image = bpy.data.images.load(background_hdri.filename)

        # Add Klevr-like lights to the scene
        scene += kb.assets.utils.get_clevr_lights()
        scene.ambient_illumination = kb.Color(0.05, 0.05, 0.05)

        # Add a camera in the scene
        scene.camera = kb.PerspectiveCamera(name="camera")

        # Generate the probe object
        obj_probe = kb.FileBasedObject(
            asset_id="probe",
            render_filename="./Models/TestJBHigh.obj",
            bounds=((-100, -100, -100), (100, 100, 100)),
            simulation_filename=None,
            position=(0, 0, 0),
            )

        # Generate a random rotation of the probe
        qua_w, qua_x, qua_y, qua_z = quaternion_cal()
        # Generate a random position of the probe
        probe_z = np.random.uniform(0, 15, 1)[0]
        # Setting the scale, rotation and the position of the probe
        obj_probe.scale = (0.03, 0.03, 0.03)
        obj_probe.quaternion = (qua_w, qua_x, qua_y, qua_z)
        obj_probe.position=(0, 0, 3.5+probe_z) 
        # Add the probe in the scene
        scene += obj_probe
        
        # Get the original coordinates of the probe in Spherical Coordinates
        original_camera_position = (7, -7, 4)
        r = np.sqrt(sum(a * a for a in original_camera_position))
        phi = np.arccos(original_camera_position[2] / r)
        theta = np.arccos(original_camera_position[0] / (r * np.sin(phi)))
        phi = 0

        # Get the position of the camera in each frame
        x, y, z = camera_position_cal(frame, r_interval, phi_change_num, theta_change_num)
        # Get the random coordinates of the direction of camera
        x_look, y_look, z_look = camera_lookat_cal()

        # Setting the position and direction of camera in the scene
        scene.camera.position = (x, y, z+probe_z)
        scene.camera.look_at((x_look, y_look, z_look+probe_z))

        # Record the position and quaternion in different frame
        scene.camera.keyframe_insert("position", frame)
        scene.camera.keyframe_insert("quaternion", frame)

        # Calculate the probe rotation matrix
        probe_rotation_matrix = probe_rotation_cal(qua_w, qua_x, qua_y, qua_z)

        # Record the label information in json files
        output_dir_matrix = os.path.join(output_path, "matrix")
        # Get the extrinsic matrix of camera and calculate the inverse of it
        camera_extrinsic = np.array(scene.camera.matrix_world)
        b = np.identity(np.shape(camera_extrinsic)[0])
        matrix_probe = np.linalg.solve(camera_extrinsic, b)
        # Get the extrinsic matrix of probe
        probe_matrix_world_array = np.array(obj_probe.matrix_world)
        # Calculate the new extrinsic matrix of camera with the probe as a reference point
        matrix_probe_in_cam_coord = matrix_probe @ probe_matrix_world_array
        # Get the intrinsic matrix of the camera
        camera_intrinsic = scene.camera.intrinsics
        # Record informations in json files for each images
        extrinsic_dict = {'matrix_probe_in_cam_coord': (matrix_probe_in_cam_coord).tolist(),
                          'matrix_probe_in_world_coord': (probe_matrix_world_array).tolist(),
                          'intrinsic_matrix_camera': (camera_intrinsic).tolist(),
                          'extrinsic_matrix_camera': (camera_extrinsic).tolist()}
        json_file_name = "bg" + str(bg_index).zfill(4) + "_" + "extrinsic_" + str(frame).zfill(5) + ".json"
        extrinsic_json = json.dumps(extrinsic_dict, indent=4)
        with open(os.path.join(output_dir_matrix, json_file_name), 'w') as json_file:
            json_file.write(extrinsic_json)
        logging.info("Saving " + os.path.join(output_dir_matrix, json_file_name))
        
        # Record the blender data for each images
        output_dir_matrix = os.path.join(output_path, "blend")
        blend_file_name = "bg" + str(bg_index).zfill(4) + "_" + "blend_" + str(frame).zfill(5) + ".blend"
        # renderer.save_state(os.path.join(output_dir_matrix, blend_file_name))

        # Record the rgba pictures from the camera
        frames = renderer.render_still()
        output_dir_matrix = os.path.join(output_path, "rgba")
        rgba_file_name = "bg" + str(bg_index).zfill(4) + "_" + "rgba_" + str(frame).zfill(5) + ".png"
        kb.write_png(frames["rgba"], os.path.join(output_dir_matrix, rgba_file_name))