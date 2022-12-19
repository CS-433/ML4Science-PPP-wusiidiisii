'''
Author: YourName
Date: 2022-12-17 23:27:33
LastEditTime: 2022-12-19 01:39:03
LastEditors: YourName
Description: 
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
parser.add_argument("--phi_change_number", type=int, default=1,
                    help="change time of parameter phi")
parser.add_argument("--theta_change_number", type=int, default=1,
                    help="change time of parameter theta")

# Configuration for the source of the assets
parser.add_argument("--kubasic_assets", type=str,
                    default="gs://kubric-public/assets/KuBasic/KuBasic.json")
parser.add_argument("--hdri_assets", type=str,
                    default="gs://kubric-public/assets/HDRI_haven/HDRI_haven.json")

# Configuration for the saving of blender
parser.add_argument("--save_state", dest="save_state", action="store_true")

# Configuration for the background setting
parser.add_argument("--prediction_number", type=int, default=100,
                    help="prediction_number")
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


def rotation_convert_quaternion(rotation_matrix):
    m00 = rotation_matrix[0, 0]
    m01 = rotation_matrix[0, 1]
    m02 = rotation_matrix[0, 2]
    m10 = rotation_matrix[1, 0]
    m11 = rotation_matrix[1, 1]
    m12 = rotation_matrix[1, 2]
    m20 = rotation_matrix[2, 0]
    m21 = rotation_matrix[2, 1]
    m22 = rotation_matrix[2, 2]

    tr = m00 + m11 + m22

    if (tr > 0):
        S = np.sqrt(tr+1.0) * 2
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif ((m00 > m11)&(m00 > m22)):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S; 
        qz = (m02 + m20) / S; 
    elif (m11 > m22): 
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S
    return (qw, qx, qy, qz)


# Get the change number of the background
prediction_number = FLAGS.prediction_number
# For each selected background, generate the corresponding probe images
for prediction_index in range(prediction_number):
    # Configuration for the output file path
    bg_index = int(prediction_index/2)
    FLAGS.job_dir = './ML_Project/prediction/model_output_test_1/rgba'
    output_path = FLAGS.job_dir

    # Get the configuration for the parameters of camera position
    r_interval = FLAGS.r_interval
    r_change_num = FLAGS.r_change_number
    phi_change_num = FLAGS.phi_change_number
    theta_change_num = FLAGS.theta_change_number

    # Generate blank folders for output
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Generate scene, random generator and setting the file path
    scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)
    # Setting the resource
    kubasic = kb.AssetSource.from_manifest(FLAGS.kubasic_assets)
    hdri_source = kb.AssetSource.from_manifest(FLAGS.hdri_assets)
    # Split the background resources and we will only use the training parts in our project
    train_backgrounds, test_backgrounds = hdri_source.get_test_split(fraction=0.1)

    # Select one background from the background dataset
    if FLAGS.backgrounds_split == "train":
        hdri_id = train_backgrounds[bg_index]
    else:
        hdri_id = test_backgrounds[bg_index]
    # Create the HDRI according to the selected background
    background_hdri = hdri_source.create(asset_id=hdri_id)
    logging.info("Using background %s", hdri_id)

    # Generate images of probe with different angles and distance in the selected background
    for frame in range(0, 1):
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

        prediction_json_file_path = './ML_Project/prediction/model_output_test_1/matrix'
        prediction_json_file_name = 'prediction_label' + str(prediction_index).zfill(4) + '.json'
        with open(os.path.join(prediction_json_file_path, prediction_json_file_name), 'r') as f:
            data = json.load(f)
        f.close()

        pred_matrix_probe_in_cam_coord = np.array(data['matrix_probe_in_cam_coord_pre'])
        camera_extrinsic = np.array(data['extrinsic_matrix_camera'])

        pred_matrix_probe_in_world_coord = camera_extrinsic @ pred_matrix_probe_in_cam_coord
        pred_probe_rotation = pred_matrix_probe_in_world_coord[0:3, 0:3]
        pred_probe_position = pred_matrix_probe_in_world_coord[0:3, 3]
        pred_camera_rotation = camera_extrinsic[0:3, 0:3]
        pred_camera_position = camera_extrinsic[0:3, 3]

        pred_probe_quaternion = rotation_convert_quaternion(pred_probe_rotation)
        pred_camera_quaternion = rotation_convert_quaternion(pred_camera_rotation)

        # Setting the scale, rotation and the position of the probe
        obj_probe.scale = (0.03, 0.03, 0.03)
        obj_probe.quaternion = pred_probe_quaternion
        obj_probe.position = pred_probe_position
        # Add the probe in the scene
        scene += obj_probe

        # Setting the position and direction of camera in the scene
        scene.camera.position = pred_camera_position
        scene.camera.quaternion = pred_camera_quaternion
        # scene.camera.look_at((0, 0, pred_probe_position[2]))

        # Record the position and quaternion in different frame
        scene.camera.keyframe_insert("position", frame)
        scene.camera.keyframe_insert("quaternion", frame)
        
        # Record the blender data for each images
        output_dir_matrix = os.path.join(output_path, "blend")
        blend_file_name = "bg" + str(bg_index).zfill(4) + "_" + "blend_" + str(prediction_index).zfill(5) + ".blend"
        # renderer.save_state(os.path.join(output_dir_matrix, blend_file_name))

        # Record the rgba pictures from the camera
        frames = renderer.render_still()
        rgba_file_name = "bg" + str(bg_index).zfill(4) + "_" + "rgba_" + str(prediction_index).zfill(5) + ".png"
        kb.write_png(frames["rgba"], os.path.join(output_path, rgba_file_name))