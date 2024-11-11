import pyrender, trimesh
from scene.colmap_loader import read_intrinsics_binary, read_extrinsics_binary
from utils.graphics_utils import focal2fov
import numpy as np
import os
from argparse import ArgumentParser
import json, math
import cv2
from matplotlib import pyplot as plt

def get_camera(camera_intr, resolution=1):
    if camera_intr.model=="SIMPLE_PINHOLE":
        focal_length_x = camera_intr.params[0]
        FovY = focal2fov(focal_length_x, camera_intr.height)
        FovX = focal2fov(focal_length_x, camera_intr.width)
    elif camera_intr.model=="PINHOLE":
        focal_length_x = camera_intr.params[0]
        focal_length_y = camera_intr.params[1]
        FovY = focal2fov(focal_length_y, camera_intr.height)
        FovX = focal2fov(focal_length_x, camera_intr.width)
    else:
        assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        
    width = camera_intr.width // resolution
    height = camera_intr.height // resolution

    if focal_length_x is None or focal_length_y is None:
        if FovX is not None and FovY is not None:
            # Convert FOV to radians
            FovX_rad = math.radians(FovX)
            FovY_rad = math.radians(FovY)
            # Calculate focal lengths based on FOV and image dimensions
            focal_length_x = width / (2 * math.tan(FovX_rad / 2))
            focal_length_y = height / (2 * math.tan(FovY_rad / 2))
        else:
            raise ValueError("FovX, FovY or focal_length_x, focal_length_y is required")

    K = np.array([
        [focal_length_x, 0, width / 2],
        [0, focal_length_y, height / 2],
        [0, 0, 1]
    ])

    camera = pyrender.IntrinsicsCamera(
        fx=focal_length_x,
        fy=focal_length_y,
        cx=width / 2,
        cy=height / 2,
    )
    return camera

def read_extrinsics(args : ArgumentParser):
    split_file = os.path.join(args.model, "split.json")
    if os.path.exists(split_file):
        train_test_split = json.load(open(split_file))
        train_list = train_test_split["train"]
        test_list = train_test_split["test"]
    extrinsics = read_extrinsics_binary(os.path.join(args.model, "sparse/images.bin"))
    if train_test_split is not None:
        # get those partern in extrinsics which name in test_list
        test_extrinsics_idx = [extrinsic for extrinsic in extrinsics if extrinsics[extrinsic].name.split(".")[0] in test_list]
        test_extrinsics = {extrinsic: extrinsics[extrinsic] for extrinsic in test_extrinsics_idx}
    else :
        test_extrinsics = extrinsics
    return test_extrinsics

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def getTranspose(qvec, tvec):
    # rotation = R.from_quat(qvec)
    # rotation_matrix = rotation.as_matrix()
    
    rotation_matrix = qvec2rotmat(qvec)
    
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = rotation_matrix
    extrinsic_matrix[:3, 3] = tvec
    return extrinsic_matrix

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--mesh", type=str)
    
    args = parser.parse_args()
    
    # init pyrender scene
    scene = pyrender.Scene()
    mesh = trimesh.load(args.mesh)
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh)

    intrinsics = read_intrinsics_binary(os.path.join(args.model, "sparse/cameras.bin"))
    camera = get_camera(intrinsics[1], resolution=1)
    
    camera_node = scene.add(camera)
    
    r = pyrender.OffscreenRenderer(1964, 1104)
    
    test_extrinsics = read_extrinsics(args)
    for extrinsic_idx in test_extrinsics:
        extrinsic = test_extrinsics[extrinsic_idx]
        camera_pose = getTranspose(extrinsic.qvec, extrinsic.tvec)
        
        flip_yz = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        camera_pose = np.dot(flip_yz, camera_pose)
        
        camera_pose = np.linalg.inv(camera_pose)
        
        # camera_pose = np.linalg.inv(extrinsic_matrix)
        scene.set_pose(camera_node, pose=camera_pose)
        
        # add light
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3)
        scene.add(light, pose=camera_pose)
        color, depth = r.render(scene)
        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        color = cv2.resize(color, (color.shape[1]//2, color.shape[0]//2))
        cv2.imwrite(f"view_output/{extrinsic.name}", color)
        