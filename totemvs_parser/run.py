import argparse
import glob
import os
import json
import numpy as np
import shutil
from scipy.spatial.transform import Rotation as R

parser = argparse.ArgumentParser(description="Isaac raw output to DTU-like format parser")
parser.add_argument("input_folder", help="The input folder of the raw data")
parser.add_argument("output_folder", help="The input folder of the raw data")
parser.add_argument(
    "-m", "--material", choices=["diffuse", "specular", "transparent"], help="Specify material type", required=True
)
parser.add_argument("-o", "--offset", action="store_true", help="Set if you want the camera's to be offset by 3")

args = parser.parse_args()

cameras = json.load(open(f"{args.input_folder}/cameras"))
cam_cnt = len(cameras)
distances = sorted([glob.glob(f"{args.input_folder}/distance_rp{i}*") for i in range(cam_cnt)])
instances = sorted([glob.glob(f"{args.input_folder}/instance_rp{i}*") for i in range(cam_cnt)])
images = sorted([glob.glob(f"{args.input_folder}/rp{i}*") for i in range(cam_cnt)])


def create_folders():
    for i, name in enumerate(images[0]):
        step_id = name.split("/")[-1].split("_")[2].split(".")[0]
        os.makedirs(f"{args.output_folder}/tote{step_id}/{args.material}", exist_ok=True)
        os.makedirs(f"{args.output_folder}/tote{step_id}/depth", exist_ok=True)
        os.makedirs(f"{args.output_folder}/tote{step_id}/instance", exist_ok=True)


def load_cams_if_exists():
    if os.path.isfile(f"{args.output_folder}/cams.json"):
        return json.load(open(f"{args.output_folder}/cams.json"))
    return {}


def create_cams():
    out = load_cams_if_exists()

    for i, cam in enumerate(cameras):
        cam_id = i
        if args.offset:
            cam_id += 3
        position = cam["position"]
        rotvec = cam["rotation"]
        fx = cam["focal_x"]
        fy = cam["focal_y"]
        cx = cam["center_x"]
        cy = cam["center_y"]
        near = cam["near"]
        far = cam["far"]

        intrinsic = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R.from_rotvec(rotvec).as_matrix()
        extrinsic[:, [0, 2]] = extrinsic[:, [2, 0]]  # Swap axis
        extrinsic[:, 0] *= -1

        extrinsic[:3, 3] = np.array(position)

        out[str(cam_id)] = {"extrinsic": extrinsic.tolist(), "intrinsic": intrinsic, "near": near, "far": far}
    with open(f"{args.output_folder}/cams.json", "w") as f:
        json.dump(out, f, indent=4)


def copy_files():
    for i in range(len(distances)):
        for j in range(len(distances[i])):
            dfile = distances[i][j]
            instancefile = instances[i][j]
            imfile = images[i][j]

            cam_index = int(imfile.split("/")[-1].split("_")[0][2:])
            if args.offset:
                cam_index += 3

            tote_index = imfile.split("/")[-1].split("_")[2].split(".")[0]
            shutil.copy(dfile, f"{args.output_folder}/tote{tote_index}/depth/{cam_index}.npy")
            shutil.copy(instancefile, f"{args.output_folder}/tote{tote_index}/instance/{cam_index}.json")
            shutil.copy(imfile, f"{args.output_folder}/tote{tote_index}/{args.material}/{cam_index}.png")


create_folders()
create_cams()
copy_files()
