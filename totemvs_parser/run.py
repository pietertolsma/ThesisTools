import argparse
import glob
import os
import json
import numpy as np
import shutil
from scipy.spatial.transform import Rotation as R
from PIL import Image
from collections import defaultdict
import tqdm

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
images = sorted(
    [
        sorted(
            glob.glob(f"{args.input_folder}/rp{i}*"),
            key=lambda x: int(x.split("/")[-1].split("_")[2]),
        )
        for i in range(cam_cnt)
    ]
)
distances = sorted(
    [sorted(glob.glob(f"{args.input_folder}/distance_rp{i}*")[: len(images[0])]) for i in range(cam_cnt)]
)
instances = sorted(
    [
        sorted(
            glob.glob(f"{args.input_folder}/instance_rp{i}*.npy")[: len(images[0])],
            key=lambda x: int(x.split("/")[-1].split("_")[3].split(".")[0]),
        )
        for i in range(cam_cnt)
    ]
)
instances_meta = sorted(
    glob.glob(f"{args.input_folder}/instance_step*.json"),
    key=lambda x: int(x.split("/")[-1].split("_")[-1].split(".")[0]),
)


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

    for i in range(len(cameras)):
        cam_id = i
        cam = cameras[i]
        position = np.array(cameras[i]["position"], dtype=np.float)
        rotvec = np.array(cam["rotation"])
        # rotvec[0] = 180
        # the following line comes close
        # position[0] *= -1
        # position[1] *= 0.73

        print(position)
        # position[2] = 0=
        # position[1] *= 0.78
        # rotvec[1] = 90 - rotvec[1]
        print(rotvec)
        fy = cam["focal_y"]
        fx = cam["focal_y"]
        cy = cam["center_x"]
        cx = cam["center_y"]
        near = cam["near"]
        far = cam["far"]

        intrinsic = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        extrinsic = np.eye(4)
        rotvec[1] += 180
        extrinsic[:3, :3] = R.from_rotvec(rotvec, degrees=True).as_matrix()
        # extrinsic[:, :3] = extrinsic[:, [2, 1, 0]]  # Swap axis

        # position = position[[1, 0, 2]]
        # extrinsic[:, 2] *= -1
        # extrinsic[:3, :3] = extrinsic[:3, :3] @ R.from_rotvec([np.pi / 2, 0, 0]).as_matrix()
        # extrinsic[:3, 2] *= -1
        tf_mat = np.eye(4)
        tf_mat[:3, 3] = np.array(position)
        extrinsic = extrinsic @ tf_mat
        # extrinsic[:3, 3] = position
        out[str(cam_id)] = {"extrinsic": extrinsic.tolist(), "intrinsic": intrinsic, "near": near, "far": far}
    # path = "/Users/pietertolsma/Thesis/paper_code/MVSTER/data/ToteMVS/cams.json"
    path = f"{args.output_folder}/cams.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=4)


metas = defaultdict(dict)


def copy_files():
    for i in tqdm.tqdm(range(len(images))):
        for j in range(len(images[i])):
            # dfile = distances[i][j]
            instancefile = instances[i][j]
            imfile = images[i][j]

            cam_index = int(imfile.split("/")[-1].split("_")[0][2:])
            tote_index = imfile.split("/")[-1].split("_")[2]  # .split(".")[0]
            d_min = int(imfile.split("/")[-1].split("_")[3])
            d_max = int(imfile.split("/")[-1].split("_")[4].split(".")[0])

            image = np.array(Image.open(imfile))
            d = image[:, :, 3]

            # d_min, d_max = d.min(), d.max()

            metas[f"tote{tote_index}"][str(cam_index)] = {"d_min": str(d_min), "d_max": str(d_max)}

            # TODO: Remove this scaling once Isaac does this.
            # d = 255 * ((d - d_min) / (d_max - d_min))

            # image[:, :, 3] = d
            # d_im = Image.fromarray(d.astype(np.uint8), mode="L")

            # rgba_img = np.concatenate((image[:, :, :3], d[:, :, None]), axis=2).astype(np.uint8)

            # Image.fromarray(d.astype(np.uint8), "L").save(
            #     f"{args.output_folder}/tote{tote_index}/depth/{cam_index}.png"
            # )

            # d_im.save(f"{args.output_folder}/tote{tote_index}/depth/{cam_index}.png")
            # shutil.copy(dfile, f"{args.output_folder}/tote{tote_index}/depth/{cam_index}.npy")

            print(f"{imfile} {instancefile} {cam_index}")
            if i == 0:
                metas[f"tote{tote_index}"]["labels"] = json.load(open(instances_meta[j], "r"))
            shutil.copy(instancefile, f"{args.output_folder}/tote{tote_index}/instance/{cam_index}.npy")
            shutil.copy(imfile, f"{args.output_folder}/tote{tote_index}/{args.material}/{cam_index}.png")

    for tote in metas.keys():
        content = metas[tote]
        json.dump(content, open(f"{args.output_folder}/{tote}/meta.json", "w"), indent=4)


create_folders()
create_cams()
copy_files()
