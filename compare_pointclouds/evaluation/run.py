# ----------------------------------------------------------------------------
# -                   TanksAndTemples Website Toolbox                        -
# -                    http://www.tanksandtemples.org                        -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2017
# Arno Knapitsch <arno.knapitsch@gmail.com >
# Jaesik Park <syncle@gmail.com>
# Qian-Yi Zhou <Qianyi.Zhou@gmail.com>
# Vladlen Koltun <vkoltun@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# ----------------------------------------------------------------------------
#
# This python script is for downloading dataset from www.tanksandtemples.org
# The dataset has a different license, please refer to
# https://tanksandtemples.org/license/

# this script requires Open3D python binding
# please follow the intructions in setup.py before running this script.
import numpy as np
import open3d as o3d
import os
import argparse

from config import scenes_tau_dict
from registration import (
    trajectory_alignment,
    registration_vol_ds,
    registration_unif,
    read_trajectory,
)
from evaluation import EvaluateHisto
from util import make_dir
from plot import plot_graph


def run_evaluation(scene, gt_ply_path, ply_path, out_dir):
    # scene = os.path.basename(os.path.normpath(dataset_dir))

    # if scene not in scenes_tau_dict:
    #     print(dataset_dir, scene)
    #     raise Exception("invalid dataset-dir, not in scenes_tau_dict")

    print("")
    print("===========================")
    print("Evaluating %s" % scene)
    print("===========================")

    dTau = 0.5  # scenes_tau_dict[scene]
    # put the crop-file, the GT file, the COLMAP SfM log file and
    # the alignment of the according scene in a folder of
    # the same scene name in the dataset_dir
    # colmap_ref_logfile = os.path.join(dataset_dir, scene + "_COLMAP_SfM.log")
    # alignment = os.path.join(dataset_dir, scene + "_trans.txt")
    # gt_filen = os.path.join(dataset_dir, scene + ".ply")
    # cropfile = os.path.join(dataset_dir, scene + ".json")
    # map_file = os.path.join(dataset_dir, scene + "_mapping_reference.txt")

    make_dir(out_dir)

    # Load reconstruction and according GT
    print(ply_path)
    pcd = o3d.io.read_point_cloud(ply_path)
    # print(gt_filen)
    gt_pcd = o3d.io.read_point_cloud(gt_ply_path)

    # gt_trans = np.loadtxt(alignment)
    # traj_to_register = read_trajectory(traj_path)
    # gt_traj_col = read_trajectory(colmap_ref_logfile)

    # trajectory_transform = trajectory_alignment(map_file, traj_to_register, gt_traj_col, gt_trans, scene)

    # Refine alignment by using the actual GT and MVS pointclouds
    # vol = o3d.visualization.read_selection_polygon_volume(cropfile)
    # big pointclouds will be downlsampled to this number to speed up alignment
    dist_threshold = dTau

    # Registration refinment in 3 iterations
    # r2 = registration_vol_ds(pcd, gt_pcd, trajectory_transform, vol, dTau, dTau * 80, 20)
    # r3 = registration_vol_ds(pcd, gt_pcd, r2.transformation, vol, dTau / 2.0, dTau * 20, 20)
    # r = registration_unif(pcd, gt_pcd, r3.transformation, vol, 2 * dTau, 20)
    for mat in ["diffuse", "clear", "specular"]:
        os.makedirs(f"{out_dir}/{scene}/{mat}", exist_ok=True)

        # Histogramms and P/R/F1
        plot_stretch = 5
        [
            precision,
            recall,
            fscore,
            edges_source,
            cum_source,
            edges_target,
            cum_target,
        ] = EvaluateHisto(
            mat,
            pcd,
            gt_pcd,
            dTau / 2.0,
            dTau,
            f"{out_dir}/{scene}/{mat}",
            plot_stretch,
            scene,
        )
        eva = [precision, recall, fscore]
        print("==============================")
        print("evaluation result : %s" % scene)
        print("==============================")
        print("distance tau : %.3f" % dTau)
        print("precision : %.4f" % eva[0])
        print("recall : %.4f" % eva[1])
        print("f-score : %.4f" % eva[2])
        print("==============================")

        # Plotting
        plot_graph(
            mat,
            scene,
            fscore,
            dist_threshold,
            edges_source,
            cum_source,
            edges_target,
            cum_target,
            plot_stretch,
            f"{out_dir}/{scene}/{mat}",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt-path",
        type=str,
        required=True,
        help="path to trajectory file. See `convert_to_logfile.py` to create this file.",
    )
    parser.add_argument(
        "--ply-path",
        type=str,
        required=True,
        help="path to reconstruction ply file",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="output directory, default: an evaluation directory is created in the directory of the ply file",
    )
    args = parser.parse_args()

    if args.out_dir.strip() == "":
        args.out_dir = os.path.join(os.path.dirname(args.ply_path), "evaluation")

    run_evaluation(
        scene="tote5",
        gt_ply_path=args.gt_path,
        ply_path=args.ply_path,
        out_dir=args.out_dir,
    )
