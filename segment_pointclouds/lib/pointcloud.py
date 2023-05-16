import json
import numpy as np
import cv2
from plyfile import PlyData, PlyElement
import os

# scan = data.scans[0]
# print(scan)


def get_material_labels(meta_dict: dict, material: str) -> list:
    res = []
    labelsToId = {v: int(k) for k, v in meta_dict["mappings"]["idToLabels"].items()}

    all_labels = set(labelsToId.values())
    if material == "all":
        for mat in ["specular", "clear", "diffuse"]:
            all_labels -= set(get_material_labels(meta_dict, mat))
        return list(all_labels)

    for key, value in meta_dict.items():
        if not key.startswith("/MyScope/"):
            continue
        if value["material"] == material:
            if key in labelsToId:
                res.append(int(labelsToId[key]))
    return res


def filter_mask(segmap: np.array, ids: list):
    res = np.isin(segmap, ids).astype(np.uint8)
    return res


# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(
        np.linalg.inv(intrinsics_ref), np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1])
    )
    # source 3D space
    xyz_src = np.matmul(
        np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)), np.vstack((xyz_ref, np.ones_like(x_ref)))
    )[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(
        np.linalg.inv(intrinsics_src), np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1])
    )
    # reference 3D space
    xyz_reprojected = np.matmul(
        np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)), np.vstack((xyz_src, np.ones_like(x_ref)))
    )[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / (K_xyz_reprojected[2:3] + 1e-10)
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(
        depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src
    )
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask = np.logical_and(dist < 4, relative_depth_diff < 0.05)
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected, x2d_src, y2d_src


def filter_depth(cfg, data, startIndex):
    scan = data.scans[startIndex // 6]
    meta = data.get_scan_labels(scan)
    # print(data.scans)

    ref_views = np.arange(0, 6)
    all_src_views = [np.delete(ref_views, i, None) for i in ref_views]
    pair_data = []
    for ref, src_views in zip(ref_views, all_src_views):
        pair_data.append((ref, list(src_views)))

    imgs = data[startIndex]["imgs"]

    material_colors = {"all": [0, 0, 0], "diffuse": [1.0, 0, 0], "clear": [0, 1.0, 0], "specular": [0, 0, 1.0]}

    vertexs = []
    vertex_colors = []

    W, H = imgs[0].shape[2], imgs[0].shape[1]
    for ref_view, src_views in pair_data:
        segmap = data.get_segmentation(scan, ref_view)

        proj = data[ref_view + startIndex]["proj_matrices"]["stage4"]

        image = imgs[ref_view]
        intr, extr = (
            proj[0][1, :3, :3],
            proj[0][0, :, :],
        )
        zz = data[ref_view + startIndex]["depth"]["stage4"]

        geo_mask_sum = 0
        all_srcview_depth_ests = []
        for i, src in enumerate(src_views):
            src_proj = data[src + startIndex]["proj_matrices"]["stage4"]
            src_intr, src_extr = (
                src_proj[0][1, :3, :3],
                src_proj[0][0, :, :],
            )

            src_zz = data[src + startIndex]["depth"]["stage4"]

            geo_mask, depth_repr, x2d_src, y2d_src = check_geometric_consistency(
                zz, intr, extr, src_zz, src_intr, src_extr
            )

            geo_mask_sum += geo_mask.astype(np.int32)
            all_srcview_depth_ests.append(depth_repr * geo_mask)

        zz_avg = (sum(all_srcview_depth_ests) + zz) / (geo_mask_sum + 1)
        geo_mask = geo_mask_sum >= cfg.data.thres_view

        for material in ["all", "clear", "diffuse", "specular"]:
            diff_ids = get_material_labels(meta, material)
            mask = filter_mask(segmap, diff_ids).astype(bool)
            color = np.matrix(material_colors[material]).T
            image[:, mask] = color

        # print(zz.shape)
        # y_idx, x_idx = np.where(mask)
        xx, yy = np.meshgrid(np.arange(0, H), np.arange(0, W))

        xx = xx[geo_mask]
        yy = yy[geo_mask]
        zz = zz_avg[geo_mask]

        xx, yy, zz = xx.flatten(), yy.flatten(), zz.flatten()

        # print(mask.max())
        xyz_ref = np.matmul(np.linalg.inv(intr), np.vstack((xx, yy, np.ones_like(xx)) * zz))

        xyz_world = np.matmul(np.linalg.inv(extr), np.vstack((xyz_ref, np.ones_like(xx))))[:3]

        color = image[:, geo_mask].transpose((1, 0))

        vertexs.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color * 255).astype(np.uint8))

    # print(xyz_ref.shape)
    vertexs = np.concatenate(vertexs, axis=0)

    vertex_colors = np.concatenate(vertex_colors, axis=0)
    vertexs = np.array([tuple(v) for v in vertexs], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[("red", "u1"), ("green", "u1"), ("blue", "u1")])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    plyfilename = f"{scan}.ply"
    pth = f"./output/{plyfilename}"
    if not os.path.exists("./output"):
        os.makedirs("./output", exist_ok=True)
    el = PlyElement.describe(vertex_all, "vertex")
    PlyData([el]).write(pth)
    print("saving the final model to", plyfilename)
