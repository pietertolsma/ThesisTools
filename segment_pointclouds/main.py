import json
import numpy as np
import cv2
from omegaconf import OmegaConf
from lib.totemvs import MVSDataset
from plyfile import PlyData, PlyElement
import tqdm

from lib.pointcloud import filter_depth


if __name__ == "__main__":
    cfg = OmegaConf.load("config.yaml")
    data = MVSDataset(cfg, "test")

    for i in tqdm.tqdm(range(len(data) // cfg.data.nviews)):
        filter_depth(cfg, data, i * cfg.data.nviews)
