from turtle import left
import torch

import os


import cv2
import numpy as np
import dataload.imread as imread


from torch.utils.data import Dataset
from dataload.entry import Entry

Intrinsic1 = torch.tensor([[1050.0, 0.0, 479.5], [0.0, 1050.0, 269.5], [0.0, 0.0, 1.0]])
Intrinsic2 = torch.tensor([[450.0, 0.0, 479.5], [0.0, 450.0, 269.5], [0.0, 0.0, 1.0]])


class Flying3dDataset(Dataset):
    def __init__(self, scale=1.0):
        folders = []
        self.scale = scale
        for data in os.listdir("data/Flying3d"):
            folders.append(os.path.join("data/Flying3d", data))
        self.folders = folders

        self.entries = []
        for folder in self.folders:
            self.entries += self.read_FlyingThings3d_folder(
                folder, Intrinsic2 if "Driving" in folder else Intrinsic1
            )

    def __len__(self):
        return len(self.entries)

    def read_FlyingThings3d_folder(self, folder_path, intrinsic):
        files = [
            x.split(".")[0] for x in os.listdir(os.path.join(folder_path, "disparity"))
        ]
        camera_calib = self.read_calib_file(
            os.path.join(folder_path, "camera_data.txt")
        )

        entriy_path = []
        for file in files:
            rgb_left_path = os.path.join(
                folder_path, "RGB_cleanpass", "left", f"{file}.png"
            )
            rgb_right_path = os.path.join(
                folder_path, "RGB_cleanpass", "right", f"{file}.png"
            )
            if os.path.exists(os.path.join(folder_path, "disparity", "left")):
                disparity_left_path = os.path.join(
                    folder_path, "disparity", "left", f"{file}.pfm"
                )
                disparity_right_path = os.path.join(
                    folder_path, "disparity", "right", f"{file}.pfm"
                )
            else:
                disparity_left_path = os.path.join(
                    folder_path, "disparity", f"{file}.pfm"
                )
                disparity_right_path = None
            if os.path.exists(os.path.join(folder_path, "material_index", "left")):
                material_left_path = os.path.join(
                    folder_path, "material_index", "left", f"{file}.pfm"
                )
                material_right_path = os.path.join(
                    folder_path, "material_index", "right", f"{file}.pfm"
                )
            else:
                material_left_path = os.path.join(
                    folder_path, "material_index", f"{file}.pfm"
                )
                material_right_path = None
            calibration = camera_calib[int(file)]
            calibration["L"]["intrinsic"] = intrinsic
            calibration["R"]["intrinsic"] = intrinsic
            entriy_path.append(
                (
                    (rgb_left_path, rgb_right_path),
                    (disparity_left_path, disparity_right_path),
                    (material_left_path, material_right_path),
                    calibration,
                )
            )
        return entriy_path

    def read_calib_file(self, path):
        with open(path, "r") as file:
            lines = file.readlines()

        data = {}
        frame_id = None

        for line in lines:
            parts = line.strip().split()
            if len(parts) == 0:
                continue
            if parts[0] == "Frame":
                frame_id = int(parts[1])
                data[frame_id] = {"L": None, "R": None, "baseline": None}
            elif parts[0] in ["L", "R"]:
                camera = parts[0]
                extrinsic = torch.tensor([float(x) for x in parts[1:13]]).view(3, 4)
                data[frame_id][camera] = {"extrinsic": extrinsic}
                if data[frame_id]["L"] is not None and data[frame_id]["R"] is not None:
                    baseline = torch.norm(
                        data[frame_id]["L"]["extrinsic"][:, 3]
                        - data[frame_id]["R"]["extrinsic"][:, 3]
                    )
                    data[frame_id]["baseline"] = baseline

        return data

    def scale_image(self, image, scale=0, scale_value=False):
        if scale == 0:
            scale = self.scale
        if scale == 1:
            return torch.tensor(image).float()
        return torch.tensor(
            cv2.resize(
                image,
                (int(image.shape[1] * scale), int(image.shape[0] * scale)),
                interpolation=cv2.INTER_NEAREST,
            )
            * (scale if scale_value else 1)
        ).float()

    def __getitem__(self, idx):
        (
            (left_path, right_path),
            (left_dis_path, right_disp_path),
            (left_material_path, right_material_path),
            calibration,
        ) = self.entries[idx]
        left_rgb = torch.Tensor(imread.read(left_path))
        right_rgb = torch.Tensor(imread.read(right_path))
        left_dis = torch.Tensor(imread.read(left_dis_path).copy())
        right_dis = (
            torch.Tensor(imread.read(right_disp_path).copy())
            if right_disp_path
            else None
        )
        left_material = torch.Tensor(imread.read(left_material_path).copy())
        right_material = (
            torch.Tensor(imread.read(right_material_path).copy())
            if right_material_path
            else None
        )
        return Entry(
            (left_rgb, right_rgb),
            (left_dis, right_dis),
            calibration,
            (left_material, right_material),
        )
