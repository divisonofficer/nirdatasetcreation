import torch

import os

from typing import List, Tuple, Optional, Union

import cv2
import numpy as np
import dataload.imread as imread


from torch.utils.data import Dataset
from dataload.entry import Entry, StereoCalibration, CameraCalibration

Intrinsic1 = torch.tensor([[1050.0, 0.0, 479.5], [0.0, 1050.0, 269.5], [0.0, 0.0, 1.0]])
Intrinsic2 = torch.tensor([[450.0, 0.0, 479.5], [0.0, 450.0, 269.5], [0.0, 0.0, 1.0]])


PATH1 = ["TRAIN", "TEST"]
PATH2 = ["A", "B", "C"]
PATH3 = []


RGB_FRAME = "frames_cleanpass"
JSON_FILENAME = "Flow3dFlyingThings3d.json"

import json


class Flow3dFlying3dDataset(Dataset):
    def __init__(self, root: str, scale=1.0, search_paths=True):

        self.root = root
        self.scale = scale

        if search_paths:
            self.search_paths(root)
        else:
            self.read_json()

    def read_json(self):
        with open(JSON_FILENAME, "r") as file:
            entries = json.load(file)
        self.entries = []
        for entry in entries:
            nir_path = entry["rgb"].replace(RGB_FRAME, "nir_rendered")
            nir_ambient_path = entry["rgb"].replace(RGB_FRAME, "nir_ambient")
            if os.path.exists(nir_path) and os.path.exists(nir_ambient_path):
                continue
            self.entries.append(
                (
                    entry["rgb"],
                    entry["disparity"],
                    entry["material_index"],
                    StereoCalibration(
                        CameraCalibration(
                            torch.tensor(entry["calibration"]["left"]["intrinsic"]),
                            torch.tensor(entry["calibration"]["left"]["extrinsic"]),
                        ),
                        CameraCalibration(
                            torch.tensor(entry["calibration"]["right"]["intrinsic"]),
                            torch.tensor(entry["calibration"]["right"]["extrinsic"]),
                        ),
                        torch.tensor(entry["calibration"]["baseline"]),
                    ),
                )
            )

    def search_paths(self, root):
        """
        Driving
        """
        folders = []

        for path1 in PATH1:
            for path2 in PATH2:
                for folder in os.listdir(os.path.join(root, RGB_FRAME, path1, path2)):
                    folders.append(os.path.join(path1, path2, folder))

        self.entries: List[
            Tuple[Tuple[str, str], Tuple[str, str], Tuple[str, str], StereoCalibration]
        ] = []
        for folder in folders:
            self.entries += self.read_FlyingThings3d_folder(
                root,
                folder,
                Intrinsic2 if "15mm" in folder else Intrinsic1,
            )

        self.dumps_entries()

    def dumps_entries(self):
        entries = []
        for entry in self.entries:
            entries.append(
                {
                    "rgb": entry[0],
                    "disparity": entry[1],
                    "material_index": entry[2],
                    "calibration": entry[3].__dict__(),
                }
            )
        with open(JSON_FILENAME, "w") as file:
            json.dump(entries, file)

    def __len__(self):
        return len(self.entries)

    def read_FlyingThings3d_folder(self, root, folder_path, intrinsic):
        files = [
            x.split(".")[0]
            for x in os.listdir(os.path.join(root, RGB_FRAME, folder_path, "left"))
            if x.endswith(".png")
        ]
        camera_calib = self.read_calib_file(
            os.path.join(root, "camera_data", folder_path, "camera_data.txt"), intrinsic
        )

        entriy_path = []
        for file in files:
            rgb_left_path = os.path.join(
                root, RGB_FRAME, folder_path, "left", f"{file}.png"
            )
            rgb_right_path = os.path.join(
                root, RGB_FRAME, folder_path, "right", f"{file}.png"
            )
            if os.path.exists(os.path.join(root, "disparity", folder_path, "left")):
                disparity_left_path = os.path.join(
                    root, "disparity", folder_path, "left", f"{file}.pfm"
                )
                disparity_right_path = os.path.join(
                    root, "disparity", folder_path, "right", f"{file}.pfm"
                )
            else:
                disparity_left_path = os.path.join(
                    root, "disparity", folder_path, f"{file}.pfm"
                )
                disparity_right_path = None
            if os.path.exists(
                os.path.join(root, "material_index", folder_path, "left")
            ):
                material_left_path = os.path.join(
                    root, "material_index", folder_path, "left", f"{file}.pfm"
                )
                material_right_path = os.path.join(
                    root, "material_index", folder_path, "right", f"{file}.pfm"
                )
            else:
                material_left_path = os.path.join(
                    root, "material_index", folder_path, f"{file}.pfm"
                )
                material_right_path = None
            if not int(file) in camera_calib:
                continue
            calibration = camera_calib[int(file)]
            entriy_path.append(
                (
                    (rgb_left_path, rgb_right_path),
                    (disparity_left_path, disparity_right_path),
                    (material_left_path, material_right_path),
                    calibration,
                )
            )
        return entriy_path

    def read_calib_file(self, path, intrinsic):
        with open(path, "r") as file:
            lines = file.readlines()

        data: dict[int, StereoCalibration] = {}
        frame_id = -1
        extrinsics: dict[str, Optional[torch.Tensor]] = {"L": None, "R": None}
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 0:
                continue
            if parts[0] == "Frame":
                frame_id = int(parts[1])
                extrinsics = {"L": None, "R": None}
            elif parts[0] in ["L", "R"]:
                camera = parts[0]
                extrinsic = torch.tensor([float(x) for x in parts[1:13]]).view(3, 4)
                extrinsics[camera] = extrinsic
                if extrinsics["L"] is not None and extrinsics["R"] is not None:
                    baseline = torch.norm(extrinsics["L"][:, 3] - extrinsics["R"][:, 3])
                    data[frame_id] = StereoCalibration(
                        CameraCalibration(intrinsic, extrinsics["L"]),
                        CameraCalibration(intrinsic, extrinsics["R"]),
                        baseline,
                    )

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
            left_path,
            (left_rgb, right_rgb),
            (left_dis, right_dis),
            calibration,
            (left_material, right_material),
        )

    def store_generated_item(
        self,
        frame_id: str,
        items_left: dict[str, Union[torch.Tensor, np.ndarray]],
        items_right: dict[str, Union[torch.Tensor, np.ndarray]],
    ):
        folder_path = frame_id.split(f"{RGB_FRAME}/")[-1].split("/left")[0]
        frame_number = frame_id.split("/")[-1].split(".")[0]

        for side, items in zip(["left", "right"], [items_left, items_right]):
            for item_name, item in items.items():
                item_folder = os.path.join(self.root, item_name, folder_path, side)
                if isinstance(item, torch.Tensor):
                    item = item.cpu().numpy()
                    type = "pfm"
                else:
                    type = "png"

                item_path = os.path.join(item_folder, f"{frame_number}.{type}")
                if not os.path.exists(item_folder):
                    os.makedirs(item_folder)
                if type == "pfm":
                    imread.write(item_path, item)
                else:
                    cv2.imwrite(item_path, item)
