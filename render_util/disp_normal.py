import pandas as pd
import numpy as np
import torch
from pyntcloud import PyntCloud


class DispNormal:
    def __init__(self):
        pass

    def disp_to_pointcloud(
        self, disp: torch.Tensor, intrinsic: torch.Tensor, baseline: float
    ):
        depth_map = self.disp_to_depth(disp, intrinsic, baseline)
        # 이미지의 높이와 너비
        h, w = depth_map.shape

        # 카메라 내부 매트릭스에서 파라미터 추출
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]

        # 2D 그리드 생성 (높이와 너비에 맞게 생성)
        # torch.meshgrid는 기본적으로 (열, 행) 순서로 그리드를 생성합니다.
        u, v = torch.meshgrid(
            torch.arange(w, dtype=torch.float32), torch.arange(h, dtype=torch.float32)
        )
        u = u.t()
        v = v.t()
        # 깊이 값으로 3D 좌표 계산
        x = (u - cx) * depth_map / fx
        y = (v - cy) * depth_map / fy
        z = depth_map

        # 3D 점군 생성
        pointcloud = torch.stack([x, y, z], dim=-1)  # shape: (H, W, 3)

        return pointcloud

    def disp_to_depth(
        self, disp: torch.Tensor, intrinsic: torch.Tensor, baseline: float
    ):
        # Compute the depth map
        depth = baseline * intrinsic[0, 0] / disp
        return depth

    def pointcloud_to_normal(self, XYZ, neighbour_n=20):
        N, M, _ = XYZ.shape

        # Flatten the XYZ data
        x_flat = XYZ[..., 0].flatten()
        y_flat = XYZ[..., 1].flatten()
        z_flat = XYZ[..., 2].flatten()

        # Create a DataFrame and handle NaNs and infs
        points = pd.DataFrame(
            {
                "x": x_flat,
                "y": y_flat,
                "z": z_flat,
            }
        )

        # Create a mask for valid (non-NaN and non-inf) values
        valid_mask = ~points[["x", "y", "z"]].isna().any(axis=1) & np.isfinite(
            points[["x", "y", "z"]]
        ).all(axis=1)

        if valid_mask.sum() == 0:
            # Handle case where there are no valid points
            return torch.tensor(np.empty((0, 0, 3)), dtype=torch.float32)

        # Apply the mask
        valid_points = points[valid_mask]

        # Create a PyntCloud object with valid points
        cloud = PyntCloud(valid_points)

        k_neighbors = cloud.get_neighbors(k=neighbour_n)
        cloud.add_scalar_field("normals", k_neighbors=k_neighbors)

        # print(cloud.points.describe())

        # Extract normals and reshape
        nx = cloud.points["nx(%d)" % (neighbour_n + 1)].to_numpy()
        ny = cloud.points["ny(%d)" % (neighbour_n + 1)].to_numpy()
        nz = cloud.points["nz(%d)" % (neighbour_n + 1)].to_numpy()

        # Reshape to the original shape and apply the valid mask
        nx_full = np.full((N * M,), np.nan)
        ny_full = np.full((N * M,), np.nan)
        nz_full = np.full((N * M,), np.nan)

        nx_full[valid_mask] = -nx
        ny_full[valid_mask] = ny
        nz_full[valid_mask] = -nz

        nx = nx_full.reshape((N, M, 1))
        ny = ny_full.reshape((N, M, 1))
        nz = nz_full.reshape((N, M, 1))

        # Adjust normals based on the valid mask
        invalid = np.isnan(nz) | np.isnan(nx) | np.isnan(ny)
        nx[invalid] = 0
        ny[invalid] = 0
        nz[invalid] = 0

        normals = np.concatenate((nx, ny, nz), axis=2)
        return torch.tensor(normals, dtype=torch.float32)

    def disp_to_normal(
        self,
        disp: torch.Tensor,
        intrinsic: torch.Tensor,
        baseline: float,
        neighbour_n=10,
    ):
        # Convert disparity to point cloud
        pointcloud = self.disp_to_pointcloud(disp, intrinsic, baseline)
        # Convert point cloud to normal
        normals = self.pointcloud_to_normal(pointcloud, neighbour_n=neighbour_n)
        return normals
