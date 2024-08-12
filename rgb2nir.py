from typing import Callable
import numpy as np
import cv2
import os

import tqdm
from dataload.entry import Entry

from dataload.read_flyingthings3d import Flying3dDataset
from dataload.read_flow3d_flyingthings3d import Flow3dFlying3dDataset
import torch


from renderer2 import Renderer2, RenderInput, PinholeProjection, RenderOption
from datautil.image_horizontal_concat import image_horizontal_concat
from datautil.visualizer import TensorDictVisualizer


class RGB2NIR:
    def __init__(self):
        self.renderer = Renderer2(
            RenderOption(
                light_alpha=0.5,
                ambient_alpha=0.05,
                reflectance_alpha=0.4,
                parallel_normal=True,
                binete_ratio=0.8,
            )
        )

    def color_transform(self, rgb: torch.Tensor, isRgb: bool = False):
        # Reverse the channels and use torch.maximum
        rgb = rgb.float() / rgb.max()
        if isRgb:
            rgb = rgb[..., [2, 1, 0]]
        interm = torch.maximum(rgb, 1 - rgb)

        # Compute the weighted sum and apply the power operation
        nir = (
            interm[..., 0] * 0.114 + interm[..., 1] * 0.587 + interm[..., 2] * 0.229
        ) ** (1 / 0.25)

        return nir

    def process_data(self, entry: Entry):
        left_occ = (
            self.compute_disp_occ(*entry.disp_pair)[0]
            if entry.disp_pair[1] is not None
            else None
        )
        right_occ = (
            self.compute_disp_occ(entry.disp_pair[1], entry.disp_pair[0], direction=-1)[
                0
            ]
            if entry.disp_pair[1] is not None
            else None
        )

        left_nir = self.color_transform(entry.rgb_pair[0], isRgb=True)
        right_nir = self.color_transform(entry.rgb_pair[1], isRgb=True)
        calibration = entry.calibration
        image_left, metric_left = self.renderer.render_image(
            RenderInput(
                left_nir,
                entry.disp_pair[0],
                left_occ,
                entry.material_index_pair[0],
                calibration.left.intrinsic,
                calibration.baseline,
                PinholeProjection(
                    torch.tensor(
                        [calibration.baseline / 2, calibration.baseline / 2, 0]
                    )
                ),
            )
        )
        if entry.disp_pair[1] is None:
            image_right = None
            metric_right = None
        else:
            image_right, metric_right = self.renderer.render_image(
                RenderInput(
                    right_nir,
                    entry.disp_pair[1],
                    right_occ,
                    entry.material_index_pair[1],
                    calibration.right.intrinsic,
                    calibration.baseline,
                    PinholeProjection(
                        torch.tensor(
                            [-calibration.baseline / 2, calibration.baseline / 2, 0]
                        )
                    ),
                )
            )
        return (left_nir, image_left, metric_left), (
            right_nir,
            image_right,
            metric_right,
        )

    def render_image_and_store(
        self,
        entry: Entry,
        callback: Callable[
            [str, dict[str, torch.Tensor], dict[str, torch.Tensor]], None
        ],
    ):
        (left_nir, image_left, metric_left), (right_nir, image_right, metric_right) = (
            self.process_data(entry)
        )

        image_left = (image_left.cpu().numpy() * 255).astype(np.uint8)
        image_right = (image_right.cpu().numpy() * 255).astype(np.uint8)
        left_nir = (left_nir.cpu().numpy() * 255).astype(np.uint8)
        right_nir = (right_nir.cpu().numpy() * 255).astype(np.uint8)
        metric_left["nir_ambient"] = left_nir
        metric_right["nir_ambient"] = right_nir
        metric_left["nir_rendered"] = image_left
        metric_right["nir_rendered"] = image_right

        callback(entry.frame_id, metric_left, metric_right)

    def compute_disp_occ(self, left_disp, right_disp, direction=1, max_disp_error=0.5):
        left_disp = torch.tensor(left_disp).unsqueeze(0).cuda()
        right_disp = torch.tensor(right_disp).unsqueeze(0).cuda()
        # 오클루전 맵 초기화 (모든 값을 0으로 설정)
        # 왼쪽 시차 맵의 shape 가져오기
        batch_size, height, width = left_disp.shape

        # 오클루전 맵 초기화 (모든 값을 0으로 설정)
        occ_map = torch.zeros_like(left_disp)

        # x 좌표 생성 (크기는 [height, width])
        x_coords = (
            torch.arange(width)
            .view(1, 1, -1)
            .expand(batch_size, height, -1)
            .to(left_disp.device)
        )

        # right_x 계산: x 좌표에서 왼쪽 시차 값을 뺀다
        right_x_coords = x_coords - left_disp * direction

        # 유효 범위(0 <= right_x < width)에 있는지 체크
        valid_x = (right_x_coords >= 0) & (right_x_coords < width)

        # 오른쪽 시차 맵에서 valid_x에 해당하는 값 가져오기
        right_x_coords_clamped = right_x_coords.clamp(0, width - 1).long()
        right_d = torch.gather(right_disp, 2, right_x_coords_clamped)

        # 시차값 비교 (오차 허용치 사용)
        disparity_diff = torch.abs(left_disp - right_d)
        match = disparity_diff < max_disp_error

        # valid_x와 match가 모두 true인 경우에만 occ_map을 1로 설정
        occ_map = valid_x & match

        return occ_map.float().cpu().numpy()  # boolean 값을 float으로 변환하여 반환

    def process_dataset(self, dataset: Flying3dDataset):
        for entry in tqdm.tqdm(dataset):
            self.render_image_and_store(entry, dataset.store_generated_item)


if __name__ == "__main__":
    # Load the image
    # dataset = Flying3dDataset(root="/bean/flow3d", search_paths=False)
    dataset = Flow3dFlying3dDataset(root="/bean/flyingthings3d", search_paths=False)
    # Create an instance of the class
    rgb2nir = RGB2NIR()

    # Transform the image
    #

    rgb2nir.process_dataset(dataset)
    # input = dataset[0]
    # (nir, left, _), (nir2, right, _) = rgb2nir.process_data(input)
    # cv2.imshow("Left", left.cpu().numpy())
    # cv2.imshow("Right", right.cpu().numpy())

    # Display the image
    # cv2.imshow("RGB Image", image_left)
    # cv2.imshow("NIR Image", nir)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
