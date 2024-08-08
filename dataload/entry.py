from typing import Tuple, Dict, Optional
import torch


class CameraCalibration:
    def __init__(
        self,
        intrinsic: torch.Tensor,
        extrinsic: torch.Tensor,
    ):
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic

    def __dict__(self):
        return {
            "intrinsic": self.intrinsic.numpy().tolist(),
            "extrinsic": self.extrinsic.numpy().tolist(),
        }


class StereoCalibration:
    def __init__(
        self,
        left: CameraCalibration,
        right: CameraCalibration,
        baseline: torch.Tensor,
    ):
        self.left = left
        self.right = right
        self.baseline = baseline

    def __dict__(self):
        return {
            "left": self.left.__dict__(),
            "right": self.right.__dict__(),
            "baseline": self.baseline.numpy().tolist(),
        }


class Entry:
    def __init__(
        self,
        frame_id: str,
        rgb_pair: Tuple[torch.Tensor, torch.Tensor],
        disp_pair: Tuple[torch.Tensor, Optional[torch.Tensor]],
        calibration: StereoCalibration,
        material_index_pair: Optional[
            Tuple[torch.Tensor, Optional[torch.Tensor]]
        ] = None,
    ):
        self.frame_id = frame_id
        self.rgb_pair = rgb_pair
        self.disp_pair = disp_pair
        self.calibration = calibration
        self.material_index_pair = material_index_pair
