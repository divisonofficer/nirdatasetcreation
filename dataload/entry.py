from typing import Tuple, Dict, Optional
import torch


class Entry:
    def __init__(
        self,
        rgb_pair: Tuple[torch.Tensor, torch.Tensor],
        disp_pair: Tuple[torch.Tensor, Optional[torch.Tensor]],
        calibration: Dict[str, torch.Tensor],
        material_index_pair: Optional[
            Tuple[torch.Tensor, Optional[torch.Tensor]]
        ] = None,
    ):
        self.rgb_pair = rgb_pair
        self.disp_pair = disp_pair
        self.calibration = calibration
        self.material_index_pair = material_index_pair
