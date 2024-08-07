import numpy as np
import matplotlib.pyplot as plt


class TensorVisualizer:
    def __init__(self, tensor):
        self.tensor = tensor
        self.width = tensor.shape[0]
        self.height = tensor.shape[1]
        self.channels = tensor.shape[2] if len(tensor.shape) > 2 else 1

    def plot(self):
        # normalmap을 시각화하기 위해 -1에서 1의 범위를 0에서 1로 변환합니다.
        normalmap = self.tensor
        # 각 채널을 분리합니다.
        r_channel = normalmap[:, :, 0]
        g_channel = normalmap[:, :, 1]
        b_channel = normalmap[:, :, 2]

        # matplotlib를 사용하여 각 채널을 시각화합니다.
        fig, axes = plt.subplots(1, 3, figsize=(25, 10))

        # Red 채널
        cax_r = axes[0].imshow(r_channel, cmap="coolwarm", vmin=-1, vmax=1)
        axes[0].set_title("Red Channel")
        axes[0].axis("off")
        fig.colorbar(cax_r, ax=axes[0], orientation="vertical")

        # Green 채널
        cax_g = axes[1].imshow(g_channel, cmap="coolwarm", vmin=-1, vmax=1)
        axes[1].set_title("Green Channel")
        axes[1].axis("off")
        fig.colorbar(cax_g, ax=axes[1], orientation="vertical")

        # Blue 채널
        cax_b = axes[2].imshow(b_channel, cmap="coolwarm", vmin=-1, vmax=1)
        axes[2].set_title("Blue Channel")
        axes[2].axis("off")
        fig.colorbar(cax_b, ax=axes[2], orientation="vertical")

        plt.show()
