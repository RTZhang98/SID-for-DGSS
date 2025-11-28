import os
from PIL import Image
import numpy as np
import torchvision.transforms as T
from mmseg.models.builder import BACKBONES, MODELS
from .reins import Reins
from .dino_v2 import DinoVisionTransformer
from .utils import set_requires_grad, set_train

# 保存张量为图片的函数
def save_tensor_as_images(x, output_dir="output_images"):
    """
    将张量 `x` 保存为图片文件。
    假设输入的 `x` 形状为 (batch_size, channels, height, width)，其中 channels=3。
    """
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # 创建输出文件夹

    # 将张量转为 NumPy 格式，方便保存为图片
    x = x.detach().cpu().numpy()  # 转为 NumPy 数组
    batch_size, channels, height, width = x.shape

    for i in range(batch_size):
        img = x[i]  # 取出第 i 张图片: 形状为 (3, height, width)

        # 如果 channels=3，转为 (height, width, channels) 格式
        if channels == 3:
            img = img.transpose(1, 2, 0)  # 通道维度移动到最后

        # 标准化到 [0, 255]，以便保存为图像
        img_min, img_max = img.min(), img.max()
        img = (img - img_min) / (img_max - img_min)  # 归一化到 [0, 1]
        img = (img * 255).astype(np.uint8)  # 转为 [0, 255] 的 uint8 格式

        # 创建 PIL 图像对象
        image = Image.fromarray(img)

        # 保存图片
        image_path = os.path.join(output_dir, f"image_{i}.png")
        image.save(image_path)
        print(f"Saved: {image_path}")

@BACKBONES.register_module()
class ReinsDinoVisionTransformer(DinoVisionTransformer):
    def __init__(
        self,
        reins_config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reins: Reins = MODELS.build(reins_config)

    def forward_features(self, x, masks=None):
        if x.shape[1] != 3:
            x=x.view(-1, 3, x.shape[-2], x.shape[-1])
        #save_tensor_as_images(x)
        
        B, _, h, w = x.shape
        H, W = h // self.patch_size, w // self.patch_size
        x = self.prepare_tokens_with_masks(x, masks)
        outs = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.reins.forward(
                x,
                idx,
                batch_first=True,
                has_cls_token=True,
            )
            if idx in self.out_indices:
                outs.append(
                    x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
                )
        return self.reins.return_auto(outs)

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["reins"])
        set_train(self, ["reins"])

    def state_dict(self, destination, prefix, keep_vars):
        state = super().state_dict(destination, prefix, keep_vars)
        keys = [k for k in state.keys() if "rein" not in k]
        for key in keys:
            state.pop(key)
            if key in destination:
                destination.pop(key)
        return state
