import torch
from safetensors import safe_open, save_file
from tqdm.auto import tqdm
import torch.nn as nn

class FP8ConverterNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP",)
    FUNCTION = "convert_to_fp8"
    CATEGORY = "Model Processing"

    def convert_to_fp8(self, model: nn.Module, clip: nn.Module):
        try:
            # モデルとクリップをFP8に変換
            model_fp8 = self.convert_module_to_fp8(model)
            clip_fp8 = self.convert_module_to_fp8(clip)
            return model_fp8, clip_fp8
        except Exception as e:
            print(f"FP8変換中にエラーが発生しました: {str(e)}")
            return model, clip  # エラー発生時は元のモデルとクリップを返す

    def convert_module_to_fp8(self, module: nn.Module):
        # モジュールのすべてのパラメータとバッファをFP8形式に変換
        for name, param in module.named_parameters():
            module.register_parameter(name, nn.Parameter(param.to(dtype=torch.float8_e4m3fn)))

        for name, buffer in module.named_buffers():
            module.register_buffer(name, buffer.to(dtype=torch.float8_e4m3fn))

        return module

