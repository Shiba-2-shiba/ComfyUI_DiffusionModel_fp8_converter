import torch
from tqdm.auto import tqdm

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

    def convert_to_fp8(self, model, clip):
        try:
            # モデルとクリップをFP8に変換
            model_fp8 = self.convert_module_to_fp8(model)
            clip_fp8 = self.convert_module_to_fp8(clip)
            return model_fp8, clip_fp8
        except Exception as e:
            print(f"FP8変換中にエラーが発生しました: {str(e)}")
            return model, clip

    def convert_module_to_fp8(self, module):
        # モジュール内のすべてのパラメータとバッファをFP8形式に変換
        for name, param in tqdm(module.named_parameters(), desc="モデルパラメータをFP8に変換中"):
            if param.dtype == torch.float16:
                with torch.no_grad():
                    param.copy_(param.to(dtype=torch.float8_e4m3fn))

        for name, buffer in tqdm(module.named_buffers(), desc="モデルバッファをFP8に変換中"):
            if buffer.dtype == torch.float16:
                with torch.no_grad():
                    buffer.copy_(buffer.to(dtype=torch.float8_e4m3fn))

        return module
