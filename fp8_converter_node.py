import torch
import os

# 必要な場所からModelPatcherをインポート
from comfy.model_patcher import ModelPatcher

class FP8ConverterNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "convert_to_fp8"

    CATEGORY = "conversion"

    def convert_to_fp8(self, model, clip):
        try:
            # モデルをFP8形式に変換し、元の形式に戻す
            if hasattr(model, 'diffusion_model'):
                model.diffusion_model = model.diffusion_model.to(torch.float8_e4m3fn)
                return (model, clip)
            elif isinstance(model, ModelPatcher):
                # ModelPatcherオブジェクトの場合
                model.model = model.model.to(torch.float8_e4m3fn)
                return (model, clip)
            else:
                model = model.to(torch.float8_e4m3fn)
                return (model, clip)
        except Exception as e:
            print(f"FP8への変換中にエラーが発生しました: {str(e)}")
            return (model, clip)  # エラー時は元のデータを返す

# ComfyUIのノードにこのクラスを登録するための定義
NODE_CLASS_MAPPINGS = {
    "FP8ConverterNode": FP8ConverterNode
}
