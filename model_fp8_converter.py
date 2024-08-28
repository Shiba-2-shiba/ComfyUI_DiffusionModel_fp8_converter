import torch
import os

# 必要な場所からModelPatcherをインポート
from comfy.model_patcher import ModelPatcher

class ModelFP8ConverterNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "convert_to_fp8"

    CATEGORY = "conversion"

    def convert_to_fp8(self, model):
        try:
            # モデルを float8_e4m3fn 形式に変換
            if hasattr(model, 'diffusion_model'):
                model.diffusion_model = model.diffusion_model.to(torch.float8_e4m3fn)
                return (model,)
            elif isinstance(model, ModelPatcher):
                # ModelPatcherオブジェクトの場合
                model.model = model.model.to(torch.float8_e4m3fn)
                return (model,)
            else:
                model = model.to(torch.float8_e4m3fn)
                return (model,)
        except Exception as e:
            print(f"float8_e4m3fn への変換中にエラーが発生しました: {str(e)}")
            return (model,)  # エラー時は元のデータを返す

# ComfyUIのノードにこのクラスを登録するための定義
NODE_CLASS_MAPPINGS = {
    "ModelFP8ConverterNode": ModelFP8ConverterNode
}
