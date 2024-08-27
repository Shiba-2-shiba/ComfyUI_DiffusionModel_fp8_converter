import torch
import os

# ComfyUIのmodel_patcher.pyからModelPatcherをインポート
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
            # モデルをFP8形式に変換
            if hasattr(model, 'diffusion_model'):
                model_fp8 = model.diffusion_model.to(torch.float8_e4m3fn)
            elif isinstance(model, ModelPatcher):
                # ModelPatcherオブジェクトの場合は、内部モデルに対して処理を行う
                model_fp8 = model.model.to(torch.float8_e4m3fn)  # 内部の適切な属性を指定
            else:
                model_fp8 = model.to(torch.float8_e4m3fn)
            
            # CLIPのモデルをFP8形式に変換
            if isinstance(clip, SDXLClipModel):
                clip_l_fp8 = clip.clip_l.to(torch.float8_e4m3fn)
                clip_g_fp8 = clip.clip_g.to(torch.float8_e4m3fn)
                
                # 新しいSDXLClipModelオブジェクトを作成し、FP8のモデルを割り当てる
                clip_fp8 = SDXLClipModel()
                clip_fp8.clip_l = clip_l_fp8
                clip_fp8.clip_g = clip_g_fp8
            else:
                raise AttributeError("CLIPオブジェクトがSDXLClipModelではありません")

            return (model_fp8, clip_fp8)
        except Exception as e:
            print(f"FP8への変換中にエラーが発生しました: {str(e)}")
            return (model, clip)  # エラー時は元のデータを返す

# ComfyUIのノードにこのクラスを登録するための定義
NODE_CLASS_MAPPINGS = {
    "FP8ConverterNode": FP8ConverterNode
}
