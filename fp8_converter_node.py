import torch
import os

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
            # ModelPatcher 内のモデルを FP8 に変換
            model_fp8 = model.model.to(torch.float8_e4m3fn) if hasattr(model, 'model') else model.to(torch.float8_e4m3fn)
            
            # CLIP のモデルを FP8 に変換
            if hasattr(clip, 'cond_stage_model'):
                clip_fp8 = clip.cond_stage_model.to(torch.float8_e4m3fn)
            else:
                raise AttributeError("CLIP オブジェクトに cond_stage_model 属性がありません")

            return (model_fp8, clip_fp8)
        except Exception as e:
            print(f"FP8への変換中にエラーが発生しました: {str(e)}")
            return (model, clip)  # エラー時は元のデータを返す

# ComfyUIのノードにこのクラスを登録するための定義
NODE_CLASS_MAPPINGS = {
    "FP8ConverterNode": FP8ConverterNode
}
