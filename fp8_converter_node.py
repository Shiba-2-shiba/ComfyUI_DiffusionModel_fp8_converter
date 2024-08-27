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
            # モデルとCLIPをFP8形式に変換
            model_fp8 = model.model.to(torch.float8_e4m3fn)  # ModelPatcher内のモデルをFP8に変換
            clip_g_fp8 = clip.clip_g.to(torch.float8_e4m3fn)  # CLIPのclip_gサブモデルをFP8に変換
            clip_l_fp8 = clip.clip_l.to(torch.float8_e4m3fn)  # CLIPのclip_lサブモデルをFP8に変換

            # 新たにFP8変換されたモデルを元にしたclipを作成
            clip_fp8 = SDXLClipModel()
            clip_fp8.clip_g = clip_g_fp8
            clip_fp8.clip_l = clip_l_fp8
            
            return (model_fp8, clip_fp8)
        except Exception as e:
            print(f"FP8への変換中にエラーが発生しました: {str(e)}")
            return (model.model, clip)  # エラー時は元のデータを返す

# ComfyUIのノードにこのクラスを登録するための定義
NODE_CLASS_MAPPINGS = {
    "FP8ConverterNode": FP8ConverterNode
}
