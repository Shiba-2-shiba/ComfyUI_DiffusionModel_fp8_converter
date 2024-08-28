import torch

class ClipFP8ConverterNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "convert_clip_to_fp8"

    CATEGORY = "conversion"

    def convert_clip_to_fp8(self, clip):
        try:
            # CLIPモデルを float8_e4m3fn 形式に変換
            clip = clip.to(torch.float8_e4m3fn)
            return (clip,)
        except Exception as e:
            print(f"CLIPモデルのfloat8_e4m3fnへの変換中にエラーが発生しました: {str(e)}")
            return (clip,)  # エラー時は元のデータを返す

# ComfyUIのノードにこのクラスを登録するための定義
NODE_CLASS_MAPPINGS = {
    "ClipFP8ConverterNode": ClipFP8ConverterNode
}
