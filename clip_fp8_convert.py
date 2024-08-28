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
            # CLIPモデルの型を確認して出力
            print(f"CLIP model type: {type(clip)}")

            # CLIPモデルを float8_e4m3fn 形式に変換する
            if isinstance(clip, torch.Tensor):
                clip = clip.to(torch.float8_e4m3fn)
            else:
                print("CLIPモデルはtorch.Tensorではないため、float8_e4m3fnへの変換がサポートされていません。")
            return (clip,)
        except Exception as e:
            print(f"CLIPモデルのfloat8_e4m3fnへの変換中にエラーが発生しました: {str(e)}")
            return (clip,)  # エラー時は元のデータを返す
