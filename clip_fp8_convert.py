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
            # CLIPオブジェクトのタイプと属性を出力して調査
            print(f"CLIP model type: {type(clip)}")
            print(f"CLIP attributes: {dir(clip)}")

            # CLIPオブジェクト内部の変換可能な属性を変換する
            if hasattr(clip, 'model'):
                # ここで、内部のモデルを変換
                clip.model = clip.model.to(torch.float8_e4m3fn)
                print("CLIPモデルをfloat8_e4m3fnに変換しました。")
            else:
                print("CLIPモデルには変換可能な 'model' 属性がありません。")
            return (clip,)
        except Exception as e:
            print(f"CLIPモデルのfloat8_e4m3fnへの変換中にエラーが発生しました: {str(e)}")
            return (clip,)  # エラー時は元のデータを返す

# ComfyUIのノードにこのクラスを登録するための定義
NODE_CLASS_MAPPINGS = {
    "ClipFP8ConverterNode": ClipFP8ConverterNode
}
