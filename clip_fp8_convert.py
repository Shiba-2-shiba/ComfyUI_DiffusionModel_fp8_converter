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

            # encodeメソッドを利用してテンソルを取得し変換する
            if hasattr(clip, 'encode'):
                encoded_tensor = clip.encode("sample text")  # 適切な引数を設定
                if isinstance(encoded_tensor, torch.Tensor):
                    encoded_tensor = encoded_tensor.to(torch.float8_e4m3fn)
                    print("encoded_tensor を float8_e4m3fn に変換しました。")
                else:
                    print("encode メソッドは torch.Tensor を返しませんでした。")
            elif hasattr(clip, 'cond_stage_model'):
                # cond_stage_model をチェックして変換可能か確認
                if isinstance(clip.cond_stage_model, torch.nn.Module):
                    clip.cond_stage_model = clip.cond_stage_model.to(torch.float8_e4m3fn)
                    print("cond_stage_model を float8_e4m3fn に変換しました。")
                else:
                    print("cond_stage_model は変換可能な torch.nn.Module ではありません。")
            else:
                print("CLIPモデルには変換可能な属性が見つかりません。")
            return (clip,)
        except Exception as e:
            print(f"CLIPモデルのfloat8_e4m3fnへの変換中にエラーが発生しました: {str(e)}")
            return (clip,)  # エラー時は元のデータを返す

# ComfyUIのノードにこのクラスを登録するための定義
NODE_CLASS_MAPPINGS = {
    "ClipFP8ConverterNode": ClipFP8ConverterNode
}
