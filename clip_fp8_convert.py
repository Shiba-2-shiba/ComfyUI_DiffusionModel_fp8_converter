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

            # clip_layerをチェックして変換
            if hasattr(clip, 'clip_layer'):
                self._convert_model_to_fp8(clip.clip_layer)
                print("clip_layer を float8_e4m3fn に変換しました。")

            # cond_stage_model をチェックして変換
            if hasattr(clip, 'cond_stage_model'):
                self._convert_model_to_fp8(clip.cond_stage_model)
                print("cond_stage_model を float8_e4m3fn に変換しました。")

            return (clip,)
        except Exception as e:
            print(f"CLIPモデルのfloat8_e4m3fnへの変換中にエラーが発生しました: {str(e)}")
            return (clip,)  # エラー時は元のデータを返す

    def _convert_model_to_fp8(self, model):
        """モデル全体を再帰的にfloat8_e4m3fn形式に変換するヘルパーメソッド"""
        if isinstance(model, torch.nn.Module):
            for name, param in model.named_parameters():
                if param is not None:
                    print(f"Converting {name} to float8_e4m3fn")
                    param.data = param.data.to(torch.float8_e4m3fn)
        elif isinstance(model, torch.Tensor):
            return model.to(torch.float8_e4m3fn)
        else:
            print("未対応の型です: ", type(model))
        return model

# ComfyUIのノードにこのクラスを登録するための定義
NODE_CLASS_MAPPINGS = {
    "ClipFP8ConverterNode": ClipFP8ConverterNode
}
