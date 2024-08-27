import torch
from tqdm.auto import tqdm

class FP8ConverterNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP",)
    FUNCTION = "convert_to_fp8"
    CATEGORY = "Model Processing"

    def convert_to_fp8(self, model_patcher, clip_patcher):
        try:
            # MODELに対してFP8変換を適用
            model_fp8_state_dict = self.convert_patcher_to_fp8(model_patcher)
            # CLIPに対してFP8変換を適用
            clip_fp8_state_dict = self.convert_clip_to_fp8(clip_patcher)

            # 変換されたモデルを保存
            self.save_fp8_model(model_fp8_state_dict, clip_fp8_state_dict, "converted_fp8_checkpoint.pth")

            return model_patcher, clip_patcher
        except Exception as e:
            print(f"FP8変換中にエラーが発生しました: {str(e)}")
            return model_patcher, clip_patcher

    def convert_patcher_to_fp8(self, patcher):
        try:
            # ModelPatcherが管理するモデルのstate_dictを取得
            model_state_dict = patcher.model.state_dict()

            # すべてのパラメータとバッファをFP8形式に変換
            for key, tensor in tqdm(model_state_dict.items(), desc="モデルパラメータをFP8に変換中"):
                if tensor.dtype in [torch.float32, torch.float16]:
                    model_state_dict[key] = tensor.to(dtype=torch.float8_e4m3fn)

            # 変換されたstate_dictを返す
            return model_state_dict
        except Exception as e:
            print(f"ModelPatcherのFP8変換中にエラーが発生しました: {str(e)}")
            raise

    def convert_clip_to_fp8(self, clip):
        try:
            # CLIPオブジェクトの内部モデルのstate_dictを取得
            clip_state_dict = clip.cond_stage_model.state_dict()

            # すべてのパラメータとバッファをFP8形式に変換
            for key, tensor in tqdm(clip_state_dict.items(), desc="CLIPパラメータをFP8に変換中"):
                if tensor.dtype in [torch.float32, torch.float16]:
                    clip_state_dict[key] = tensor.to(dtype=torch.float8_e4m3fn)

            # 変換されたstate_dictを返す
            return clip_state_dict
        except Exception as e:
            print(f"CLIPのFP8変換中にエラーが発生しました: {str(e)}")
            raise

    def save_fp8_model(self, model_state_dict, clip_state_dict, output_path):
        try:
            # state_dictを結合して保存
            combined_state_dict = {}
            if model_state_dict:
                combined_state_dict.update(model_state_dict)
            if clip_state_dict:
                combined_state_dict.update(clip_state_dict)

            # 保存時にテンソルが連続するように調整
            for key in combined_state_dict:
                tensor = combined_state_dict[key]
                if not tensor.is_contiguous():
                    combined_state_dict[key] = tensor.contiguous()

            # PyTorchのtorch.saveを使用してstate_dictを保存
            torch.save(combined_state_dict, output_path)
            print(f"FP8に変換されたチェックポイントが保存されました: {output_path}")
        except Exception as e:
            print(f"モデル保存中にエラーが発生しました: {str(e)}")
            raise
