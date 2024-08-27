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
            # ModelPatcherオブジェクトを使用してモデルとクリップをFP8に変換
            model_fp8 = self.convert_patcher_to_fp8(model_patcher)
            clip_fp8 = self.convert_patcher_to_fp8(clip_patcher)
            return model_fp8, clip_fp8
        except Exception as e:
            print(f"FP8変換中にエラーが発生しました: {str(e)}")
            return model_patcher, clip_patcher

    def convert_patcher_to_fp8(self, patcher):
        # ModelPatcherが管理するモデルのstate_dictを取得
        state_dict = patcher.model_state_dict()

        # すべてのパラメータとバッファをFP8形式に変換
        for key, tensor in tqdm(state_dict.items(), desc="モデルパラメータをFP8に変換中"):
            if tensor.dtype in [torch.float32, torch.float16]:
                state_dict[key] = tensor.to(dtype=torch.float8_e4m3fn)

        # 変換されたstate_dictを再度モデルにロード
        patcher.model.load_state_dict(state_dict)

        return patcher
