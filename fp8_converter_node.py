import torch
from safetensors import safe_open
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

    def convert_to_fp8(self, model, clip):
        try:
            # ModelPatcher対応: パラメータの取得とFP8変換の適用
            model_fp8 = self.convert_patcher_to_fp8(model)
            clip_fp8 = self.convert_patcher_to_fp8(clip)

            # テスト: 変換後のdtypeを確認
            print("Model Parameters after FP8 conversion:")
            for name, param in model_fp8.model.named_parameters():
                print(f"Param: {name}, Dtype: {param.dtype}")
            
            print("Clip Parameters after FP8 conversion:")
            for name, param in clip_fp8.model.named_parameters():
                print(f"Param: {name}, Dtype: {param.dtype}")

            return model_fp8, clip_fp8
        except Exception as e:
            print(f"FP8変換中にエラーが発生しました: {str(e)}")
            return model, clip

    def convert_patcher_to_fp8(self, patcher):
        # ModelPatcher内のパラメータを取得してFP8変換を適用する
        if hasattr(patcher, 'model_state_dict'):
            state_dict = patcher.model_state_dict()

            for key, tensor in state_dict.items():
                if tensor.dtype in [torch.float32, torch.float16]:
                    with torch.no_grad():
                        state_dict[key].copy_(tensor.to(dtype=torch.float8_e4m3fn))

            # 変換後のstate_dictをModelPatcherに再適用する
            patcher.load()  # パラメータを反映するためにロードを再度呼び出す

        return patcher
