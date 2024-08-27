import torch
from safetensors import save_file
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
            # MODELに対してFP8変換を適用
            model_fp8 = self.convert_patcher_to_fp8(model)
            # CLIPに対してFP8変換を適用
            clip_fp8 = self.convert_clip_to_fp8(clip)

            # 変換されたモデルを保存
            self.save_fp8_model(model_fp8, "converted_model_fp8.safetensors")
            self.save_fp8_model(clip_fp8, "converted_clip_fp8.safetensors")

            return model_fp8, clip_fp8
        except Exception as e:
            print(f"FP8変換中にエラーが発生しました: {str(e)}")
            return model, clip

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

    def convert_clip_to_fp8(self, clip):
        # CLIPの内部モデル（cond_stage_model）のパラメータとバッファをFP8形式に変換
        for name, param in tqdm(clip.cond_stage_model.named_parameters(), desc="CLIPパラメータをFP8に変換中"):
            if param.dtype in [torch.float32, torch.float16]:
                with torch.no_grad():
                    param.copy_(param.to(dtype=torch.float8_e4m3fn))

        for name, buffer in tqdm(clip.cond_stage_model.named_buffers(), desc="CLIPバッファをFP8に変換中"):
            if buffer.dtype in [torch.float32, torch.float16]:
                with torch.no_grad():
                    buffer.copy_(buffer.to(dtype=torch.float8_e4m3fn))

        return clip

    def save_fp8_model(self, model, filename):
        # モデルのstate_dictを保存
        state_dict = model.model_state_dict() if hasattr(model, 'model_state_dict') else model.cond_stage_model.state_dict()
        save_file(state_dict, filename, metadata={"format": "pt"})
        print(f"FP8に変換されたモデルが保存されました: {filename}")
