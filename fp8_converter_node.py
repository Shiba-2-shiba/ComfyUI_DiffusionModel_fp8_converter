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

    def convert_to_fp8(self, model, clip):
        try:
            # Convert model and clip to FP8
            model_fp8 = self.convert_to_fp8_state(model)
            clip_fp8 = self.convert_to_fp8_state(clip.cond_stage_model)

            # Save the converted models
            self.save_checkpoint("converted_model_fp8.pth", model_fp8, clip_fp8)

            return model_fp8, clip_fp8
        except Exception as e:
            print(f"FP8変換中にエラーが発生しました: {str(e)}")
            return model, clip

    def convert_to_fp8_state(self, model):
        state_dict = model.state_dict()
        for key, tensor in tqdm(state_dict.items(), desc="FP8に変換中"):
            if tensor.dtype in [torch.float32, torch.float16]:
                state_dict[key] = tensor.to(dtype=torch.float8_e4m3fn)
        return state_dict

    def save_checkpoint(self, output_path, model_fp8, clip_fp8):
        # Collect state dictionaries
        state_dict = model_fp8
        if clip_fp8:
            state_dict.update(clip_fp8)

        # Ensure all tensors are contiguous for saving
        for k in state_dict:
            t = state_dict[k]
            if not t.is_contiguous():
                state_dict[k] = t.contiguous()

        # Save using torch.save
        torch.save(state_dict, output_path)
        print(f"FP8に変換されたモデルが保存されました: {output_path}")
