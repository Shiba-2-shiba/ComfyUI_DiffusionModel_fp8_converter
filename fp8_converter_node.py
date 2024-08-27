import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm.auto import tqdm
from execution import EXE_PATH  # 変更: 相対インポートから絶対インポートに修正
from nodes import NODE_TYPE
from node_helpers import InputModel, OutputModel

class FP8ConverterNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
            },
            "optional": {
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP",)
    FUNCTION = "convert_to_fp8"
    CATEGORY = "Model Processing"

    def convert_to_fp8(self, model, clip, vae=None):
        try:
            model_fp8, clip_fp8 = self.convert_model_to_fp8(model, clip)
            return model_fp8, clip_fp8
        except Exception as e:
            print(f"FP8変換中にエラーが発生しました: {str(e)}")
            return model, clip

    def convert_model_to_fp8(self, model, clip):
        components = {"model": {}, "clip": {}, "vae": {}}

        with safe_open(model, framework="pt", device="cpu") as f:
            for key in tqdm(f.keys(), desc="テンソルを変換中"):
                tensor = f.get_tensor(key)
                if "vae" in key:
                    components["vae"][key] = tensor.to(torch.float8_e4m3fn)
                elif "clip" in key:
                    components["clip"][key] = tensor.to(torch.float8_e4m3fn)
                else:
                    components["model"][key] = tensor.to(torch.float8_e4m3fn)

        model_fp8 = self.save_component(components["model"], "model_fp8.safetensors")
        clip_fp8 = self.save_component(components["clip"], "clip_fp8.safetensors")

        return model_fp8, clip_fp8

    def save_component(self, tensors, filename):
        output_path = f"{EXE_PATH}/models/{filename}"
        save_file(tensors, output_path, metadata={"format": "pt"})
        return output_path

# Register the node to the NODE_TYPE
NODE_TYPE["FP8ConverterNode"] = FP8ConverterNode
