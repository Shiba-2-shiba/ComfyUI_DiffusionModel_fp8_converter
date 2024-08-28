from .model_fp8_converter import ModelFP8ConverterNode
from .clip_fp8_convert import ClipFP8ConverterNode  # ここを追加

# ノードクラスのマッピングを設定
NODE_CLASS_MAPPINGS = {
    "ModelFP8ConverterNode": ModelFP8ConverterNode,
    "ClipFP8ConverterNode": ClipFP8ConverterNode  # ここを追加
}
