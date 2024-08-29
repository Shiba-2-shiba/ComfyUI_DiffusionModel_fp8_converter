# ComfyUI_DiffusionModel_fp8_converter


ComfyUIで、Diffusionモデル部分のみ/CLIPモデルのみをfp8に変換するカスタムノードです。

VAEのfp8変換については対応していません。

このノード利点としては、fp8変換時にあらかじめunet/clip/vaeなどに分ける必要がなく、ComfyUIが出している、あらかじめ1つにまとめてあるsafetenrosファイルを使用できる点です。


This is a custom node to convert only the Diffusion model part or CLIP model part to fp8 in ComfyUI.

VAE fp8 conversion is not supported.

The advantage of this node is that you do not need to separate unet/clip/vae in advance when converting to fp8, but can use the safetenros files that ComfyUI provides.


## Usage


以下のように間にノードをはさんで使用します。

CLIP is not converted, but is entered as a specification.


![Example Workflow](https://github.com/Shiba-2-shiba/ComfyUI_DiffusionModel_fp8_converter/blob/main/model_convert_example.png)

## Checkpoint


SDXL・Auraflow・HunyuanDITのsafetensorsファイルをf8変換してファイルサイズの圧縮を確認しました。

変換したチェックポイントは、通常の画像生成フローで使用してもエラーの発生なく使用出来ています。


I have converted SDXL, Auraflow, and HunyuanDIT safetensors files to f8 and verified the file size compression.

The checkpoints can be used in the normal image generation flow without any errors.

＜変換によるサイズの変化＞

①  SDXL        

6.6GB    ⇒    4.1GB

②  Auraflow    

16GB     ⇒    8.9GB

③  HunyanDiT　

7.7GB　⇒    5.4GB

## Install


以下のコマンドでインストール出来ます。

You can install it with the following command.


```bash
cd Yourdirectory/ComfyUI/custom_nodes
git clone https://github.com/Shiba-2-shiba/ComfyUI_DiffusionModel_fp8_converter.git

```


