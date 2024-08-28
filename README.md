# ComfyUI_ModelsClips_fp8_converter


これは、ComfyUIで、Diffusionモデル部分のみをfp8に変換するカスタムノードです。
CLIPとVAEは対応していません。
利点としては、unetなどに分ける必要がないところです。


This is a custom node in ComfyUI that converts only the Diffusion model portion to fp8.
CLIP and VAE are not supported.
The advantage is that it does not need to be separated into unet etc.


## Usage


以下のように間にノードをはさんで使用します。CLIPは変換しないのですが、仕様として入力します。

CLIP is not converted, but is entered as a specification.


![Example Workflow](https://github.com/Shiba-2-shiba/ComfyUI_DiffusionModel_fp8_converter/blob/main/workflowexample.png)

## Checkpoint


SDXL・Auraflow・HunyuanDITのsafetensorsファイルをf8変換してファイルサイズの圧縮を確認しました。


I have converted SDXL, Auraflow, and HunyuanDIT safetensors files to f8 and verified the file size compression.

## Install


以下のコマンドでインストール出来ます。

You can install it with the following command.


```bash
cd Yourdirectory/ComfyUI/custom_nodes
git clone https://github.com/Shiba-2-shiba/ComfyUI_DiffusionModel_fp8_converter.git

```


