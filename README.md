# ComfyUI_ModelsClips_fp8_converter
これは、ComfyUIで、Diffusionモデル部分のみをfp8に変換するカスタムノードです。
CLIPとVAEは対応していません。
利点としては、unetなどに分ける必要がないところです。

## Usage
以下のように間にノードをはさんで使用します。CLIPは変換しないのですが、仕様として入力します。
![Example Workflow](https://github.com/Shiba-2-shiba/ComfyUI_DiffusionModel_fp8_converter/blob/main/workflowexample.png)

## Install
以下のコマンドでインストール出来ます。



