# ComfyUI_FreeU_V2_timestepadd

このカスタムノードは、従来の FreeU V2をアップデートし、**指定した区間（timestep）で効果を適用する**機能を追加したバージョンです。  


timestepを加えることで、いつ・どのように効果を反映するかを制御できます。


この手法自体はStable diffusion webui forgeのFreeUでは行うことができていましたが、ComfyUIではそれに対応するものが無かったので作成しました。


This custom node is an updated version of the previous FreeU V2 with the added ability to **apply effects at specified intervals (timesteps)**. 


By adding timestep, you can control when and how the effect is reflected. 


This technique itself was able to be done with the Stable diffusion webui forge FreeU, but ComfyUI did not have a counterpart for it, so we created it.

---
## スクリーンショット

下記のように、パラメータを一括で設定するUIが表示されます。

As shown below, a UI for setting parameters at once will appear.

![FreeU_V2_timestepadd](https://github.com/Shiba-2-shiba/ComfyUI_FreeU/blob/main/img1.png)

start_percentとend_percentで、FreeUの効果範囲を設定できます。

特にstart_percentを0より大きくすることで、生成画像が安定しやすくなります。

その他のパラメータの設定はFreeU V2と同様です。

The start_percent and end_percent allow you to set the range of the FreeU effect.

In particular, setting start_percent larger than 0 makes it easier to stabilize the generated image.

Other parameter settings are the same as for FreeU V2.

---
## Install


以下のコマンドでインストール出来ます。

You can install it with the following command.


```bash
cd Yourdirectory/ComfyUI/custom_nodes
git clone https://github.com/Shiba-2-shiba/ComfyUI_FreeU_V2_timestepadd.git

```

## Usage
以下の記事で経緯や使用方法について記載していますので参考にしてください。

Please refer to the following article describing the background and usage in Japanese.


https://note.com/gentle_murre488/n/n722f6a73561c

---
## Reference Script
以下のスクリプトを参考にしています。

This custom node is based on the following script.

https://github.com/comfyanonymous/ComfyUI/blob/master/comfy_extras/nodes_freelunch.py#L25
