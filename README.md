# ComfyUI_FreeU_V2_timestepadd

このカスタムノードは、従来の FreeU V2をアップデートし、**指定した区間（timestep）で効果を適用する**機能を追加したバージョンです。  

timestepを加えることで、いつ・どのように効果を反映するかを制御できます。

この手法自体はStable diffusion webui forgeのFreeUでは行うことができていましたが、ComfyUIではそれに対応するものが無かったので作成しました。

---
## スクリーンショット

下記のように、パラメータを一括で設定するUIが表示されます。

![FreeU_V2_timestepadd](https://github.com/Shiba-2-shiba/ComfyUI_FreeU/blob/main/img1.png)

start_percentとend_percentで、FreeUの効果範囲を設定できます。

特にstart_percentを0より大きくすることで、生成画像が安定しやすくなります。

その他のパラメータの設定はFreeU V2と同様です。

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
https://note.com/gentle_murre488/n/n722f6a73561c

---
## Reference Script
以下のスクリプトを参考にしています。
https://github.com/comfyanonymous/ComfyUI/blob/master/comfy_extras/nodes_freelunch.py#L25
