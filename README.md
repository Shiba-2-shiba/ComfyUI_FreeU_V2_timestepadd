# ComfyUI_FreeU_V2の改修内容

このカスタムノードは、従来の FreeU V2をアップデートし、**指定した区間（timestep）で効果を適用する**機能を追加したバージョンです。  

timestepを加えることで、いつ・どのように効果を反映するかを制御できます。

さらにSDXLの拡散モデルにおける画像生成過程に合わせられるように、B1B2, S1S2ノードを分けて使用できるようにしました。

推奨は、B1B2を画像生成の前半部、S1S2ノードを画像生成の後半に適用することでより理想的な効果が期待できます。



## Install


以下のコマンドでインストール出来ます。

You can install it with the following command.


```bash
cd Yourdirectory/ComfyUI/custom_nodes
git clone https://github.com/Shiba-2-shiba/ComfyUI_FreeU_V2_timestepadd.git

```


---
## Reference Script
以下のスクリプトを参考にしています。

This custom node is based on the following script.

https://github.com/comfyanonymous/ComfyUI/blob/master/comfy_extras/nodes_freelunch.py#L25
