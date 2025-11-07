# [ファイル名: __init__.py]
# (変更あり: FreeU_V2_timestepadd をインポートしてリストに追加)

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

# 1. 各モジュールからノードクラスをインポート
from .FreeU_B1B2 import FreeU_B_Scaling
from .FreeU_S1S2 import FreeU_S_Scaling_AdaptiveCap
from .FreeU_V2_timestepadd import FreeU_V2_timestepadd


# 2. ComfyExtensionを継承したクラスを作成します
class FreeU_Modular_Extension(ComfyExtension): 
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            FreeU_B_Scaling,
            FreeU_S_Scaling_AdaptiveCap,
            FreeU_V2_timestepadd 
        ]

# 3. comfy_entrypoint
async def comfy_entrypoint() -> FreeU_Modular_Extension:
    return FreeU_Modular_Extension()

# 3. 読み込まれたことが分かりやすいよう、print文も更新 (任意)
print("Loaded: FreeU (Modular B/S V5 + V2 Timestep) via V3 entrypoint")
