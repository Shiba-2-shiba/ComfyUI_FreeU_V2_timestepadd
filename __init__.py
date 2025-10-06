# __init__.py (V3 comfy_entrypoint 登録方式)
# 
# V3スキーマの作法に則り、ComfyExtensionを定義し、
# comfy_entrypoint関数からそのインスタンスを返すように変更します。

# 1. 必要なV3モジュールと、登録したいノードクラスをインポートします
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override
from .FreeU_V2_timestepadd import FreeU_V2_timestepadd

# 2. ComfyExtensionを継承したクラスを作成します
class FreeUV2Extension(ComfyExtension):
    # get_node_listメソッドで、登録したいノードクラスのリストを返します
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [FreeU_V2_timestepadd]

# 3. comfy_entrypointという名前の非同期関数を定義し、
#    上で作成したExtensionクラスのインスタンスを返します
async def comfy_entrypoint() -> FreeUV2Extension:
    return FreeUV2Extension()

print("Loaded: FreeU V2 (TimestepAdd) via V3 entrypoint")
