# Import the V3 node from the new file
from .FreeU_V2_timestepadd_v3 import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Expose the mappings to ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
