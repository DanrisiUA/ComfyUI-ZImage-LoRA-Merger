"""
Z-Image LoRA Merger - Custom nodes for ComfyUI
Fix overexposed images when using multiple LoRAs on distilled models

Author: DanrisiUA (https://github.com/DanrisiUA)
"""

from .zimage_lora_merger import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

