import logging

from .blocks import build_block_selection_dict, build_klein_definition, build_krea2_definition

CATEGORY = "LoRA PowerMerge"
_FLOAT = {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}


class KREA2Blocks:
    """Model-specific block definition for KREA2 LoRAs."""
    RETURN_TYPES = ("BlockDefinition",)
    RETURN_NAMES = ("block_definition",)
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Per-block weight definition for KREA2 LoRAs "
                   "(diffusion_model.blocks.N + txtfusion + txtmlp). "
                   "Connect to PM Block Selector.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "blocks_group_size": ("INT", {"default": 5, "min": 1, "max": 128}),
            "blocks_weights": ("STRING", {"default": "1.0", "multiline": False,
                "tooltip": "Comma-separated per-group weights, left-to-right. "
                           "Missing groups default to 1.0."}),
            "txtfusion_layerwise": ("FLOAT", _FLOAT),
            "txtfusion_refiner": ("FLOAT", _FLOAT),
            "txtmlp": ("FLOAT", _FLOAT),
        }}

    def build(self, blocks_group_size, blocks_weights, txtfusion_layerwise,
              txtfusion_refiner, txtmlp):
        return (build_krea2_definition(blocks_group_size, blocks_weights,
                                       txtfusion_layerwise, txtfusion_refiner, txtmlp),)


class FluxKleinBlocks:
    """Model-specific block definition for FLUX.2-Klein LoRAs."""
    RETURN_TYPES = ("BlockDefinition",)
    RETURN_NAMES = ("block_definition",)
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Per-block weight definition for FLUX.2-Klein LoRAs "
                   "(double_blocks + single_blocks). Connect to PM Block Selector.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "double_blocks_group_size": ("INT", {"default": 1, "min": 1, "max": 128}),
            "double_blocks_weights": ("STRING", {"default": "1.0", "multiline": False,
                "tooltip": "Comma-separated per-group weights for double_blocks."}),
            "single_blocks_group_size": ("INT", {"default": 5, "min": 1, "max": 128}),
            "single_blocks_weights": ("STRING", {"default": "1.0", "multiline": False,
                "tooltip": "Comma-separated per-group weights for single_blocks."}),
        }}

    def build(self, double_blocks_group_size, double_blocks_weights,
              single_blocks_group_size, single_blocks_weights):
        return (build_klein_definition(double_blocks_group_size, double_blocks_weights,
                                       single_blocks_group_size, single_blocks_weights),)


class BlockSelector:
    """Bind a BlockDefinition to one LoRA (by stack index); chain to cover multiple LoRAs."""
    RETURN_TYPES = ("BlockSelection",)
    RETURN_NAMES = ("block_selection",)
    FUNCTION = "select"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Outputs a block selection config for the LoRA at 'index' in the LoRAStack. "
                   "Chain block_selection outputs to weight multiple LoRAs. "
                   "Feed the result into PM LoRA Stack Decompose.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "block_definition": ("BlockDefinition",),
                "index": ("INT", {"default": 0, "min": 0, "max": 1000,
                                  "tooltip": "Which LoRA in the stack (0-based) to weight."}),
            },
            "optional": {
                "block_selection": ("BlockSelection",),
            },
        }

    def select(self, block_definition, index, block_selection=None):
        result = build_block_selection_dict(block_selection, index, block_definition)
        logging.info(f"[PM Block Selector] index {index}: selection now covers "
                     f"{len(result['configs'])} LoRA(s).")
        return (result,)