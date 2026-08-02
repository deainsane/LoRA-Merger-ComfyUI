import os

import comfy
import folder_paths
import torch


from .architectures.sd_lora import convert_to_regular_lora

def sanitize_for_save(tensor):
    """Return a tensor safetensors can serialize: CPU, dense, contiguous, no shared storage."""
    if not isinstance(tensor, torch.Tensor):
        return tensor
    tensor = tensor.detach()
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    if tensor.is_contiguous() and tensor.numel() == tensor.untyped_storage().nbytes() // tensor.element_size():
        return tensor
    # `.contiguous()` is a no-op for a contiguous view into a larger storage,
    # so clone those to drop the shared (oversized) storage.
    return tensor.contiguous().clone() if tensor.is_contiguous() else tensor.contiguous()


class LoraSave:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "lora": ("LoRABundle",),
            "file_name": ("STRING", {"multiline": False, "default": "merged"}), "extension": (["safetensors"], ),
        }}
    RETURN_TYPES = ()
    FUNCTION = "lora_save"
    CATEGORY = "LoRA PowerMerge"

    OUTPUT_NODE = True

    def lora_save(self, model, lora, file_name, extension):
        save_path = os.path.join(folder_paths.folder_names_and_paths["loras"][0][0], file_name + "." + extension)

        # Convert model weights from ComfyUI format to regular LoRA format
        state_dict = lora['lora']
        new_state_dict = convert_to_regular_lora(model, state_dict)

        # CLIP layers are now merged as part of the merge process,
        # so we don't need to copy them from lora_raw anymore.
        # The merged CLIP weights are already in state_dict.

        # Refactoring/merging produces transposed or sliced views (e.g. `(V * s).T`,
        # `U[:, :r]`), which safetensors refuses to serialize. Detach the views from
        # their backing storage before saving.
        new_state_dict = {k: sanitize_for_save(v) for k, v in new_state_dict.items()}

        print(f"Saving LoRA to {save_path}")
        comfy.utils.save_torch_file(new_state_dict, save_path)

        return {}
