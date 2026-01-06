"""
Z-Image LoRA Merger for ComfyUI
Custom nodes for combining multiple LoRAs without overexposure on distilled models.

Problem: Standard LoRA application adds effects additively, causing overexposure
on distilled models like Z-Image Turbo, SDXL-Turbo, LCM, etc.

Solution: Various blending strategies to normalize the combined LoRA effect.

Author: DanrisiUA (https://github.com/DanrisiUA)
"""

import torch
import logging
import math
import folder_paths
import comfy.utils
import comfy.sd
import comfy.lora


class ZImageLoRAMerger:
    """
    Node for combining multiple LoRAs with various blending strategies,
    optimized for Z-Image Turbo and other distilled models.
    """
    
    BLEND_MODES = [
        "normalize",      # Normalizes total strength to target_strength
        "average",        # Averages LoRA effects
        "sqrt_scale",     # Scales each LoRA by 1/sqrt(n)
        "linear_decay",   # Linear decay: 1, 0.5, 0.33, ...
        "geometric_decay",# Geometric decay: 1, 0.5, 0.25, ...
        "additive",       # Standard additive (for comparison)
    ]
    
    def __init__(self):
        self.loaded_loras = {}
    
    @classmethod
    def INPUT_TYPES(cls):
        lora_list = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model to apply LoRA to"}),
                "clip": ("CLIP", {"tooltip": "The CLIP model"}),
                "blend_mode": (cls.BLEND_MODES, {"default": "normalize", "tooltip": "LoRA blending mode"}),
                "target_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, 
                                              "tooltip": "Target total strength (for normalize/average)"}),
                "lora_1": (["None"] + lora_list, {"tooltip": "First LoRA"}),
                "strength_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "lora_2": (["None"] + lora_list, {"tooltip": "Second LoRA"}),
                "strength_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
            },
            "optional": {
                "lora_3": (["None"] + lora_list, {"tooltip": "Third LoRA"}),
                "strength_3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "lora_4": (["None"] + lora_list, {"tooltip": "Fourth LoRA"}),
                "strength_4": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "lora_5": (["None"] + lora_list, {"tooltip": "Fifth LoRA"}),
                "strength_5": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "clip_strength_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                                       "tooltip": "Strength multiplier for CLIP"}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "merge_loras"
    CATEGORY = "loaders/lora"
    DESCRIPTION = "Combines multiple LoRAs with normalization for Z-Image Turbo and other distilled models"
    
    def _load_lora(self, lora_name):
        """Loads LoRA file with caching"""
        if lora_name == "None" or lora_name is None:
            return None
            
        if lora_name in self.loaded_loras:
            return self.loaded_loras[lora_name]
        
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        self.loaded_loras[lora_name] = lora
        return lora
    
    def _calculate_blend_factors(self, strengths, blend_mode, target_strength):
        """
        Calculates blending coefficients for each LoRA
        based on the selected mode.
        """
        n = len(strengths)
        if n == 0:
            return []
        
        if blend_mode == "additive":
            # Standard additive application
            return strengths
        
        elif blend_mode == "normalize":
            # Normalizes so that sum of squared strengths = target_strength^2
            # This preserves the "energy" of the effect
            sum_sq = sum(s*s for s in strengths)
            if sum_sq == 0:
                return [0.0] * n
            scale = target_strength / math.sqrt(sum_sq)
            return [s * scale for s in strengths]
        
        elif blend_mode == "average":
            # Simple averaging with target_strength
            scale = target_strength / n
            return [s * scale for s in strengths]
        
        elif blend_mode == "sqrt_scale":
            # Scales by 1/sqrt(n) - maintains balance when adding LoRAs
            scale = 1.0 / math.sqrt(n)
            return [s * scale for s in strengths]
        
        elif blend_mode == "linear_decay":
            # Each subsequent LoRA has less weight: 1, 0.5, 0.33, 0.25, ...
            factors = [1.0 / (i + 1) for i in range(n)]
            total = sum(factors)
            return [strengths[i] * factors[i] * target_strength / total for i in range(n)]
        
        elif blend_mode == "geometric_decay":
            # Geometric decay: 1, 0.5, 0.25, 0.125, ...
            factors = [0.5 ** i for i in range(n)]
            total = sum(factors)
            return [strengths[i] * factors[i] * target_strength / total for i in range(n)]
        
        else:
            return strengths
    
    def merge_loras(self, model, clip, blend_mode, target_strength,
                    lora_1, strength_1, lora_2, strength_2,
                    lora_3="None", strength_3=1.0,
                    lora_4="None", strength_4=1.0,
                    lora_5="None", strength_5=1.0,
                    clip_strength_multiplier=1.0):
        """
        Main function for combining LoRAs.
        """
        
        # Collect active LoRAs
        loras_data = []
        for lora_name, strength in [
            (lora_1, strength_1),
            (lora_2, strength_2),
            (lora_3, strength_3),
            (lora_4, strength_4),
            (lora_5, strength_5),
        ]:
            if lora_name != "None" and lora_name is not None and strength != 0:
                lora = self._load_lora(lora_name)
                if lora is not None:
                    loras_data.append((lora, strength))
        
        if len(loras_data) == 0:
            return (model, clip)
        
        # Calculate blend factors
        original_strengths = [s for _, s in loras_data]
        blended_strengths = self._calculate_blend_factors(
            original_strengths, blend_mode, target_strength
        )
        
        logging.info(f"Z-Image LoRA Merger: {blend_mode} mode, {len(loras_data)} LoRAs")
        logging.info(f"  Original strengths: {original_strengths}")
        logging.info(f"  Blended strengths:  {[round(s, 4) for s in blended_strengths]}")
        
        # Apply LoRAs sequentially with calculated coefficients
        new_model = model
        new_clip = clip
        
        for i, ((lora, _), strength) in enumerate(zip(loras_data, blended_strengths)):
            clip_strength = strength * clip_strength_multiplier
            new_model, new_clip = comfy.sd.load_lora_for_models(
                new_model, new_clip, lora, strength, clip_strength
            )
        
        return (new_model, new_clip)


class ZImageLoRAStack:
    """
    Node for creating a LoRA stack that can be used with ZImageLoRAStackApply.
    Allows more flexible LoRA management through connections.
    """
    
    def __init__(self):
        self.loaded_loras = {}
    
    @classmethod
    def INPUT_TYPES(cls):
        lora_list = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "lora_name": (lora_list, {"tooltip": "LoRA file"}),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
            },
            "optional": {
                "lora_stack": ("LORA_STACK", {"tooltip": "Previous LoRA stack"}),
            }
        }
    
    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("lora_stack",)
    FUNCTION = "add_to_stack"
    CATEGORY = "loaders/lora"
    DESCRIPTION = "Adds LoRA to stack for later application with ZImageLoRAStackApply"
    
    def add_to_stack(self, lora_name, strength, lora_stack=None):
        lora_list = list(lora_stack) if lora_stack else []
        
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        
        lora_list.append({
            "name": lora_name,
            "lora": lora,
            "strength": strength
        })
        
        return (lora_list,)


class ZImageLoRAStackApply:
    """
    Applies a LoRA stack with various blending modes.
    """
    
    BLEND_MODES = ZImageLoRAMerger.BLEND_MODES
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model to apply LoRA to"}),
                "clip": ("CLIP", {"tooltip": "The CLIP model"}),
                "lora_stack": ("LORA_STACK", {"tooltip": "LoRA stack"}),
                "blend_mode": (cls.BLEND_MODES, {"default": "normalize"}),
                "target_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "clip_strength_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "apply_stack"
    CATEGORY = "loaders/lora"
    DESCRIPTION = "Applies LoRA stack with selected blending mode"
    
    def _calculate_blend_factors(self, strengths, blend_mode, target_strength):
        """Same logic as ZImageLoRAMerger"""
        n = len(strengths)
        if n == 0:
            return []
        
        if blend_mode == "additive":
            return strengths
        elif blend_mode == "normalize":
            sum_sq = sum(s*s for s in strengths)
            if sum_sq == 0:
                return [0.0] * n
            scale = target_strength / math.sqrt(sum_sq)
            return [s * scale for s in strengths]
        elif blend_mode == "average":
            scale = target_strength / n
            return [s * scale for s in strengths]
        elif blend_mode == "sqrt_scale":
            scale = 1.0 / math.sqrt(n)
            return [s * scale for s in strengths]
        elif blend_mode == "linear_decay":
            factors = [1.0 / (i + 1) for i in range(n)]
            total = sum(factors)
            return [strengths[i] * factors[i] * target_strength / total for i in range(n)]
        elif blend_mode == "geometric_decay":
            factors = [0.5 ** i for i in range(n)]
            total = sum(factors)
            return [strengths[i] * factors[i] * target_strength / total for i in range(n)]
        else:
            return strengths
    
    def apply_stack(self, model, clip, lora_stack, blend_mode, target_strength, clip_strength_multiplier):
        if not lora_stack or len(lora_stack) == 0:
            return (model, clip)
        
        original_strengths = [item["strength"] for item in lora_stack]
        blended_strengths = self._calculate_blend_factors(
            original_strengths, blend_mode, target_strength
        )
        
        logging.info(f"Z-Image LoRA Stack Apply: {blend_mode} mode, {len(lora_stack)} LoRAs")
        logging.info(f"  Original strengths: {original_strengths}")
        logging.info(f"  Blended strengths:  {[round(s, 4) for s in blended_strengths]}")
        
        new_model = model
        new_clip = clip
        
        for item, strength in zip(lora_stack, blended_strengths):
            clip_strength = strength * clip_strength_multiplier
            new_model, new_clip = comfy.sd.load_lora_for_models(
                new_model, new_clip, item["lora"], strength, clip_strength
            )
        
        return (new_model, new_clip)


class ZImageLoRAMergeToSingle:
    """
    Merges weights of multiple LoRAs into a single "virtual" LoRA
    by pre-merging weights before applying to the model.
    
    This can give better results than sequential application,
    especially for overlapping weights.
    """
    
    MERGE_METHODS = [
        "weighted_sum",    # Weighted sum
        "add_difference",  # A + (B - C) * weight
        "weighted_average",# Weighted average
    ]
    
    def __init__(self):
        self.loaded_loras = {}
    
    @classmethod
    def INPUT_TYPES(cls):
        lora_list = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "merge_method": (cls.MERGE_METHODS, {"default": "weighted_average"}),
                "output_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                              "tooltip": "Strength of the merged LoRA"}),
                "lora_1": (["None"] + lora_list,),
                "weight_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "lora_2": (["None"] + lora_list,),
                "weight_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "lora_3": (["None"] + lora_list,),
                "weight_3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "clip_strength_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "merge_to_single"
    CATEGORY = "loaders/lora"
    DESCRIPTION = "Merges multiple LoRAs into one before applying - optimal for Z-Image Turbo"
    
    def _load_lora(self, lora_name):
        if lora_name == "None" or lora_name is None:
            return None
        if lora_name in self.loaded_loras:
            return self.loaded_loras[lora_name]
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        self.loaded_loras[lora_name] = lora
        return lora
    
    def _merge_lora_weights(self, loras_with_weights, method):
        """
        Merges LoRA weights into a single dictionary.
        Handles LoRAs with different ranks - incompatible layers are taken
        proportionally from the LoRA that has them.
        """
        if len(loras_with_weights) == 0:
            return {}
        
        if len(loras_with_weights) == 1:
            lora, weight = loras_with_weights[0]
            return {k: v * weight for k, v in lora.items()}
        
        # Collect all keys
        all_keys = set()
        for lora, _ in loras_with_weights:
            all_keys.update(lora.keys())
        
        merged = {}
        skipped_keys = 0
        
        for key in all_keys:
            tensors_weights = []
            for lora, weight in loras_with_weights:
                if key in lora:
                    tensors_weights.append((lora[key], weight))
            
            if len(tensors_weights) == 0:
                continue
            
            if len(tensors_weights) == 1:
                merged[key] = tensors_weights[0][0] * tensors_weights[0][1]
                continue
            
            # Get reference shape and dtype
            ref_tensor = tensors_weights[0][0]
            ref_shape = ref_tensor.shape
            device = ref_tensor.device
            dtype = ref_tensor.dtype
            
            # Filter only tensors with compatible shapes
            compatible_tensors = [(t, w) for t, w in tensors_weights if t.shape == ref_shape]
            
            # If shapes don't match, take weighted sum of compatible ones only
            # or first tensor if none are compatible
            if len(compatible_tensors) < len(tensors_weights):
                skipped_keys += 1
                if len(compatible_tensors) == 0:
                    # No compatible - take first with its weight
                    merged[key] = ref_tensor * tensors_weights[0][1]
                    continue
                elif len(compatible_tensors) == 1:
                    merged[key] = compatible_tensors[0][0] * compatible_tensors[0][1]
                    continue
                tensors_weights = compatible_tensors
            
            if method == "weighted_sum":
                result = torch.zeros_like(ref_tensor)
                for tensor, weight in tensors_weights:
                    result += tensor.to(device=device, dtype=dtype) * weight
                merged[key] = result
                
            elif method == "weighted_average":
                result = torch.zeros_like(ref_tensor)
                total_weight = sum(w for _, w in tensors_weights)
                if total_weight > 0:
                    for tensor, weight in tensors_weights:
                        result += tensor.to(device=device, dtype=dtype) * (weight / total_weight)
                merged[key] = result
                
            elif method == "add_difference":
                # First LoRA + difference with others
                result = tensors_weights[0][0].clone() * tensors_weights[0][1]
                for tensor, weight in tensors_weights[1:]:
                    diff = tensor.to(device=device, dtype=dtype) - tensors_weights[0][0]
                    result += diff * weight
                merged[key] = result
        
        if skipped_keys > 0:
            logging.warning(f"Z-Image LoRA Merge: {skipped_keys} keys had incompatible shapes (different LoRA ranks) - used first compatible tensor")
        
        return merged
    
    def merge_to_single(self, model, clip, merge_method, output_strength,
                        lora_1, weight_1, lora_2, weight_2,
                        lora_3="None", weight_3=1.0,
                        clip_strength_multiplier=1.0):
        
        loras_with_weights = []
        for lora_name, weight in [(lora_1, weight_1), (lora_2, weight_2), (lora_3, weight_3)]:
            if lora_name != "None" and lora_name is not None and weight > 0:
                lora = self._load_lora(lora_name)
                if lora is not None:
                    loras_with_weights.append((lora, weight))
        
        if len(loras_with_weights) == 0:
            return (model, clip)
        
        logging.info(f"Z-Image LoRA Merge: {merge_method}, {len(loras_with_weights)} LoRAs -> 1")
        
        merged_lora = self._merge_lora_weights(loras_with_weights, merge_method)
        
        clip_strength = output_strength * clip_strength_multiplier
        new_model, new_clip = comfy.sd.load_lora_for_models(
            model, clip, merged_lora, output_strength, clip_strength
        )
        
        return (new_model, new_clip)


class ZImageLoRATrueMerge:
    """
    "True" LoRA merging by computing full weight diffs.
    
    Works for LoRAs of ANY rank!
    
    Instead of merging raw A/B matrices (impossible for different ranks),
    computes the full diff = A @ B × alpha for each LoRA,
    then averages these diffs and applies as a single patch.
    
    Warning: Requires more memory and time than standard application.
    """
    
    MERGE_MODES = [
        "weighted_average",  # Weighted average of diffs
        "weighted_sum",      # Weighted sum (can be brighter)
        "normalize",         # Energy normalization
    ]
    
    def __init__(self):
        self.loaded_loras = {}
    
    @classmethod
    def INPUT_TYPES(cls):
        lora_list = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model to apply LoRA to"}),
                "clip": ("CLIP", {"tooltip": "The CLIP model"}),
                "merge_mode": (cls.MERGE_MODES, {"default": "weighted_average"}),
                "output_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                              "tooltip": "Strength of the merged effect"}),
                "lora_1": (["None"] + lora_list,),
                "strength_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "lora_2": (["None"] + lora_list,),
                "strength_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
            },
            "optional": {
                "lora_3": (["None"] + lora_list,),
                "strength_3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "lora_4": (["None"] + lora_list,),
                "strength_4": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "clip_strength_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "true_merge"
    CATEGORY = "loaders/lora"
    DESCRIPTION = "True LoRA merging for any rank combination by computing full weight diffs"
    
    def _load_lora(self, lora_name):
        """Loads LoRA file with caching"""
        if lora_name == "None" or lora_name is None:
            return None
        if lora_name in self.loaded_loras:
            return self.loaded_loras[lora_name]
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        self.loaded_loras[lora_name] = lora
        return lora
    
    def _get_lora_key_info(self, lora_dict, key_prefix):
        """
        Extracts LoRA information for the given key.
        Returns (mat_up, mat_down, alpha, mid) or None.
        """
        # LoRA key formats
        formats = [
            ("{}.lora_up.weight", "{}.lora_down.weight"),           # regular
            ("{}_lora.up.weight", "{}_lora.down.weight"),           # diffusers
            ("{}.lora_B.weight", "{}.lora_A.weight"),               # diffusers2
            ("{}.lora.up.weight", "{}.lora.down.weight"),           # diffusers3
        ]
        
        for up_fmt, down_fmt in formats:
            up_key = up_fmt.format(key_prefix)
            down_key = down_fmt.format(key_prefix)
            
            if up_key in lora_dict and down_key in lora_dict:
                mat_up = lora_dict[up_key]
                mat_down = lora_dict[down_key]
                
                # Alpha
                alpha_key = "{}.alpha".format(key_prefix)
                alpha = lora_dict.get(alpha_key, None)
                if alpha is not None:
                    alpha = alpha.item()
                else:
                    alpha = mat_down.shape[0]  # rank as default
                
                # Mid (for LoCon)
                mid_key = "{}.lora_mid.weight".format(key_prefix)
                mid = lora_dict.get(mid_key, None)
                
                return (mat_up, mat_down, alpha, mid)
        
        return None
    
    def _compute_lora_diff(self, mat_up, mat_down, alpha, mid, target_shape):
        """
        Computes full diff for a single LoRA.
        diff = mat_up @ mat_down × (alpha / rank)
        """
        rank = mat_down.shape[0]
        scale = alpha / rank
        
        if mid is not None:
            # LoCon with mid matrix (rare)
            final_shape = [mat_down.shape[1], mat_down.shape[0], mid.shape[2], mid.shape[3]]
            mat_down = (
                torch.mm(
                    mat_down.transpose(0, 1).flatten(start_dim=1),
                    mid.transpose(0, 1).flatten(start_dim=1),
                )
                .reshape(final_shape)
                .transpose(0, 1)
            )
        
        # Compute diff
        diff = torch.mm(
            mat_up.flatten(start_dim=1).float(),
            mat_down.flatten(start_dim=1).float()
        )
        
        # Try to reshape to target shape
        try:
            diff = diff.reshape(target_shape)
        except RuntimeError:
            # If shape doesn't match, skip
            return None
        
        return diff * scale
    
    def _merge_diffs(self, diffs_with_weights, mode):
        """
        Merges a list of diffs with their weights.
        """
        if len(diffs_with_weights) == 0:
            return None
        
        if len(diffs_with_weights) == 1:
            diff, weight = diffs_with_weights[0]
            return diff * weight
        
        # All diffs should have the same shape (verified during computation)
        ref_diff = diffs_with_weights[0][0]
        device = ref_diff.device
        dtype = ref_diff.dtype
        
        if mode == "weighted_average":
            result = torch.zeros_like(ref_diff, dtype=torch.float32)
            total_weight = sum(abs(w) for _, w in diffs_with_weights)
            if total_weight == 0:
                return result.to(dtype)
            for diff, weight in diffs_with_weights:
                result += diff.to(device=device, dtype=torch.float32) * (weight / total_weight)
            return result.to(dtype)
        
        elif mode == "weighted_sum":
            result = torch.zeros_like(ref_diff, dtype=torch.float32)
            for diff, weight in diffs_with_weights:
                result += diff.to(device=device, dtype=torch.float32) * weight
            return result.to(dtype)
        
        elif mode == "normalize":
            # Normalization by "energy" (sum of squared weights)
            weights = [w for _, w in diffs_with_weights]
            sum_sq = sum(w*w for w in weights)
            if sum_sq == 0:
                return torch.zeros_like(ref_diff)
            scale = 1.0 / math.sqrt(sum_sq)
            
            result = torch.zeros_like(ref_diff, dtype=torch.float32)
            for diff, weight in diffs_with_weights:
                result += diff.to(device=device, dtype=torch.float32) * weight * scale
            return result.to(dtype)
        
        return None
    
    def true_merge(self, model, clip, merge_mode, output_strength,
                   lora_1, strength_1, lora_2, strength_2,
                   lora_3="None", strength_3=1.0,
                   lora_4="None", strength_4=1.0,
                   clip_strength_multiplier=1.0):
        """
        Main function for true LoRA merging.
        """
        
        # Collect active LoRAs
        loras_data = []
        for lora_name, strength in [
            (lora_1, strength_1),
            (lora_2, strength_2),
            (lora_3, strength_3),
            (lora_4, strength_4),
        ]:
            if lora_name != "None" and lora_name is not None and strength != 0:
                lora = self._load_lora(lora_name)
                if lora is not None:
                    loras_data.append((lora_name, lora, strength))
        
        if len(loras_data) == 0:
            return (model, clip)
        
        logging.info(f"Z-Image LoRA True Merge: {merge_mode} mode, {len(loras_data)} LoRAs")
        for name, _, strength in loras_data:
            logging.info(f"  - {name}: strength={strength}")
        
        # Get key_map for model
        model_keys = {}
        if model is not None:
            model_keys = comfy.lora.model_lora_keys_unet(model.model, {})
        
        clip_keys = {}
        if clip is not None:
            clip_keys = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, {})
        
        # Collect all keys from all LoRAs
        all_lora_prefixes = set()
        for _, lora_dict, _ in loras_data:
            for key in lora_dict.keys():
                # Extract prefix (before .lora_up, .lora_down, etc.)
                for suffix in [".lora_up.weight", ".lora_down.weight", "_lora.up.weight", 
                              "_lora.down.weight", ".lora_B.weight", ".lora_A.weight",
                              ".lora.up.weight", ".lora.down.weight", ".alpha"]:
                    if key.endswith(suffix):
                        prefix = key[:-len(suffix)]
                        all_lora_prefixes.add(prefix)
                        break
        
        # For each key, compute merged diff
        merged_patches = {}
        processed_keys = 0
        
        for lora_prefix in all_lora_prefixes:
            # Find target key in model
            target_key = None
            is_clip = False
            
            if lora_prefix in model_keys:
                target_key = model_keys[lora_prefix]
            elif lora_prefix in clip_keys:
                target_key = clip_keys[lora_prefix]
                is_clip = True
            else:
                # Try to find directly
                for k in model_keys:
                    if lora_prefix in k or k in lora_prefix:
                        target_key = model_keys[k]
                        break
            
            if target_key is None:
                continue
            
            # Handle tuple keys (for sliced weights)
            actual_key = target_key[0] if isinstance(target_key, tuple) else target_key
            
            # Get target weight shape
            try:
                if is_clip:
                    target_weight = comfy.utils.get_attr(clip.cond_stage_model, actual_key)
                else:
                    target_weight = comfy.utils.get_attr(model.model, actual_key)
                target_shape = target_weight.shape
            except:
                continue
            
            # Compute diff for each LoRA
            diffs_with_weights = []
            for _, lora_dict, strength in loras_data:
                lora_info = self._get_lora_key_info(lora_dict, lora_prefix)
                if lora_info is None:
                    continue
                
                mat_up, mat_down, alpha, mid = lora_info
                diff = self._compute_lora_diff(mat_up, mat_down, alpha, mid, target_shape)
                
                if diff is not None:
                    diffs_with_weights.append((diff, strength))
            
            if len(diffs_with_weights) == 0:
                continue
            
            # Merge diffs
            merged_diff = self._merge_diffs(diffs_with_weights, merge_mode)
            if merged_diff is not None:
                merged_patches[target_key] = ("diff", (merged_diff,))
                processed_keys += 1
        
        logging.info(f"  Processed {processed_keys} weight keys")
        
        # Apply to model
        new_model = model
        new_clip = clip
        
        if model is not None and len(merged_patches) > 0:
            new_model = model.clone()
            # Filter patches for model
            model_patches = {k: v for k, v in merged_patches.items() 
                           if not isinstance(k, str) or not k.startswith("clip")}
            new_model.add_patches(model_patches, output_strength)
        
        if clip is not None and len(merged_patches) > 0:
            new_clip = clip.clone()
            clip_strength = output_strength * clip_strength_multiplier
            # Apply all patches to clip (key filtering is more complex)
            new_clip.add_patches(merged_patches, clip_strength)
        
        return (new_model, new_clip)


# Node registration
NODE_CLASS_MAPPINGS = {
    "ZImageLoRAMerger": ZImageLoRAMerger,
    "ZImageLoRAStack": ZImageLoRAStack,
    "ZImageLoRAStackApply": ZImageLoRAStackApply,
    "ZImageLoRAMergeToSingle": ZImageLoRAMergeToSingle,
    "ZImageLoRATrueMerge": ZImageLoRATrueMerge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageLoRAMerger": "Z-Image LoRA Merger",
    "ZImageLoRAStack": "Z-Image LoRA Stack",
    "ZImageLoRAStackApply": "Z-Image LoRA Stack Apply",
    "ZImageLoRAMergeToSingle": "Z-Image LoRA Merge to Single",
    "ZImageLoRATrueMerge": "Z-Image LoRA True Merge",
}
