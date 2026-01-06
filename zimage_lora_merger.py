"""
Z-Image Turbo Multi-LoRA Merger
Кастомная нода для решения проблемы "выжженности" при использовании нескольких LoRA
на дистилированных моделях типа z-image turbo.

Проблема: стандартное применение LoRA суммирует эффекты аддитивно, что приводит
к перенасыщению на дистилированных моделях.

Решение: различные стратегии смешивания LoRA для нормализации суммарного эффекта.
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
    Нода для объединения нескольких LoRA с различными стратегиями смешивания,
    оптимизированная для z-image turbo и других дистилированных моделей.
    """
    
    BLEND_MODES = [
        "normalize",      # Нормализует суммарную силу к target_strength
        "average",        # Усредняет эффекты LoRA
        "sqrt_scale",     # Масштабирует каждую LoRA на 1/sqrt(n)
        "linear_decay",   # Линейное уменьшение силы: 1, 0.5, 0.33, ...
        "geometric_decay",# Геометрическое уменьшение: 1, 0.5, 0.25, ...
        "additive",       # Стандартное аддитивное (для сравнения)
    ]
    
    def __init__(self):
        self.loaded_loras = {}
    
    @classmethod
    def INPUT_TYPES(cls):
        lora_list = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Модель для применения LoRA"}),
                "clip": ("CLIP", {"tooltip": "CLIP модель"}),
                "blend_mode": (cls.BLEND_MODES, {"default": "normalize", "tooltip": "Режим смешивания LoRA"}),
                "target_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, 
                                              "tooltip": "Целевая суммарная сила (для normalize/average)"}),
                "lora_1": (["None"] + lora_list, {"tooltip": "Первая LoRA"}),
                "strength_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "lora_2": (["None"] + lora_list, {"tooltip": "Вторая LoRA"}),
                "strength_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
            },
            "optional": {
                "lora_3": (["None"] + lora_list, {"tooltip": "Третья LoRA"}),
                "strength_3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "lora_4": (["None"] + lora_list, {"tooltip": "Четвертая LoRA"}),
                "strength_4": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "lora_5": (["None"] + lora_list, {"tooltip": "Пятая LoRA"}),
                "strength_5": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "clip_strength_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                                       "tooltip": "Множитель силы для CLIP"}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "merge_loras"
    CATEGORY = "loaders/lora"
    DESCRIPTION = "Объединяет несколько LoRA с нормализацией для z-image turbo и других дистилированных моделей"
    
    def _load_lora(self, lora_name):
        """Загружает LoRA файл с кэшированием"""
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
        Вычисляет коэффициенты смешивания для каждой LoRA
        в зависимости от выбранного режима.
        """
        n = len(strengths)
        if n == 0:
            return []
        
        if blend_mode == "additive":
            # Стандартное аддитивное применение
            return strengths
        
        elif blend_mode == "normalize":
            # Нормализует так, чтобы сумма квадратов сил = target_strength^2
            # Это сохраняет "энергию" эффекта
            sum_sq = sum(s*s for s in strengths)
            if sum_sq == 0:
                return [0.0] * n
            scale = target_strength / math.sqrt(sum_sq)
            return [s * scale for s in strengths]
        
        elif blend_mode == "average":
            # Простое усреднение с учетом target_strength
            scale = target_strength / n
            return [s * scale for s in strengths]
        
        elif blend_mode == "sqrt_scale":
            # Масштабирует на 1/sqrt(n) - сохраняет баланс при добавлении LoRA
            scale = 1.0 / math.sqrt(n)
            return [s * scale for s in strengths]
        
        elif blend_mode == "linear_decay":
            # Каждая последующая LoRA имеет меньший вес: 1, 0.5, 0.33, 0.25, ...
            factors = [1.0 / (i + 1) for i in range(n)]
            total = sum(factors)
            return [strengths[i] * factors[i] * target_strength / total for i in range(n)]
        
        elif blend_mode == "geometric_decay":
            # Геометрическое убывание: 1, 0.5, 0.25, 0.125, ...
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
        Основная функция объединения LoRA.
        """
        
        # Собираем активные LoRA
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
        
        # Вычисляем коэффициенты смешивания
        original_strengths = [s for _, s in loras_data]
        blended_strengths = self._calculate_blend_factors(
            original_strengths, blend_mode, target_strength
        )
        
        logging.info(f"Z-Image LoRA Merger: {blend_mode} mode, {len(loras_data)} LoRAs")
        logging.info(f"  Original strengths: {original_strengths}")
        logging.info(f"  Blended strengths:  {[round(s, 4) for s in blended_strengths]}")
        
        # Применяем LoRA последовательно с вычисленными коэффициентами
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
    Нода для создания стека LoRA, который можно использовать с ZImageLoRAStackApply.
    Позволяет более гибко управлять LoRA через соединения.
    """
    
    def __init__(self):
        self.loaded_loras = {}
    
    @classmethod
    def INPUT_TYPES(cls):
        lora_list = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "lora_name": (lora_list, {"tooltip": "LoRA файл"}),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
            },
            "optional": {
                "lora_stack": ("LORA_STACK", {"tooltip": "Предыдущий стек LoRA"}),
            }
        }
    
    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("lora_stack",)
    FUNCTION = "add_to_stack"
    CATEGORY = "loaders/lora"
    DESCRIPTION = "Добавляет LoRA в стек для последующего применения с ZImageLoRAStackApply"
    
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
    Применяет стек LoRA с различными режимами смешивания.
    """
    
    BLEND_MODES = ZImageLoRAMerger.BLEND_MODES
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Модель для применения LoRA"}),
                "clip": ("CLIP", {"tooltip": "CLIP модель"}),
                "lora_stack": ("LORA_STACK", {"tooltip": "Стек LoRA"}),
                "blend_mode": (cls.BLEND_MODES, {"default": "normalize"}),
                "target_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "clip_strength_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "apply_stack"
    CATEGORY = "loaders/lora"
    DESCRIPTION = "Применяет стек LoRA с выбранным режимом смешивания"
    
    def _calculate_blend_factors(self, strengths, blend_mode, target_strength):
        """Используем ту же логику что и в ZImageLoRAMerger"""
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
    Объединяет веса нескольких LoRA в одну "виртуальную" LoRA
    путем предварительного слияния весов перед применением к модели.
    
    Это может дать лучшие результаты чем последовательное применение,
    особенно для перекрывающихся весов.
    """
    
    MERGE_METHODS = [
        "weighted_sum",    # Взвешенная сумма
        "add_difference",  # A + (B - C) * weight
        "weighted_average",# Взвешенное среднее
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
                                              "tooltip": "Сила итогового объединенного LoRA"}),
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
    DESCRIPTION = "Объединяет несколько LoRA в одну перед применением - оптимально для z-image turbo"
    
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
        Объединяет веса LoRA в один словарь.
        Обрабатывает LoRA с разным рангом - несовместимые слои берутся 
        пропорционально весам из той LoRA, которая их имеет.
        """
        if len(loras_with_weights) == 0:
            return {}
        
        if len(loras_with_weights) == 1:
            lora, weight = loras_with_weights[0]
            return {k: v * weight for k, v in lora.items()}
        
        # Собираем все ключи
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
            
            # Получаем референсную форму и dtype
            ref_tensor = tensors_weights[0][0]
            ref_shape = ref_tensor.shape
            device = ref_tensor.device
            dtype = ref_tensor.dtype
            
            # Фильтруем только тензоры с совместимыми формами
            compatible_tensors = [(t, w) for t, w in tensors_weights if t.shape == ref_shape]
            
            # Если формы не совпадают, берём взвешенную сумму только совместимых
            # или первый тензор если совместимых нет
            if len(compatible_tensors) < len(tensors_weights):
                skipped_keys += 1
                if len(compatible_tensors) == 0:
                    # Нет совместимых - берём первый с его весом
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
                # Первый LoRA + разница с остальными
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
    "Честное" слияние LoRA через вычисление полных diff'ов.
    
    Работает для LoRA ЛЮБЫХ рангов!
    
    Вместо объединения сырых A/B матриц (что невозможно для разных рангов),
    вычисляет полный diff = A @ B × alpha для каждой LoRA,
    затем усредняет эти diff'ы и применяет как один патч.
    
    ⚠️ Требует больше памяти и времени чем обычное применение.
    """
    
    MERGE_MODES = [
        "weighted_average",  # Взвешенное среднее diff'ов
        "weighted_sum",      # Взвешенная сумма (может быть ярче)
        "normalize",         # Нормализация по энергии
    ]
    
    def __init__(self):
        self.loaded_loras = {}
    
    @classmethod
    def INPUT_TYPES(cls):
        lora_list = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Модель для применения LoRA"}),
                "clip": ("CLIP", {"tooltip": "CLIP модель"}),
                "merge_mode": (cls.MERGE_MODES, {"default": "weighted_average"}),
                "output_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                              "tooltip": "Сила итогового объединённого эффекта"}),
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
    DESCRIPTION = "Честное слияние LoRA любых рангов через вычисление полных diff'ов"
    
    def _load_lora(self, lora_name):
        """Загружает LoRA файл с кэшированием"""
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
        Извлекает информацию о LoRA для заданного ключа.
        Возвращает (mat_up, mat_down, alpha, mid) или None.
        """
        # Форматы LoRA ключей
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
                    alpha = mat_down.shape[0]  # rank как дефолт
                
                # Mid (для LoCon)
                mid_key = "{}.lora_mid.weight".format(key_prefix)
                mid = lora_dict.get(mid_key, None)
                
                return (mat_up, mat_down, alpha, mid)
        
        return None
    
    def _compute_lora_diff(self, mat_up, mat_down, alpha, mid, target_shape):
        """
        Вычисляет полный diff для одной LoRA.
        diff = mat_up @ mat_down × (alpha / rank)
        """
        rank = mat_down.shape[0]
        scale = alpha / rank
        
        if mid is not None:
            # LoCon с mid матрицей (редко)
            final_shape = [mat_down.shape[1], mat_down.shape[0], mid.shape[2], mid.shape[3]]
            mat_down = (
                torch.mm(
                    mat_down.transpose(0, 1).flatten(start_dim=1),
                    mid.transpose(0, 1).flatten(start_dim=1),
                )
                .reshape(final_shape)
                .transpose(0, 1)
            )
        
        # Вычисляем diff
        diff = torch.mm(
            mat_up.flatten(start_dim=1).float(),
            mat_down.flatten(start_dim=1).float()
        )
        
        # Пытаемся привести к целевой форме
        try:
            diff = diff.reshape(target_shape)
        except RuntimeError:
            # Если форма не совпадает, пропускаем
            return None
        
        return diff * scale
    
    def _merge_diffs(self, diffs_with_weights, mode):
        """
        Объединяет список diff'ов с их весами.
        """
        if len(diffs_with_weights) == 0:
            return None
        
        if len(diffs_with_weights) == 1:
            diff, weight = diffs_with_weights[0]
            return diff * weight
        
        # Все diff'ы должны быть одной формы (проверено при вычислении)
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
            # Нормализация по "энергии" (сумме квадратов весов)
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
        Основная функция честного слияния LoRA.
        """
        
        # Собираем активные LoRA
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
        
        # Получаем key_map для модели
        model_keys = {}
        if model is not None:
            model_keys = comfy.lora.model_lora_keys_unet(model.model, {})
        
        clip_keys = {}
        if clip is not None:
            clip_keys = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, {})
        
        # Собираем все ключи из всех LoRA
        all_lora_prefixes = set()
        for _, lora_dict, _ in loras_data:
            for key in lora_dict.keys():
                # Извлекаем префикс (до .lora_up, .lora_down и т.д.)
                for suffix in [".lora_up.weight", ".lora_down.weight", "_lora.up.weight", 
                              "_lora.down.weight", ".lora_B.weight", ".lora_A.weight",
                              ".lora.up.weight", ".lora.down.weight", ".alpha"]:
                    if key.endswith(suffix):
                        prefix = key[:-len(suffix)]
                        all_lora_prefixes.add(prefix)
                        break
        
        # Для каждого ключа вычисляем объединённый diff
        merged_patches = {}
        processed_keys = 0
        
        for lora_prefix in all_lora_prefixes:
            # Находим целевой ключ в модели
            target_key = None
            is_clip = False
            
            if lora_prefix in model_keys:
                target_key = model_keys[lora_prefix]
            elif lora_prefix in clip_keys:
                target_key = clip_keys[lora_prefix]
                is_clip = True
            else:
                # Пробуем найти напрямую
                for k in model_keys:
                    if lora_prefix in k or k in lora_prefix:
                        target_key = model_keys[k]
                        break
            
            if target_key is None:
                continue
            
            # Handle tuple keys (for sliced weights)
            actual_key = target_key[0] if isinstance(target_key, tuple) else target_key
            
            # Получаем форму целевого веса
            try:
                if is_clip:
                    target_weight = comfy.utils.get_attr(clip.cond_stage_model, actual_key)
                else:
                    target_weight = comfy.utils.get_attr(model.model, actual_key)
                target_shape = target_weight.shape
            except:
                continue
            
            # Вычисляем diff для каждой LoRA
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
            
            # Объединяем diff'ы
            merged_diff = self._merge_diffs(diffs_with_weights, merge_mode)
            if merged_diff is not None:
                merged_patches[target_key] = ("diff", (merged_diff,))
                processed_keys += 1
        
        logging.info(f"  Processed {processed_keys} weight keys")
        
        # Применяем к модели
        new_model = model
        new_clip = clip
        
        if model is not None and len(merged_patches) > 0:
            new_model = model.clone()
            # Фильтруем патчи для модели
            model_patches = {k: v for k, v in merged_patches.items() 
                           if not isinstance(k, str) or not k.startswith("clip")}
            new_model.add_patches(model_patches, output_strength)
        
        if clip is not None and len(merged_patches) > 0:
            new_clip = clip.clone()
            clip_strength = output_strength * clip_strength_multiplier
            # Все патчи применяем к clip (фильтрация по ключам сложнее)
            new_clip.add_patches(merged_patches, clip_strength)
        
        return (new_model, new_clip)


# Регистрация нод
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

