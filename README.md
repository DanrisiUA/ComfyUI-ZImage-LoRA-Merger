# Z-Image LoRA Merger for ComfyUI

Custom nodes for combining multiple LoRAs **without the "burned/overexposed" look** on distilled models like **Z-Image Turbo**, **SDXL-Turbo**, **LCM** and similar.

![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom_Node-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 🔥 The Problem

When you chain multiple LoRAs in ComfyUI, their effects are **added together**:

```
model += lora1_effect × strength1
model += lora2_effect × strength2
Total effect = strength1 + strength2  ← Can exceed 1.0!
```

On distilled/turbo models this causes **overexposure and artifacts** because these models are already optimized for fewer inference steps and can't handle the accumulated LoRA effects.

## ✨ The Solution

This pack provides **5 nodes** with different strategies:

### 1. Z-Image LoRA Merger
Applies multiple LoRAs with **automatic strength normalization**.

| Mode | What it does |
|------|-------------|
| `normalize` | Keeps total "energy" (sum of squares) at target level — **recommended** |
| `average` | Divides each strength by number of LoRAs |
| `sqrt_scale` | Scales by 1/√n — good for independent effects |
| `linear_decay` | First LoRA strongest, others progressively weaker |
| `geometric_decay` | Aggressive decay: 1, 0.5, 0.25, 0.125... |
| `additive` | Standard behavior (for comparison) |

**Example:**
```
Input:  LoRA1=0.6, LoRA2=1.0
Mode:   normalize, target=1.0

Output: LoRA1=0.51, LoRA2=0.86  (total "energy" normalized)
```

### 2. Z-Image LoRA True Merge ⭐ 
**Properly merges LoRAs of ANY rank!**

Standard merging can't combine LoRAs with different ranks (e.g., rank-32 + rank-256).
This node computes the **full weight diff** for each LoRA first, then averages them:

```
Standard: A₁[×32] + A₂[×256] = ❌ ERROR (different shapes)

True Merge:
  diff1 = A₁ @ B₁ × alpha → [4096×4096] ✓
  diff2 = A₂ @ B₂ × alpha → [4096×4096] ✓  
  merged = average(diff1, diff2) → Works! ✓
```

⚠️ Uses more memory and is slower, but gives true averaging for any rank combination.

### 3-4. Z-Image LoRA Stack + Stack Apply
Flexible node-based LoRA chaining with blend modes.

### 5. Z-Image LoRA Merge to Single
Merges LoRA weights before applying (works best with same-rank LoRAs).

## 📊 Comparison

| Method | Different Ranks | Memory | Speed | Best For |
|--------|----------------|--------|-------|----------|
| Standard chaining | ✅ | Low | Fast | Can overexpose |
| **LoRA Merger** (normalize) | ✅ | Low | Fast | Most cases |
| **LoRA True Merge** | ✅ | High | Slow | Mixed ranks |
| Merge to Single | ❌ Same rank | Medium | Medium | Same rank LoRAs |

## 🎯 Recommended Settings

### For Z-Image Turbo / Distilled Models:
1. Use **Z-Image LoRA Merger** with `normalize` mode
2. Set `target_strength` to **0.7-0.9**
3. If still overexposed, try `sqrt_scale`

### For LoRAs with Different Ranks:
1. Use **Z-Image LoRA True Merge**
2. Mode: `weighted_average`
3. Adjust `output_strength` as needed

## 📦 Installation

### Option 1: ComfyUI Manager (Recommended)
Search for "Z-Image LoRA Merger" in ComfyUI Manager and install.

### Option 2: Manual Installation
1. Navigate to your `ComfyUI/custom_nodes/` folder
2. Clone this repository:
```bash
git clone https://github.com/DanrisiUA/ComfyUI-ZImage-LoRA-Merger.git
```
3. Restart ComfyUI

### Option 3: Download ZIP
1. Download this repository as ZIP
2. Extract to `ComfyUI/custom_nodes/ComfyUI-ZImage-LoRA-Merger`
3. Restart ComfyUI

## 🖼️ Nodes

After installation, you'll find these nodes in the `loaders/lora` category:

- **Z-Image LoRA Merger** — Main node with blend modes
- **Z-Image LoRA True Merge** — For different rank LoRAs
- **Z-Image LoRA Stack** — Build LoRA chains
- **Z-Image LoRA Stack Apply** — Apply LoRA chains
- **Z-Image LoRA Merge to Single** — Pre-merge LoRA weights

## 📝 Blend Modes Explained

### normalize (recommended)
Normalizes so that sum of squared strengths equals `target_strength²`.
Preserves relative proportions while controlling total effect.

```python
scale = target_strength / √(Σstrength²)
new_strength[i] = strength[i] × scale
```

### average
Simply divides each strength by number of LoRAs.
```python
new_strength[i] = strength[i] × (target_strength / n)
```

### sqrt_scale
Scales by 1/√n — mathematically sound for independent effects.
```python
new_strength[i] = strength[i] / √n
```

### linear_decay
First LoRA gets full weight, others progressively less: 1, 1/2, 1/3, 1/4...

### geometric_decay
Aggressive decay: 1, 0.5, 0.25, 0.125...
Good when you want one dominant LoRA.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 🙏 Credits

- Developed by [DanrisiUA](https://github.com/DanrisiUA)
- For the ComfyUI community

---

**If this helped you, please ⭐ star the repo!**

