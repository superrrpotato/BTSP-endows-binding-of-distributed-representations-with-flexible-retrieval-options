# Behavioral Time Scale Synaptic Plasticity (BTSP) endows binding of distributed representations with flexible retrieval options

This repository provides PyTorch implementations of BTSP-inspired models for **binding and flexible retrieval** of distributed representations.  
It includes three variants:

- **FlatBindingModel**
- **HierarchicalBindingModel**
- **OnlineBindingModel**

Each variant supports **top-down** and **bottom-up unbinding**, and all rely on **BTSP-style one-shot learning** with CA1/CA3-like forward and backward projections.  
Dataset generation and cue-masking utilities are provided for reproducible synthetic experiments.

---

## 1. System requirements

### Software dependencies
| Package | Minimum version | Purpose |
|----------|----------------|----------|
| Python | 3.9 | Core language |
| PyTorch | 2.0 | Tensor operations and CUDA |
| NumPy | 1.24 | Array operations |
| Matplotlib *(optional)* | 3.8 | Visualization |
| Jupyter *(optional)* | 1.0 | Notebook demo |

Only `torch` and `numpy` are strictly required for the provided scripts.

### Operating systems
| OS | Supported | Tested |
|----|------------|---------|
| Ubuntu Linux 20.04–22.04 | ✅ | ✅ |
| Windows 10/11 (WSL2 recommended) | ✅ | ✅ |
| macOS (Intel/Apple Silicon) | ⚠️ | Untested (may require CPU-only PyTorch build) |

### Versions tested
| Component | Version | Notes |
|------------|----------|-------|
| Python | 3.10.13 | Development environment |
| PyTorch | 2.3.0 (CUDA 12.1) | GPU runtime |
| NumPy | 1.26.4 | Standard scientific stack |

### Required non-standard hardware
- **NVIDIA GPU with CUDA** (≥ 8 GB VRAM recommended).  
  The code directly calls `.cuda()` during tensor creation, so a CUDA-capable GPU and CUDA-enabled PyTorch are required.
- CPU-only mode is possible if you manually replace `.cuda()` with `.to(device)` and set `device='cpu'`.

---

## 2. Installation guide

### Instructions
```bash
# Clone the repository
git clone https://github.com/superrrpotato/BTSP-endows-binding-of-distributed-representations-with-flexible-retrieval-options
cd BTSP-endows-binding-of-distributed-representations-with-flexible-retrieval-options

# (Optional) create a clean environment
python -m venv btsp_env
source btsp_env/bin/activate      # Linux/Mac
# .\btsp_env\Scripts\activate     # Windows

# Install dependencies
pip install torch numpy
# Optional
pip install matplotlib jupyter
```

### Typical install time
≈ **2–4 minutes** on a normal desktop (PyTorch installation dominates).

---

## 3. Demo

### Files
- `binding_models.py` — model definitions (`BtspModel`, `FlatBindingModel`, `HierarchicalBindingModel`, `OnlineBindingModel`)
- `binding_data.py` — dataset and masking utilities
- `thresholds_flat_binding.pt` — precomputed thresholds for the flat demo
- `flat_binding.py`, `hierarchical_binding.py`, `online_binding.py` — example scripts
- `VSA.ipynb` — optional notebook walkthrough

> Scripts default to `device='cuda:0'`. Ensure `torch.cuda.is_available()`.

### Run the demos

#### Flat binding
```bash
python flat_binding.py
```
Expected console output pattern:
```
error of top-down unbinding: <value> %
opt_thr_ca1 during BU unbinding: <value>
error of bottom-up unbinding: <value> %
opt_thr_ca1_CAM: <value>
opt_thr_ca3_CAM: <value>
error of top-down unbinding after clean-up: <value> %
error of bottom-up unbinding after clean-up: <value> %
```

#### Hierarchical binding
```bash
python hierarchical_binding.py
```
Expected output:
```
error of top-down unbinding: <value> %
error of top-down unbinding after clean-up: <value> %
```

#### Online binding
```bash
python online_binding.py
```
Expected output:
```
error of top-down unbinding: <value> %
error of top-down unbinding after clean-up: <value> %
```

### Expected run time on a normal desktop (GPU)
| Model | num_sentences | Runtime |
|--------|----------------|----------|
| FlatBindingModel | 5000 | ~1–2 min |
| HierarchicalBindingModel | 2000 | ~1 min |
| OnlineBindingModel | 2000 | ~1 min |

CPU-only mode (after removing `.cuda()`): ~4–8 minutes depending on settings.

---

## 4. Instructions for use

### Generate a dataset
```python
from binding_data import generate_words_based_dataset

data, voc = generate_words_based_dataset(
    num_sentences=5000,  # number of samples
    Nw=1000,             # vocabulary size
    L=6000,              # word length
    K=8,                 # words per sentence
    fp=0.005,            # fraction of ones
    device='cuda:0'      # computation device
)
```

### Create masked input with two cues
```python
from binding_data import masked_input_w_two_cues
masked_input = masked_input_w_two_cues(data, K=8)
```

### Initialize and run a model
```python
import torch
from binding_models import FlatBindingModel

input_size, mem_size = 48000, 12000
fp, fq, fw, L, K, Nw = 0.005, 0.005, 0.6, 6000, 8, 1000
device = 'cuda:0'

binding_model = FlatBindingModel(input_size, mem_size, fq, fp, fw, L, K, Nw, device)

# Load thresholds from thresholds_flat_binding.pt
thr_ca1_scale = 0.67
opt_thr_ca1 = int(K * L * fp * fw * thr_ca1_scale)
thr_ca3_pt = torch.load("thresholds_flat_binding.pt")
thr_ca3_key = f"num={5000}, thr_ca1_scale={thr_ca1_scale}, fp={fp}, fq={fq}"
opt_thr_ca3 = thr_ca3_pt[thr_ca3_key]

binding_model.model.thr_ca1 = opt_thr_ca1
binding_model.model.thr_ca3 = opt_thr_ca3

# Binding
composed = binding_model.binding(data)

# Top-down unbinding
td_rec = binding_model.top_down_unbinding(composed)
```

### Compute reconstruction error
```python
target = data
td_err = ((td_rec - target).abs().mean(1) / target.mean(1)).mean(0)
print(f"error of top-down unbinding: {td_err * 100:.4f} %")
```

### Bottom-up unbinding (with masked input)
```python
bu_rec = binding_model.bottom_up_unbinding(masked_input)
bu_err = ((bu_rec - target).abs().mean(1) / target.mean(1)).mean(0)
print(f"error of bottom-up unbinding: {bu_err * 100:.4f} %")
```

### Optional: CAM cleanup
```python
from binding_models import BtspModel

CAM = BtspModel(L, 12000, fq, fp, fw, device)
CAM.__one_shot_learning__(voc)

rec = torch.cat(torch.chunk(td_rec, K, dim=1), dim=0)
target = torch.cat(torch.chunk(data, K, dim=1), dim=0)
rec = CAM.forward_backward(rec)
td_err = ((rec - target).abs().mean(1) / target.mean(1)).mean(0)
print(f"error of top-down unbinding after clean-up: {td_err * 100:.4f} %")
```

---

## (Optional) Reproduction instructions

To reproduce results:

1. Use the same parameters as the provided scripts:  
   `fp=fq=0.005`, `fw=0.6`, `L=6000–12000`, `K=8`, `Nw=1000`, `num_sentences=2000–5000`
2. Run sequentially:
   ```bash
   python flat_binding.py
   python hierarchical_binding.py
   python online_binding.py
   ```
3. Compare printed errors before and after CAM cleanup.  
   Lower errors after cleanup confirm correct reproduction.

---

## License
MIT License. See `LICENSE` file.
