"""
This script performs binding and unbinding operations using the FlatBindingModel and BtspModel classes.

The script follows the following steps:
1. Set up the necessary parameters for the binding and unbinding operations.
2. Load the optimal threshold values for the FlatBindingModel.
3. Generate a dataset of words based on the specified parameters.
4. Create an instance of the FlatBindingModel class.
5. Perform binding operation on the generated dataset.
6. Perform top-down unbinding operation and calculate the error.
7. Load the optimal threshold value for the bottom-up unbinding operation.
8. Create a masked input for the bottom-up unbinding operation.
9. Perform bottom-up unbinding operation and calculate the error.
10. Clean up the top-down unbinding result using the BTSP-based CAM.
11. Calculate the error of the top-down unbinding after clean-up.
12. Clean up the bottom-up unbinding result using the BTSP-based CAM.
13. Calculate the error of the bottom-up unbinding after clean-up.

Note: The script uses the FlatBindingModel and BtspModel classes from the binding_models module and the generate_words_based_dataset and masked_input_w_two_cues functions from the binding_data module.
"""
import torch, math
from binding_models import BaseBindingModel, FlatBindingModel, BtspModel
from binding_data import generate_words_based_dataset, masked_input_w_two_cues

# Set up the necessary parameters for the binding and unbinding operations.
input_size, mem_size = 48000, 12000
fp, fq, fw, L, K, Nw = 0.005, 0.005, 0.6, 6000, 8, 1000
num_sentences = 5000
device = 'cuda:0'


# Load the optimal threshold values for the FlatBindingModel.
thr_ca1_scale = 0.67
opt_thr_ca1 = int(K * L * fp * fw * thr_ca1_scale) # 96
# print(f"opt_thr_ca1: {opt_thr_ca1}")
thr_ca3_search_path = "thresholds_flat_binding.pt"
thr_ca3_pt = torch.load(thr_ca3_search_path)
thr_ca3_pt_key = f"num={num_sentences}, thr_ca1_scale={thr_ca1_scale}, fp={fp}, fq={fq}"
opt_thr_ca3 = thr_ca3_pt[thr_ca3_pt_key]
# print(f"opt_thr_ca3: {opt_thr_ca3}")

# Generate a dataset of words based on the specified parameters.
data, voc = generate_words_based_dataset(num_sentences, Nw, L, K, fp, device)

# Create an instance of the FlatBindingModel class.
binding_model = FlatBindingModel(input_size, mem_size, fq, fp, fw, L, K, Nw, device)

# Perform binding operation on the generated dataset.
binding_model.model.thr_ca1, binding_model.model.thr_ca3 = opt_thr_ca1, opt_thr_ca3
# print(f"binding_model.model.thr_ca1: {binding_model.model.thr_ca1}")

composed_representation = binding_model.binding(data)
# print(f"composed_representation: {composed_representation}")

# Perform top-down unbinding operation and calculate the error.
td_rec = binding_model.top_down_unbinding(composed_representation)
target = data
td_err = ((td_rec - target).abs().mean(1) / target.mean(1)).mean(0)
# td_err = (td_rec - target).abs().mean(1).mean(0) / target.mean(1).mean(0)
print(f"error of top-down unbinding: {td_err * 100:.4f} %")

# Load the optimal threshold value for the bottom-up unbinding operation.
thr_ca1_pt_key = f"K={K}, cue_idx={[0,1]}, num={num_sentences}, thr_ca1_scale={thr_ca1_scale}, L={6000}, fp={fp}, fq={fq}"
opt_thr_ca1 = thr_ca3_pt[thr_ca1_pt_key]
opt_thr_ca1 = 27
print(f"opt_thr_ca1 during BU unbinding: {opt_thr_ca1}")
binding_model.model.thr_ca1 = opt_thr_ca1

# Create a masked input for the bottom-up unbinding operation.
masked_input = masked_input_w_two_cues(data, K)

# Perform bottom-up unbinding operation and calculate the error.
bu_rec = binding_model.bottom_up_unbinding(masked_input)
print(f"(masked_input.mean(1).mean(0)): {masked_input.mean(1).mean(0)}")
target = data
bu_err = ((bu_rec - target).abs().mean(1) / target.mean(1)).mean(0)
print(f"error of bottom-up unbinding: {bu_err * 100:.4f} %")

del binding_model


# Training BTSP-based CAM on vocabulary
CAM_opt_thr_ca1 = int(L * fp * fw * thr_ca1_scale)
print(f"opt_thr_ca1_CAM: {CAM_opt_thr_ca1}")

thr_ca3_pt_key = f"K={K}, num={num_sentences}, case_CAM, thr_ca1_scale={thr_ca1_scale}, L={L}, fp={fp}, fq={fq}"
CAM_opt_thr_ca3 = thr_ca3_pt[thr_ca3_pt_key]
print(f"opt_thr_ca3_CAM: {CAM_opt_thr_ca3}")


CAM = BtspModel(L, 12000, fq, fp, fw, device)
CAM.thr_ca1, CAM.thr_ca3 = CAM_opt_thr_ca1, CAM_opt_thr_ca3
CAM.__one_shot_learning__(voc)

# Clean up the top-down unbinding result using the BTSP-based CAM.
rec = torch.cat(torch.chunk(td_rec, K, dim=1), dim=0)
target = torch.cat(torch.chunk(data, K, dim=1), dim=0)
rec = CAM.forward_backward(rec)
td_err = ((rec - target).abs().mean(1) / target.mean(1)).mean(0)
print(f"error of top-down unbinding after clean-up: {td_err * 100:.4f} %")

# Clean up the bottom-up unbinding result using the BTSP-based CAM.
rec = torch.cat(torch.chunk(bu_rec, K, dim=1), dim=0)
target = torch.cat(torch.chunk(data, K, dim=1), dim=0)
rec = CAM.forward_backward(rec)
bu_err = ((rec - target).abs().mean(1) / target.mean(1)).mean(0)
print(f"error of bottom-up unbinding after clean-up: {bu_err * 100:.4f} %")

del CAM



