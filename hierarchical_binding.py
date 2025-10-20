import torch, math
from binding_models import HierarchicalBindingModel, BtspModel
from binding_data import generate_words_based_dataset

input_size, mem_size = 24000, 12000
fp, fq, fw, L, K, Nw = 0.005, 0.005, 0.6, 12000, 8, 1000
num_sentences = 2000
device = 'cuda:0'





data, voc = generate_words_based_dataset(num_sentences, Nw, L, K, fp, device)
binding_model = HierarchicalBindingModel(input_size, mem_size, fq, fp, fw, L, K, Nw, device)

binding_model.model.CA1_WTA, binding_model.model.CA3_WTA = True, True


composed_representation = binding_model.binding(data)

td_rec = binding_model.top_down_unbinding(composed_representation)
target = data
td_err = ((td_rec - target).abs().mean(1) / target.mean(1)).mean(0)
print(f"error of top-down unbinding: {td_err * 100:.4f} %")


del binding_model



CAM = BtspModel(L, 24000, fq, fp, fw, device)
CAM.CA1_WTA, CAM.CA3_WTA = True, True
CAM.__one_shot_learning__(voc)

rec = torch.cat(torch.chunk(td_rec, K, dim=1), dim=0)
target = torch.cat(torch.chunk(data, K, dim=1), dim=0)
rec = CAM.forward_backward(rec)
td_err = ((rec - target).abs().mean(1) / target.mean(1)).mean(0)
print(f"error of top-down unbinding after clean-up: {td_err * 100:.4f} %")

del CAM



