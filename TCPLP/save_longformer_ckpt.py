import torch
import os
from transformers import LongformerForMaskedLM
from tcplp_longformer import TcplpForPretraining, TcplpConfig

longformer = LongformerForMaskedLM.from_pretrained('/data/model/longformer-base-4096/')

config = TcplpConfig.from_pretrained('/data/model/longformer-base-4096/')
config.max_item_embeddings = 51
config.max_attr_length = 32
config.attention_window = [64] * 12
config.p_content = 2

    
model = TcplpForPretraining(config)

longformer_state_dict = longformer.state_dict()
tcplp_state_dict = model.state_dict()
for name, param in longformer_state_dict.items():
    if name not in tcplp_state_dict:
        print('missing name', name)
        continue
    else:
        try:
            if not tcplp_state_dict[name].size()==param.size():
                print(name)
                print(tcplp_state_dict[name].size())
                print(param.size())
            tcplp_state_dict[name].copy_(param)
        except:
            print('wrong size', name)

for name, param in longformer_state_dict.items():
    if name not in tcplp_state_dict:
        print('missing name', name)
        continue
    if not torch.all(param == tcplp_state_dict[name]):
        print(name)

torch.save(tcplp_state_dict, '/data/longformer_ckpt/longformer-base-4096_ckpt.bin')
# python save_longformer_ckpt.py