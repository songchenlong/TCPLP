import torch
from collections import OrderedDict
from tcplp_longformer import TcplpModel, TcplpConfig, TcplpForFinetune

PRETRAINED_CKPT_PATH = '' # pretrain_model.bin
LONGFORMER_CKPT_PATH = '/data/longformer_ckpt/longformer-base-4096_cpkt.bin'
LONGFORMER_TYPE = '/data/model/longformer-base-4096/'
TCPLP_OUTPUT_PATH = './TcplpForFinetune.bin'

input_file = PRETRAINED_CKPT_PATH
state_dict = torch.load(input_file)

longformer_file = LONGFORMER_CKPT_PATH
longformer_state_dict = torch.load(longformer_file)

if not torch.all(state_dict['model.longformer.embeddings.word_embeddings.weight'] == longformer_state_dict['longformer.embeddings.word_embeddings.weight']):
    raise ValueError("Error!!!!!!!!!!")
state_dict['_forward_module.model.longformer.embeddings.word_embeddings.weight'] = longformer_state_dict['longformer.embeddings.word_embeddings.weight']

output_file = TCPLP_OUTPUT_PATH

new_state_dict = OrderedDict()
for key, value in state_dict.items():
    if key.startswith('_forward_module.model.'):
        new_key = key[len('_forward_module.model.'):]
        new_state_dict[new_key] = value
    elif key.startswith('model.'):
        new_key = key[len('model.'):]
        new_state_dict[new_key] = value


config = TcplpConfig.from_pretrained(LONGFORMER_TYPE)
config.max_item_embeddings = 51
config.attention_window = [64] * 12
config.p_content = 2

model = TcplpForFinetune(config)

model.load_state_dict(new_state_dict, strict=False)

print('Convert successfully.')
torch.save(new_state_dict, output_file)
