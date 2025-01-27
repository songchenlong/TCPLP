# Readme--TCPLP
Semantic-enhanced Co-attention Prompt Learning for Non-overlapping Cross-Domain Recommendation

## Dataset
https://drive.google.com/drive/folders/1hC4IVJgzYJzYaFRGqb3W0qH_vg-mjaZJ?usp=drive_link

## Requirements
···
Python 3.10.10
PyTorch 2.0.0
PyTorch Lightning 2.0.0
Transformers 4.28.0
Deepspeed 0.9.0
···

## Usage
1. Using `python save_longformer_ckpt.py` to adjust pretrained Longformer checkpoint to the model.
2. Using `bash pretrain.sh` to perform the pre-training stage.
3. If you use the training strategy deepspeed_stage_2 (default setting in the script), you need to first convert zero checkpoint to lightning checkpoint by running `python zero_to_fp32.py` (automatically generated to checkpoint folder from pytorch-lightning).
4. Using `python convert_pretrain_ckpt.py` to convert the lightning checkpoint to pytorch checkpoint.
5. Using `bash prompt_tuing.sh` to perform the prompt-tuning stage.

## Acknowledgement
https://github.com/AaronHeee/RecFormer
