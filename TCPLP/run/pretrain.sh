CUDA_VISIBLE_DEVICES=2,3 python lightning_pretrain.py \
    --model_name_or_path /data/model/longformer-base-4096/ \
    --longformer_ckpt /data/longformer_ckpt/longformer-base-4096_ckpt.bin \
    --train_file ./pretrain_train.json \
    --dev_file ./pretrain_val.json \
    --item_attr_file ./OR_Pantry_Instruments_meta_data.json \
    --output_dir ./result/pretrain_OR_Pantry_Instrument/ \
    --num_train_epochs 32 \
    --valid_step 900 \
    --gradient_accumulation_steps 8 \
    --preprocessing_num_workers 8 \
    --dataloader_num_workers 8  \
    --batch_size 12 \
    --learning_rate 5e-5 \
    --temp 0.05 \
    --device 2 \
    --fix_word_embedding \
    --p_content 2 \



