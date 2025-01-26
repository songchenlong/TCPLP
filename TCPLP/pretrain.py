import os
from torch.utils.tensorboard import SummaryWriter
import logging
from torch.utils.data import DataLoader
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm
import json
import argparse
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from tcplp_longformer import TcplpForPretraining, TcplpTokenizer, TcplpConfig, LitWrapper
# from tcplp_bert import TcplpForPretraining, TcplpTokenizer, TcplpConfig, LitWrapper

from collator import PretrainDataCollatorWithPadding
from lightning_dataloader import ClickDataset

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, default=None)
parser.add_argument('--temp', type=float, default=0.05, help="Temperature for softmax.")
parser.add_argument('--preprocessing_num_workers', type=int, default=8, help="The number of processes to use for the preprocessing.") #预处理工作的并行进程数。
parser.add_argument('--train_file', type=str)
parser.add_argument('--dev_file', type=str)
parser.add_argument('--item_attr_file', type=str)
parser.add_argument('--output_dir', type=str)

parser.add_argument('--num_train_epochs', type=int, default=10)
parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
parser.add_argument('--dataloader_num_workers', type=int, default=2)
parser.add_argument('--mlm_probability', type=float, default=0.15)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=5e-5)
parser.add_argument('--valid_step', type=int, default=3000)
parser.add_argument('--log_step', type=int, default=3000)
parser.add_argument('--device', type=int, default=1)
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--longformer_ckpt', type=str, default='/data/longformer_ckpt/longformer-base-4096.bin')
parser.add_argument('--fix_word_embedding', action='store_true')
parser.add_argument('--p_content', type=int, default=2)




tokenizer_glb: TcplpTokenizer = None
def _par_tokenize_doc(doc):
    item_id, item_attr = doc
    input_ids = tokenizer_glb.encode_item(item_attr)

    return item_id, input_ids


def main():
    args = parser.parse_args()
    print(args)
    seed_everything(42)

    config = TcplpConfig.from_pretrained(args.model_name_or_path)
    config.max_attr_length = 32
    config.max_item_embeddings = 51  # 50 item and 1 for cls
    config.attention_window = [64] * 12
    config.max_token_num = 1024
    config.p_content = args.p_content

    tokenizer = TcplpTokenizer.from_pretrained(args.model_name_or_path, config)

    global tokenizer_glb
    tokenizer_glb = tokenizer

    path_corpus = Path(args.item_attr_file)

    print(f'Loading attribute data {path_corpus}')
    item_attrs = json.load(open(path_corpus))
    pool = Pool(processes=args.preprocessing_num_workers)
    pool_func = pool.imap(func=_par_tokenize_doc, iterable=item_attrs.items())
    doc_tuples = list(tqdm(pool_func, total=len(item_attrs), ncols=100, desc=f'[Tokenize] {path_corpus}'))
    tokenized_items = {item_id: input_ids for item_id, input_ids in doc_tuples}
    pool.close()
    pool.join()
    print(f'Successfully load {len(tokenized_items)} tokenized items.')

    train_data_collator = PretrainDataCollatorWithPadding(tokenizer, tokenized_items, mode='train')
    eval_data_collator = PretrainDataCollatorWithPadding(tokenizer, tokenized_items, mode='val')

    train_data = ClickDataset(args.train_file, train_data_collator)
    dev_data = ClickDataset(args.dev_file, eval_data_collator)

    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              collate_fn=train_data.collate_fn,
                              num_workers=args.dataloader_num_workers,
                              drop_last=True)
    dev_loader = DataLoader(dev_data,
                            batch_size=args.batch_size,
                            collate_fn=dev_data.collate_fn,
                            num_workers=args.dataloader_num_workers,
                            drop_last=True
                            )


    pytorch_model = TcplpForPretraining(config)
    pytorch_model.load_state_dict(torch.load(args.longformer_ckpt))


    if args.fix_word_embedding:
        print('Fix word embeddings.')
        for param in pytorch_model.longformer.embeddings.word_embeddings.parameters():
            param.requires_grad = False

  
    model = LitWrapper(pytorch_model, learning_rate=args.learning_rate)
 
    checkpoint_callback = ModelCheckpoint(save_top_k=5, monitor="accuracy", mode="max", filename="{epoch}-{accuracy:.4f}")

    trainer = Trainer(accelerator="gpu",
                     max_epochs=args.num_train_epochs,
                     devices=args.device,
                     accumulate_grad_batches=args.gradient_accumulation_steps,
                     val_check_interval=args.valid_step,
                     default_root_dir=args.output_dir,
                     gradient_clip_val=1.0,
                     log_every_n_steps=args.log_step,
                     precision=16 if args.fp16 else 32,
                     strategy='deepspeed_stage_2',
                     callbacks=[checkpoint_callback]
                     )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=dev_loader, ckpt_path=args.ckpt)


if __name__ == "__main__":
    main()