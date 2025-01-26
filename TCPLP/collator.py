from typing import Optional, Union, List, Dict, Tuple, Type
from dataclasses import dataclass
# from tcplp import TcplpTokenizer
import torch
import unicodedata
import random


# Data collator
@dataclass
class PretrainDataCollatorWithPadding:

    tokenizer: Type
    tokenized_items: Dict
    # mlm_probability: float
    mode: str
   
    def __call__(self, batch_item_ids: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        '''
        features: A batch of list of item ids
        1. sample training pairs
        2. convert item ids to item features
        # 3. mask tokens for mlm

        input_ids: (batch_size, seq_len)
        item_position_ids: (batch_size, seq_len)
        token_type_ids: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len)
        global_attention_mask: (batch_size, seq_len)
        '''
        
        batch_item_seq_a, batch_item_seq_b = self.sample_pairs(batch_item_ids)
        batch_feature_a = self.extract_features(batch_item_seq_a)
        batch_feature_b = self.extract_features(batch_item_seq_b)

        batch_encode_features_a = self.encode_features(batch_feature_a)
        batch_encode_features_b = self.encode_features(batch_feature_b)
        batch_a = self.tokenizer.padding(batch_encode_features_a, pad_to_max=False)
        batch_b = self.tokenizer.padding(batch_encode_features_b, pad_to_max=False)

        # batch_a["mlm_input_ids"], batch_a["mlm_labels"] = self.mask_mlm(batch_encode_features_a)
        # batch_b["mlm_input_ids"], batch_b["mlm_labels"] = self.mask_mlm(batch_encode_features_b)

        batch = dict()

        for k, v in batch_a.items():
            batch[k+'_a'] = torch.LongTensor(v)
        
        for k, v in batch_b.items():
            batch[k+'_b'] = torch.LongTensor(v)

        # batch_key:  input_ids_a, item_position_ids_a, token_type_ids_a, attention_mask_a, global_attention_mask_a, mlm_input_ids_a, mlm_labels_a; input_ids_b, item_position_ids_b, token_type_ids_b, attention_mask_b, global_attention_mask_b, mlm_input_ids_b, mlm_labels_b;

        return batch


    def sample_pairs(self, batch_item_ids):
        batch_item_seq_a = []
        batch_item_seq_b = []

        for item_ids in batch_item_ids:
            item_ids = item_ids['items']
            item_seq_len = len(item_ids)
            start = (item_seq_len-1) // 2
            if self.mode == 'train':
                target_pos = random.randint(start, item_seq_len-1) #左闭右闭
            elif self.mode == 'val':
                target_pos = item_seq_len-1
            batch_item_seq_a.append(item_ids[:target_pos])
            batch_item_seq_b.append([item_ids[target_pos]])


        return batch_item_seq_a, batch_item_seq_b


    def extract_features(self, batch_item_seq):

        features = []
        for item_seq in batch_item_seq:
            feature_seq = []
            for item in item_seq:
                input_ids = self.tokenized_items[item]
                feature_seq.append(input_ids)
            features.append(feature_seq)

        return features

 
    def encode_features(self, batch_feature):
        features = []
        for feature in batch_feature:
            features.append(self.tokenizer.encode(feature, encode_item=False))

        return features

 
    def _collate_batch(self, examples, pad_to_multiple_of: Optional[int] = None):
        """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
        # Tensorize if necessary.
        if isinstance(examples[0], (list, tuple)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]

        # Check if padding is necessary.
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
            return torch.stack(examples, dim=0)

        # If yes, check if we have a `pad_token`.
        if self.tokenizer._pad_token is None:
            raise ValueError(
                "You are attempting to pad samples but the tokenizer you are using"
                f" ({self.tokenizer.__class__.__name__}) does not have a pad token."
            )

        # Creating the full tensor and filling it with our data.
        max_length = max(x.size(0) for x in examples)
        if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
        result = examples[0].new_full([len(examples), max_length], self.tokenizer.pad_token_id)
        for i, example in enumerate(examples):
            if self.tokenizer.padding_side == "right":
                result[i, : example.shape[0]] = example
            else:
                result[i, -example.shape[0] :] = example
        return result


@dataclass
class FinetuneDataCollatorWithPadding:

    tokenizer: Type
    tokenized_items: Dict
    item2id_new: Dict
    def __call__(self, batch_item_ids: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        '''
        features: A batch of list of item ids
        1. sample training pairs
        2. convert item ids to item features
        3. mask tokens for mlm

        input_ids: (batch_size, seq_len)
        item_position_ids: (batch_size, seq_len)
        token_type_ids: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len)
        global_attention_mask: (batch_size, seq_len)
        '''
        
        batch_item_seq, labels = self.sample_train_data(batch_item_ids)

        batch_feature = self.extract_features(batch_item_seq)
        batch_encode_features = self.encode_features(batch_feature)
        batch = self.tokenizer.padding(batch_encode_features, pad_to_max=False)
        batch["labels"] = labels

        for k, v in batch.items():
            batch[k] = torch.LongTensor(v)
        
        return batch

    def sample_train_data(self, batch_item_ids):
        batch_item_seq = []
        labels = []

        for item_ids in batch_item_ids:
            item_ids = item_ids['items']
            items = []
            for item in item_ids:
                # items.append(int(self.item2id_new[item]))
                items.append(self.item2id_new[item])
            item_seq_len = len(items)
            batch_item_seq.append(items[-50:-1])
            labels.append(items[-1])

        return batch_item_seq, labels


    def extract_features(self, batch_item_seq):
        features = []
        for item_seq in batch_item_seq:
            feature_seq = []
            for item in item_seq:
                # input_ids = self.tokenized_items[str(item)]
                input_ids = self.tokenized_items[item]
                feature_seq.append(input_ids)
            features.append(feature_seq)

        return features

    def encode_features(self, batch_feature):
        features = []
        for feature in batch_feature:
            features.append(self.tokenizer.encode(feature, encode_item=False))

        return features


@dataclass
class EvalDataCollatorWithPadding:
    tokenizer: Type
    tokenized_items: Dict
    item2id_new: Dict
    def __call__(self, batch_data: List[Dict[str, Union[int, List[int], List[List[int]], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        batch_item_seq, labels = self.prepare_eval_data(batch_data)
        batch_feature = self.extract_features(batch_item_seq)
        batch_encode_features = self.encode_features(batch_feature)
        batch = self.tokenizer.padding(batch_encode_features, pad_to_max=False)

        for k, v in batch.items():
            batch[k] = torch.LongTensor(v)

        labels = torch.LongTensor(labels)
        
        return batch, labels

    def prepare_eval_data(self, batch_data):
        batch_item_seq = []
        labels = []
        for data_line in batch_data:
            item_ids = data_line['items']
            items = []
            for item in item_ids:
                items.append(self.item2id_new[item])
            label = data_line['label']
            label = self.item2id_new[label]
            batch_item_seq.append(items)
            labels.append(int(label))
        return batch_item_seq, labels


    def extract_features(self, batch_item_seq):
        features = []
        for item_seq in batch_item_seq:
            feature_seq = []
            for item in item_seq:
                input_ids = self.tokenized_items[item]
                feature_seq.append(input_ids)
            features.append(feature_seq)

        return features

    def encode_features(self, batch_feature):
        
        features = []
        for feature in batch_feature:
            features.append(self.tokenizer.encode(feature, encode_item=False))

        return features


    tokenizer: Type
    tokenized_items_s: Dict
    tokenized_items_t: Dict
    item2id_s: Dict
    item2id_t: Dict

    def __call__(self, batch_data: List[Dict[str, Union[int, List[int], List[List[int]], torch.Tensor]]]) -> Dict[
        str, torch.Tensor]:

        batch_item_seq, labels = self.prepare_eval_data(batch_data)
        batch_feature = self.extract_features(batch_item_seq)
        batch_encode_features = self.encode_features(batch_feature)
        batch = self.tokenizer.padding(batch_encode_features, pad_to_max=False)

        for k, v in batch.items():
            batch[k] = torch.LongTensor(v)
        labels = torch.LongTensor(labels)

        return batch, labels

    def prepare_eval_data(self, batch_data):
        batch_item_seq = []
        labels = []
        for data_line in batch_data:
            item_ids = data_line['items']
            items = []
            for item in item_ids:
                items.append(self.item2id_s[item])
            label = data_line['label']
            label = self.item2id_t[label]
            batch_item_seq.append(items)
            labels.append(label)
        return batch_item_seq, labels

    def extract_features(self, batch_item_seq):
        features = []
        for item_seq in batch_item_seq:
            feature_seq = []
            for item in item_seq:
                input_ids = self.tokenized_items_s[item]
                feature_seq.append(input_ids)
            features.append(feature_seq)

        return features

    def encode_features(self, batch_feature):

        features = []
        for feature in batch_feature:
            features.append(self.tokenizer.encode(feature, encode_item=False))

        return features