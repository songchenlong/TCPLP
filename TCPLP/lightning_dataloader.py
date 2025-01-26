from typing import List
from torch.utils.data import Dataset
from collator import PretrainDataCollatorWithPadding
import json

class ClickDataset(Dataset):
    def __init__(self, dataset_path: str, collator: PretrainDataCollatorWithPadding):
        super().__init__()
        self.dataset_path = dataset_path

        with open(self.dataset_path, 'r') as file:
            data = json.load(file)
        self.data = list(data.values())

        self.collator = collator

    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]

    def collate_fn(self, data):
        return self.collator([{'items': line} for line in data])





