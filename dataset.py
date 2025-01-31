import tiktoken
tokenizer_subword = tiktoken.get_encoding("gpt2")

from typing import Any
from torch.utils.data import Dataset, DataLoader
from torch import tensor, arange

GPTTokenizer = tokenizer_subword

class GPTDatasetV1(Dataset):

    def __init__(self, txt, context_size, stride, tokenizer):
        self.input_ids = []
        self.output_ids = []
        ## tokenizing the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        ## looping over the sequences
        for i in range(0, len(token_ids) - context_size, stride):
            input_chunk = token_ids[i:i+context_size]
            target_chunk = token_ids[i+1:i+context_size+1]
            self.input_ids.append(tensor(input_chunk))
            self.output_ids.append(tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {"x":self.input_ids[idx],
                "y":self.output_ids[idx]}


def get_dataloader(txt, batch_size = 4, context_size = 1024, stride = 512,
                   shuffle = True, drop_last = True, num_workers = 0):
    '''Creates a dataloader object'''
    ## loading the sub-word tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    ## creating dataset oject
    dataset = GPTDatasetV1(txt, tokenizer=tokenizer_subword,
                           context_size=context_size, stride=stride)
    ## creating the dataloader object
    dataloader = DataLoader(dataset,batch_size=batch_size,
                            drop_last=drop_last,
                            num_workers=num_workers,
                            shuffle = shuffle)
    return dataloader