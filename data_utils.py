import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer

class LMDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return (len(self.tokens) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        input_ids = self.tokens[start:end]
        labels = self.tokens[start+1:end+1]
        return {"input_ids": input_ids, "labels": labels}

def get_wikitext103_dataloader(split, seq_len, batch_size, device):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    text = "\n".join(dataset["text"])
    tokens = tokenizer(text, return_tensors="pt")["input_ids"].squeeze()
    ds = LMDataset(tokens, seq_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=(split=="train"))
    return loader, len(ds)
