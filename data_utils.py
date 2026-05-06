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
        labels = self.tokens[start + 1:end + 1]
        return {"input_ids": input_ids, "labels": labels}


def get_wikitext103_dataloader(split, seq_len, batch_size, device):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)

    # Tokenize article-by-article to avoid passing one giant string to the
    # tokenizer (which triggers the "sequence length > max_length" warning and
    # can cause indexing errors).
    all_ids = []
    for text in dataset["text"]:
        if text.strip():
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            all_ids.extend(ids)

    tokens = torch.tensor(all_ids, dtype=torch.long)
    ds = LMDataset(tokens, seq_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"))
    return loader, len(ds)
