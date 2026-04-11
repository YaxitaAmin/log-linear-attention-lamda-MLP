"""
Data loading utilities for language modeling experiments.
Handles TinyShakespeare dataset loading and tokenization.
"""

import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np


class TinyShakespeareDataset(IterableDataset):
    """
    Streaming iterable dataset for TinyShakespeare with tokenization.
    Yields flattened sequences of tokens with context windows.
    """
    
    def __init__(
        self,
        split: str = "train",
        tokenizer_name: str = "gpt2",
        seq_len: int = 1024,
        batch_size: int = 32,
        device: str = "cuda",
    ):
        """
        Args:
            split: "train" or "validation"
            tokenizer_name: HuggingFace tokenizer name
            seq_len: Sequence length for each sample
            batch_size: Batch size (used for internal buffering)
            device: "cuda" or "cpu"
        """
        self.split = split
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device
        
        # Load dataset
        self.dataset = load_dataset("tiny_shakespeare")
        self.text = self.dataset[split]["text"][0]  # Single large text
        
        # Tokenize entire dataset
        self.tokens = self.tokenizer.encode(self.text, add_special_tokens=False)
        self.tokens = torch.tensor(self.tokens, dtype=torch.long, device=device)
        
        self.num_samples = len(self.tokens) // seq_len
    
    def __iter__(self):
        """Yield sequences of tokenized text."""
        for i in range(self.num_samples):
            start = i * self.seq_len
            end = start + self.seq_len
            
            input_ids = self.tokens[start:end].unsqueeze(0)  # (1, seq_len)
            labels = self.tokens[start+1:end+1].unsqueeze(0)  # Shift for LM
            
            yield {"input_ids": input_ids, "labels": labels}
    
    def __len__(self):
        return self.num_samples


def get_tiny_shakespeare_dataloader(
    split: str = "train",
    seq_len: int = 1024,
    batch_size: int = 32,
    num_workers: int = 0,
    device: str = "cuda",
) -> DataLoader:
    """
    Create a DataLoader for TinyShakespeare.
    
    Args:
        split: "train" or "validation"
        seq_len: Sequence length per sample
        batch_size: Batch size
        num_workers: Number of data loading workers
        device: "cuda" or "cpu"
        
    Returns:
        DataLoader for TinyShakespeare
    """
    dataset = TinyShakespeareDataset(
        split=split,
        tokenizer_name="gpt2",
        seq_len=seq_len,
        batch_size=batch_size,
        device=device,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )
    
    return dataloader, dataset.num_samples
