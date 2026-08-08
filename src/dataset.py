import torch
from torch.utils.data import Dataset, DataLoader

from data_preprocessing import load_excel_dataset, flatten_to_samples


# Label → integer mapping for all 8 dialects
DIALECT_LABEL_MAP = {
    "standard":   0,
    "rajshahi":   1,
    "sylheti":    2,
    "chittagong": 3,
    "rangpur":    4,
    "mymensingh": 5,
    "barishal":   6,
    "rakhain":    7,
}


class BanglaDialectDataset(Dataset):
    """
    Dialect-aware Bangla parallel corpus dataset.

    Each sample returns:
        input_ids       – tokenized dialect sentence        [max_seq_len]
        attention_mask  – padding mask                      [max_seq_len]
        dialect_label   – integer label (0-7)               scalar
        standard_ids    – tokenized standard Bangla equiv.  [max_seq_len]

    Args:
        samples   : list of dicts from flatten_to_samples()
        tokenizer : any HuggingFace tokenizer
        max_seq_len: maximum token length (default 128)
    """

    def __init__(self, samples: list[dict], tokenizer, max_seq_len: int = 128):
        self.samples     = samples
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # Tokenize the dialect sentence
        encoded = self._tokenize(sample["text"])

        # Tokenize the standard Bangla equivalent
        standard_encoded = self._tokenize(sample["standard"])

        # Dialect integer label
        dialect_label = DIALECT_LABEL_MAP.get(sample["dialect"], 0)

        return {
            "input_ids":      encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "dialect_label":  torch.tensor(dialect_label, dtype=torch.long),
            "standard_ids":   standard_encoded["input_ids"].squeeze(0),
        }

    # ------------------------------------------------------------------
    def _tokenize(self, text: str) -> dict:
        return self.tokenizer(
            text,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )


# ──────────────────────────────────────────────
# Factory function  ← easy to call from train.py
# ──────────────────────────────────────────────

def build_dataloaders(
    excel_path: str,
    tokenizer,
    max_seq_len:  int   = 128,
    batch_size:   int   = 32,
    train_ratio:  float = 0.8,
    val_ratio:    float = 0.1,
    seed:         int   = 42,
):
    """
    Load the Excel dataset, split into train/val/test,
    and return three DataLoaders.

    Split ratio: 80% train | 10% val | 10% test  (default)

    Returns:
        train_loader, val_loader, test_loader
    """
    # 1. Load & flatten
    df      = load_excel_dataset(excel_path)
    samples = flatten_to_samples(df)

    print(f"Total samples (all dialects): {len(samples)}")

    # 2. Shuffle
    import random
    random.seed(seed)
    random.shuffle(samples)

    # 3. Split
    n       = len(samples)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train_samples = samples[:n_train]
    val_samples   = samples[n_train : n_train + n_val]
    test_samples  = samples[n_train + n_val :]

    print(f"Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")

    # 4. Build Dataset objects
    train_ds = BanglaDialectDataset(train_samples, tokenizer, max_seq_len)
    val_ds   = BanglaDialectDataset(val_samples,   tokenizer, max_seq_len)
    test_ds  = BanglaDialectDataset(test_samples,  tokenizer, max_seq_len)

    # 5. Build DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader