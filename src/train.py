import torch
from torch.utils.data import DataLoader
import yaml
from src.dataset import BanglaDialectDataset
from src.tokenizer import get_tokenizer
from src.model import BanglaDialectEmbeddingModel
from tqdm import tqdm

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train():
    config = load_config()

    torch.manual_seed(config["training"]["seed"])

    tokenizer = get_tokenizer()

    train_dataset = BanglaDialectDataset(
        data_path=config["data"]["train_path"],
        tokenizer=tokenizer,
        max_seq_len=config["data"]["max_seq_len"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )

    model = BanglaDialectEmbeddingModel()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"]
    )

    model.train()
    for epoch in range(config["training"]["epochs"]):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            optimizer.zero_grad()
            embeddings = model(input_ids, attention_mask)

            # Placeholder loss — actual loss Sunday te decide hobe
            loss = embeddings.mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} — Loss: {total_loss:.4f}")

if __name__ == "__main__":
    train()