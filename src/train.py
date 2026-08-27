import torch
import yaml
import os
from tqdm import tqdm

from src.dataset import build_dataloaders
from src.tokenizer import get_tokenizer
from src.model import BanglaDialectEmbeddingModel, ContrastiveLoss, TripletLoss, BanglaBERTBaseline, Gemma2Baseline

# ──────────────────────────────────────────────
# Config loader
# ──────────────────────────────────────────────

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────
# Validation loop
# ──────────────────────────────────────────────

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            standard_ids   = batch["standard_ids"].to(device)

            # Dialect sentence embedding
            dialect_emb  = model(input_ids, attention_mask)

            # Standard Bangla embedding
            # attention_mask for standard — all ones (no padding assumed)
            std_mask     = (standard_ids != 0).long()
            standard_emb = model(standard_ids, std_mask)

            loss = criterion(dialect_emb, standard_emb)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss


# ──────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────

def train():
    # 1. Load config
    config = load_config()
    torch.manual_seed(config["training"]["seed"])

    # 2. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 3. Tokenizer
    tokenizer = get_tokenizer()

    # 4. DataLoaders
    train_loader, val_loader, test_loader = build_dataloaders(
        excel_path   = config["data"]["excel_path"],
        tokenizer    = tokenizer,
        max_seq_len  = config["data"]["max_seq_len"],
        batch_size   = config["training"]["batch_size"],
        train_ratio  = config["data"].get("train_ratio", 0.8),
        val_ratio    = config["data"].get("val_ratio",   0.1),
        seed         = config["training"]["seed"],
    )

        # 5. Model
    model_cfg = config.get('model', {})
    architecture = model_cfg.get('architecture', 'mamba2')

    if architecture == 'banglabert':
        model = BanglaBERTBaseline(
            embed_dim   = model_cfg.get('embed_dim', 128),
            dropout     = model_cfg.get('dropout', 0.1),
            freeze_bert = model_cfg.get('freeze_bert', True),
        ).to(device)
    elif architecture == 'gemma2':
        model = Gemma2Baseline(
            embed_dim    = model_cfg.get('embed_dim', 128),
            dropout      = model_cfg.get('dropout', 0.1),
            freeze_gemma = model_cfg.get('freeze_gemma', True),
            model_name   = model_cfg.get('gemma_model', 'google/gemma-2-2b'),
        ).to(device)
    else:
        model = BanglaDialectEmbeddingModel(
            vocab_size   = model_cfg.get('vocab_size', 101975),
            d_model      = model_cfg.get('d_model', 128),
            n_layers     = model_cfg.get('n_layers', 2),
            embed_dim    = model_cfg.get('embed_dim', 128),
            d_state      = model_cfg.get('d_state', 16),
            d_conv       = model_cfg.get('d_conv', 4),
            expand       = model_cfg.get('expand', 2),
            headdim      = model_cfg.get('headdim', 64),
            dropout      = model_cfg.get('dropout', 0.1),
            architecture = architecture,
        ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 6. Loss & Optimizer
    loss_type = config["training"].get("loss", "nt_xent")

    if loss_type == "triplet":
        criterion = TripletLoss(margin=config["training"].get("margin", 1.0))
    else:
        criterion = ContrastiveLoss(temperature=config["training"].get("temperature", 0.07))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = config["training"]["learning_rate"],
        weight_decay = config["training"].get("weight_decay", 1e-2),
    )

    # 7. LR Scheduler (optional — cosine decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max = config["training"]["epochs"],
    )

    # 8. Checkpoint directory
    ckpt_dir = config["training"].get("checkpoint_dir", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val_loss = float("inf")

    # ── Training epochs ───────────────────────
    for epoch in range(1, config["training"]["epochs"] + 1):
        model.train()
        total_train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{config['training']['epochs']}"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            standard_ids   = batch["standard_ids"].to(device)

            # Forward pass — dialect embedding
            dialect_emb  = model(input_ids, attention_mask)

            # Forward pass — standard Bangla embedding
            std_mask     = (standard_ids != 0).long()
            standard_emb = model(standard_ids, std_mask)

            # Contrastive loss
            loss = criterion(dialect_emb, standard_emb)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()

        scheduler.step()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss   = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
            torch.save({
                "epoch":      epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss":   avg_val_loss,
            }, ckpt_path)
            print(f"  ✓ Best model saved → {ckpt_path}")

    print("\nTraining complete.")
    print(f"Best Val Loss: {best_val_loss:.4f}")


# ──────────────────────────────────────────────

if __name__ == "__main__":
    train()