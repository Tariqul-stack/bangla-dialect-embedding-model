import torch
import torch.nn.functional as F
import yaml
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import build_dataloaders
from src.tokenizer import get_tokenizer
from src.model import BanglaDialectEmbeddingModel


# ──────────────────────────────────────────────
# Config loader
# ──────────────────────────────────────────────

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────
# Cosine Similarity Evaluation
# ──────────────────────────────────────────────

def evaluate_similarity(model, test_loader, device):
    """
    প্রতিটা dialect sentence এবং তার standard Bangla equivalent-এর
    cosine similarity measure করে।

    High similarity → model ভালোভাবে শিখেছে যে দুটো same অর্থ বহন করে।
    Low similarity  → model এখনো শেখেনি।

    Returns:
        avg_pos_sim : positive pair average similarity  (higher is better)
        avg_neg_sim : negative pair average similarity  (lower is better)
    """
    model.eval()

    pos_similarities = []  # same sentence, different dialect
    neg_similarities = []  # different sentences

    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            standard_ids   = batch["standard_ids"].to(device)

            # Dialect embedding
            dialect_emb  = model(input_ids, attention_mask)

            # Standard Bangla embedding
            std_mask     = (standard_ids != 0).long()
            standard_emb = model(standard_ids, std_mask)

            # L2 normalize
            d = F.normalize(dialect_emb,  dim=-1)  # [B, D]
            s = F.normalize(standard_emb, dim=-1)  # [B, D]

            # Positive similarity: diagonal (same sentence pair)
            pos_sim = (d * s).sum(dim=-1)           # [B]
            pos_similarities.extend(pos_sim.cpu().tolist())

            # Negative similarity: off-diagonal (different sentences)
            sim_matrix = d @ s.T                    # [B, B]
            B = d.size(0)
            mask = ~torch.eye(B, dtype=torch.bool, device=device)
            neg_sim = sim_matrix[mask]              # [B*(B-1)]
            neg_similarities.extend(neg_sim.cpu().tolist())

    avg_pos_sim = sum(pos_similarities) / len(pos_similarities)
    avg_neg_sim = sum(neg_similarities) / len(neg_similarities)

    return avg_pos_sim, avg_neg_sim


# ──────────────────────────────────────────────
# Per-dialect similarity breakdown
# ──────────────────────────────────────────────

def evaluate_per_dialect(model, test_loader, device):
    """
    প্রতিটা dialect আলাদাভাবে কতটা similar সেটা দেখায়।
    """
    from collections import defaultdict

    model.eval()
    dialect_sims = defaultdict(list)

    LABEL_MAP = {
        0: "standard",
        1: "rajshahi",
        2: "sylheti",
        3: "chittagong",
        4: "rangpur",
        5: "mymensingh",
        6: "barishal",
        7: "rakhain",
    }

    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            standard_ids   = batch["standard_ids"].to(device)
            dialect_labels = batch["dialect_label"]

            dialect_emb  = model(input_ids, attention_mask)
            std_mask     = (standard_ids != 0).long()
            standard_emb = model(standard_ids, std_mask)

            d = F.normalize(dialect_emb,  dim=-1)
            s = F.normalize(standard_emb, dim=-1)

            pos_sim = (d * s).sum(dim=-1).cpu().tolist()

            for sim, label in zip(pos_sim, dialect_labels.tolist()):
                dialect_name = LABEL_MAP.get(label, "unknown")
                dialect_sims[dialect_name].append(sim)

    return {
        dialect: sum(sims) / len(sims)
        for dialect, sims in dialect_sims.items()
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def evaluate():
    config = load_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Tokenizer & DataLoaders
    tokenizer = get_tokenizer()
    _, _, test_loader = build_dataloaders(
        excel_path  = config["data"]["excel_path"],
        tokenizer   = tokenizer,
        max_seq_len = config["data"]["max_seq_len"],
        batch_size  = config["training"]["batch_size"],
        train_ratio = config["data"].get("train_ratio", 0.8),
        val_ratio   = config["data"].get("val_ratio",   0.1),
        seed        = config["training"]["seed"],
    )

    # Load model
    model = BanglaDialectEmbeddingModel().to(device)
    ckpt_path = config["training"].get("checkpoint_dir", "checkpoints") + "/best_model.pt"

    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(val loss: {checkpoint['val_loss']:.4f})\n")

    # ── Overall similarity ──────────────────────
    avg_pos, avg_neg = evaluate_similarity(model, test_loader, device)

    print("=" * 45)
    print("        SIMILARITY EVALUATION RESULTS")
    print("=" * 45)
    print(f"  Positive pair similarity : {avg_pos:.4f}  ↑ higher is better")
    print(f"  Negative pair similarity : {avg_neg:.4f}  ↓ lower is better")
    print(f"  Similarity gap           : {avg_pos - avg_neg:.4f}")
    print("=" * 45)

    # ── Per-dialect breakdown ───────────────────
    print("\nPer-dialect similarity (vs standard Bangla):\n")
    per_dialect = evaluate_per_dialect(model, test_loader, device)
    for dialect, sim in sorted(per_dialect.items(), key=lambda x: -x[1]):
        bar = "█" * int(sim * 20)
        print(f"  {dialect:<12} : {sim:.4f}  {bar}")

    print()


if __name__ == "__main__":
    evaluate()