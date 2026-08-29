"""
distill.py — Knowledge Distillation Training
=============================================
Teacher: Mamba3 + Triplet Loss (best model)
Student: StudentDialectModel (lightweight, ~4MB)

Loss = Triplet Loss + α * MSE(student_emb, teacher_emb)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import os
from tqdm import tqdm

from src.dataset import build_dataloaders
from src.tokenizer import get_tokenizer
from src.model import (
    BanglaDialectEmbeddingModel,
    StudentDialectModel,
    TripletLoss,
)


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def validate_student(student, teacher, val_loader, criterion, device, alpha=0.5):
    student.eval()
    teacher.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            standard_ids   = batch["standard_ids"].to(device)
            std_mask       = (standard_ids != 0).long()

            # Student embeddings
            student_dialect  = student(input_ids, attention_mask)
            student_standard = student(standard_ids, std_mask)

            # Teacher embeddings (frozen)
            teacher_dialect  = teacher(input_ids, attention_mask)
            teacher_standard = teacher(standard_ids, std_mask)

            # Task loss
            task_loss = criterion(student_dialect, student_standard)

            # Distillation loss
            distill_loss = F.mse_loss(student_dialect, teacher_dialect) + \
                           F.mse_loss(student_standard, teacher_standard)

            loss = task_loss + alpha * distill_loss
            total_loss += loss.item()

    return total_loss / len(val_loader)


def distill():
    config = load_config()
    torch.manual_seed(config["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Tokenizer + DataLoaders
    tokenizer = get_tokenizer()
    train_loader, val_loader, test_loader = build_dataloaders(
        excel_path  = config["data"]["excel_path"],
        tokenizer   = tokenizer,
        max_seq_len = config["data"]["max_seq_len"],
        batch_size  = config["training"]["batch_size"],
        train_ratio = config["data"].get("train_ratio", 0.8),
        val_ratio   = config["data"].get("val_ratio", 0.1),
        seed        = config["training"]["seed"],
    )

    # Teacher Model (Mamba3, frozen)
    model_cfg = config.get('model', {})
    teacher = BanglaDialectEmbeddingModel(
        vocab_size   = model_cfg.get('vocab_size', 101975),
        d_model      = model_cfg.get('d_model', 128),
        n_layers     = model_cfg.get('n_layers', 2),
        embed_dim    = model_cfg.get('embed_dim', 128),
        d_state      = model_cfg.get('d_state', 16),
        d_conv       = model_cfg.get('d_conv', 4),
        expand       = model_cfg.get('expand', 2),
        headdim      = model_cfg.get('headdim', 64),
        dropout      = model_cfg.get('dropout', 0.1),
        architecture = 'mamba3',
    ).to(device)

    # Load teacher checkpoint
    ckpt_path = "checkpoints/teacher_mamba3.pt"
    if not os.path.exists(ckpt_path):
        print(f"Teacher checkpoint not found: {ckpt_path}")
        print("Please copy your best Mamba3 checkpoint to checkpoints/teacher_mamba3.pt")
        return

    ckpt = torch.load(ckpt_path, map_location=device)
    teacher.load_state_dict(ckpt["model_state_dict"])
    teacher.eval()

    # Freeze teacher
    for param in teacher.parameters():
        param.requires_grad = False

    print(f"Teacher loaded from epoch {ckpt['epoch']} (val loss: {ckpt['val_loss']:.4f})")

    # Student Model
    student = StudentDialectModel(
        vocab_size = model_cfg.get('vocab_size', 101975),
        d_model    = 64,
        n_layers   = 2,
        embed_dim  = 128,
        dropout    = model_cfg.get('dropout', 0.1),
    ).to(device)

    student_params = sum(p.numel() for p in student.parameters())
    print(f"Student parameters: {student_params:,}")

    # Loss + Optimizer
    criterion = TripletLoss(margin=1.0)
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=0.0001,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=30
    )

    # Distillation weight
    alpha = 0.5  # task_loss + 0.5 * distill_loss

    # Checkpoint dir
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    best_val_loss = float("inf")

    print("\nStarting Knowledge Distillation...")

    for epoch in range(1, 31):
        student.train()
        total_train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/30"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            standard_ids   = batch["standard_ids"].to(device)
            std_mask       = (standard_ids != 0).long()

            # Student forward
            student_dialect  = student(input_ids, attention_mask)
            student_standard = student(standard_ids, std_mask)

            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_dialect  = teacher(input_ids, attention_mask)
                teacher_standard = teacher(standard_ids, std_mask)

            # Task loss (Triplet)
            task_loss = criterion(student_dialect, student_standard)

            # Distillation loss (MSE)
            distill_loss = F.mse_loss(student_dialect, teacher_dialect) + \
                           F.mse_loss(student_standard, teacher_standard)

            # Combined loss
            loss = task_loss + alpha * distill_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()

        scheduler.step()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss   = validate_student(
            student, teacher, val_loader, criterion, device, alpha
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
            }, f"{ckpt_dir}/student_model.pt")
            print(f"  ✓ Best student saved → {ckpt_dir}/student_model.pt")

    print(f"\nDistillation complete. Best Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    distill()