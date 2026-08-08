import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


# ──────────────────────────────────────────────
# Config loader
# ──────────────────────────────────────────────

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────
# Contrastive Loss
# ──────────────────────────────────────────────

class ContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss for dialect-aware embeddings.

    Same dialect pair  → embeddings কাছাকাছি হবে  (pulled together)
    Different sentence → embeddings দূরে থাকবে    (pushed apart)

    margin: minimum distance between negative pairs (default 1.0)
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        dialect_emb: torch.Tensor,   # [B, hidden_dim]  — dialect sentence embedding
        standard_emb: torch.Tensor,  # [B, hidden_dim]  — standard Bangla embedding
    ) -> torch.Tensor:
        """
        dialect_emb  : embedding of dialect sentence   (e.g. Sylheti)
        standard_emb : embedding of standard Bangla equivalent

        Same-row pairs are POSITIVE (label=1) — same meaning, different dialect.
        Cross-row pairs are NEGATIVE (label=0) — different sentences.

        Loss = mean over all pairs of:
            positive: distance²
            negative: max(0, margin - distance)²
        """
        B = dialect_emb.size(0)

        # L2-normalize embeddings for stable distance computation
        d = F.normalize(dialect_emb,  dim=-1)   # [B, D]
        s = F.normalize(standard_emb, dim=-1)   # [B, D]

        # Pairwise Euclidean distance matrix  [B, B]
        # dist[i, j] = distance between dialect[i] and standard[j]
        dist_matrix = torch.cdist(d, s, p=2)    # [B, B]

        # Positive pairs: diagonal (same sentence, different dialect)
        pos_dist = torch.diagonal(dist_matrix)  # [B]
        pos_loss  = pos_dist.pow(2)

        # Negative pairs: off-diagonal (different sentences)
        mask = ~torch.eye(B, dtype=torch.bool, device=dialect_emb.device)
        neg_dist = dist_matrix[mask]            # [B*(B-1)]
        neg_loss  = F.relu(self.margin - neg_dist).pow(2)

        loss = pos_loss.mean() + neg_loss.mean()
        return loss


# ──────────────────────────────────────────────
# SSM Placeholder Block
# (Mamba2 architecture — Sunday meeting-এ decide হবে)
# ──────────────────────────────────────────────

class SSMPlaceholderBlock(nn.Module):
    """
    Placeholder for Mamba2/SSM block.
    এখন simple Linear + ReLU দিয়ে রাখা হয়েছে।
    Meeting-এর পরে এই block টা replace হবে real Mamba2 block দিয়ে।
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1     = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2     = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm    = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len, hidden_dim]
        residual = x
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.norm(x + residual)   # residual connection
        return x


# ──────────────────────────────────────────────
# Main Model
# ──────────────────────────────────────────────

class BanglaDialectEmbeddingModel(nn.Module):
    """
    Bangla Dialect Embedding Model — SSM-based (Mamba2 placeholder).

    Input  : tokenized Bangla text (input_ids, attention_mask)
    Output : sentence-level embedding [batch_size, hidden_dim]

    Architecture:
        Embedding layer
            → N × SSMPlaceholderBlock   (will be replaced with Mamba2)
            → Mean pooling (masked)
            → projection head
    """

    def __init__(self, config_path="configs/config.yaml"):
        super().__init__()
        config = load_config(config_path)

        self.hidden_dim  = config["model"]["hidden_dim"]
        self.num_layers  = config["model"]["num_layers"]
        self.dropout_p   = config["model"]["dropout"]
        vocab_size       = config["model"].get("vocab_size", 32000)

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, self.hidden_dim, padding_idx=0)

        # SSM blocks (placeholder — replace with Mamba2 after meeting)
        self.ssm_blocks = nn.ModuleList([
            SSMPlaceholderBlock(self.hidden_dim, self.dropout_p)
            for _ in range(self.num_layers)
        ])

        # Final projection head (embedding space)
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.dropout = nn.Dropout(self.dropout_p)

    # ------------------------------------------------------------------
    def _mean_pool(
        self,
        x: torch.Tensor,              # [B, seq_len, hidden_dim]
        attention_mask: torch.Tensor, # [B, seq_len]
    ) -> torch.Tensor:                # [B, hidden_dim]
        """Masked mean pooling — padding token গুলো ignore করে।"""
        mask  = attention_mask.unsqueeze(-1).float()   # [B, seq_len, 1]
        summed = (x * mask).sum(dim=1)                 # [B, hidden_dim]
        count  = mask.sum(dim=1).clamp(min=1e-9)       # [B, 1]
        return summed / count

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids:      torch.Tensor,        # [B, seq_len]
        attention_mask: torch.Tensor = None, # [B, seq_len]
    ) -> torch.Tensor:                       # [B, hidden_dim]

        x = self.embedding(input_ids)        # [B, seq_len, hidden_dim]
        x = self.dropout(x)

        for block in self.ssm_blocks:
            x = block(x)                     # [B, seq_len, hidden_dim]

        # Sentence-level representation via mean pooling
        if attention_mask is not None:
            x = self._mean_pool(x, attention_mask)
        else:
            x = x.mean(dim=1)               # [B, hidden_dim]

        x = self.projection(x)              # [B, hidden_dim]
        return x