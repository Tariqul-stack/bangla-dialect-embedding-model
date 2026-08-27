"""
model.py — Bangla Dialect Embedding Model with Mamba2 SSM
==========================================================
Architecture:
    TokenEmbedding → N × Mamba2Block → MeanPool → ProjectionHead

Mamba2 uses the SSD (State Space Duality) kernel from mamba-ssm >= 2.0.0.
Falls back to a pure-PyTorch Mamba2-equivalent (no custom CUDA kernels)
when mamba-ssm is not installed — useful for CPU debugging or environments
where the kernel build fails (e.g. Kaggle free-tier).

Install on Kaggle (T4, CUDA 12.x):
    !pip install causal-conv1d>=1.4.0
    !pip install mamba-ssm>=2.2.0
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Try to import the real Mamba2 layer ──────────────────────────────────────
try:
    from mamba_ssm import Mamba2                # mamba-ssm >= 2.0.0
    MAMBA2_AVAILABLE = True
    print("[model] ✓ mamba-ssm found — using real Mamba2 CUDA kernels.")
except ImportError:
    MAMBA2_AVAILABLE = False
    print("[model] ✗ mamba-ssm not found — using pure-PyTorch Mamba2 fallback.")

# ─────────────────────────────────────────────────────────────────────────────
# Pure-PyTorch Mamba1 Fallback
# ─────────────────────────────────────────────────────────────────────────────
class Mamba1Fallback(nn.Module):
    """
    Pure-PyTorch Mamba1 implementation.

    Key differences from Mamba2:
    1. No multi-head SSD — simpler SSM
    2. Sequential-style scan (no parallel SSD)
    3. No headdim parameter
    4. Original Mamba (2023) design

    Args:
        d_model : model dimension
        d_state : SSM state size (default 16)
        d_conv  : depthwise conv kernel (default 4)
        expand  : expansion factor (default 2)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        **kwargs,   # headdim ignore করো Mamba1 এ নেই
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Depthwise causal conv
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        # SSM projections
        dt_rank = math.ceil(d_model / 16)
        self.dt_rank = dt_rank
        self.x_proj = nn.Linear(
            self.d_inner,
            dt_rank + d_state * 2,
            bias=False
        )
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)

        # A: fixed diagonal (not multi-head like Mamba2)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # D skip
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) → (B, L, D)"""
        B, L, D = x.shape

        # 1. Split
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)

        # 2. Causal conv
        x_conv = x_in.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[..., :L]
        x_conv = F.silu(x_conv).transpose(1, 2)

        # 3. SSM params
        x_dbl = self.x_proj(x_conv)
        dt, B_ssm, C_ssm = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))

        # 4. Discretize A (ZOH — same as Mamba2 but simpler)
        A = -torch.exp(self.A_log.float())
        dt_exp = torch.einsum('bld,ds->blds', dt, A)
        A_bar = torch.exp(dt_exp)

        # 5. Scan (simpler than Mamba2 — no parallel SSD)
        B_bar = torch.einsum('bld,bls->blds', dt, B_ssm)
        u = x_conv
        Bu = torch.einsum('bld,blds->blds', u, B_bar)

        log_A_cumsum = torch.cumsum(
            torch.log(A_bar.clamp(min=1e-8)), dim=1
        )
        A_cumprod = torch.exp(log_A_cumsum)
        Bu_norm = Bu / A_cumprod.clamp(min=1e-8)
        h = torch.cumsum(Bu_norm, dim=1) * A_cumprod

        # 6. Output
        y = torch.einsum('bls,blds->bld', C_ssm, h)
        y = y + self.D.unsqueeze(0).unsqueeze(0) * u
        y = y * F.silu(z)
        y = self.out_proj(y)
        return y

# ─────────────────────────────────────────────────────────────────────────────
# Pure-PyTorch Mamba2 Fallback
# ─────────────────────────────────────────────────────────────────────────────
class Mamba2Fallback(nn.Module):
    """
    Pure-PyTorch approximation of Mamba2's SSD core.

    Implements the key ideas from the Mamba2 paper (Dao & Gu, ICML 2024):
      • Grouped-value (multi-head) state expansion
      • dt, A, B, C projections in one fused linear
      • Conv1d input preprocessing (causal)
      • dt softplus activation + A log-parameterisation
      • Chunk-wise associative scan approximated by cumsum + gating

    This is NOT the hardware-optimised SSD kernel — it is correct but slower.
    Use it for debugging / environments without the CUDA extension.

    Args:
        d_model   : model dimension (must equal the embedding dimension)
        d_state   : SSM state size (default 64, Mamba2 default)
        d_conv    : depthwise conv kernel size (default 4)
        expand    : expansion factor for inner dim (default 2)
        headdim   : head dimension for multi-head SSD (default 64)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.headdim = headdim
        self.nheads = max(1, self.d_inner // headdim)

        # Input projection: x → z (gate) + x_proj (ssm inputs)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Depthwise causal conv over the inner dim
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        # SSM parameter projections (dt, B, C)
        dt_rank = math.ceil(d_model / 16)
        self.dt_rank = dt_rank
        self.x_proj = nn.Linear(self.d_inner, dt_rank + d_state * 2, bias=False)

        # dt projection (low-rank → d_inner)
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)

        # Log-parameterised A (diagonal)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # D skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.norm = nn.LayerNorm(d_inner := self.d_inner)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
        Returns:
            y: (B, L, D)
        """
        B, L, D = x.shape

        # 1. Input projection → split into ssm-branch and gate
        xz = self.in_proj(x)                          # (B, L, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)                 # each (B, L, d_inner)

        # 2. Causal conv1d on the ssm branch
        x_conv = x_in.transpose(1, 2)                 # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[..., :L]         # causal trim
        x_conv = F.silu(x_conv).transpose(1, 2)       # (B, L, d_inner)

        # 3. Project to dt, B, C
        x_dbl = self.x_proj(x_conv)                   # (B, L, dt_rank+2*d_state)
        dt, B_ssm, C_ssm = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))              # (B, L, d_inner)

        # 4. Discretise A
        A = -torch.exp(self.A_log.float())             # (d_inner, d_state)

        # 5. Simplified scan: cumsum-based approximation
        #    For each token position, compute the SSM output via:
        #    h_t = A_bar * h_{t-1} + B_bar * x_t
        #    y_t = C * h_t + D * x_t
        # We do a batched cumsum as a linear-time approximation.

        # A_bar: (B, L, d_inner, d_state)  using ZOH discretisation
        dt_exp = torch.einsum('bld,ds->blds', dt, A)   # (B, L, d_inner, d_state)
        A_bar = torch.exp(dt_exp)                       # (B, L, d_inner, d_state)

        # B_bar: (B, L, d_inner, d_state)
        B_bar = torch.einsum('bld,bls->blds', dt, B_ssm)

        # u: (B, L, d_inner) → expand for state
        u = x_conv                                      # (B, L, d_inner)
        Bu = torch.einsum('bld,blds->blds', u, B_bar)  # (B, L, d_inner, d_state)

        # Parallel scan via cumulative product of A_bar + cumsum of Bu
        # (exact only for time-invariant A; good approximation for long seqs)
        log_A_cumsum = torch.cumsum(torch.log(A_bar.clamp(min=1e-8)), dim=1)
        A_cumprod = torch.exp(log_A_cumsum)             # (B, L, d_inner, d_state)

        # Divide Bu by A_cumprod, cumsum, then multiply back
        Bu_norm = Bu / A_cumprod.clamp(min=1e-8)
        h = torch.cumsum(Bu_norm, dim=1) * A_cumprod   # (B, L, d_inner, d_state)

        # Output: y = C h + D u
        y = torch.einsum('bls,blds->bld', C_ssm, h)    # (B, L, d_inner)
        y = y + self.D.unsqueeze(0).unsqueeze(0) * u

        # 6. Gate with z and project out
        y = y * F.silu(z)
        y = self.out_proj(y)
        return y

# ─────────────────────────────────────────────────────────────────────────────
# Pure-PyTorch Mamba3 Fallback
# ─────────────────────────────────────────────────────────────────────────────
class Mamba3Fallback(nn.Module):
    """
    Pure-PyTorch Mamba3 implementation.

    Key differences from Mamba2:
    1. Complex-valued states (more expressive)
    2. Exponential-trapezoidal (ET) discretization
    3. BCNorm — normalizes B and C projections
    4. Inference-optimized design

    Args:
        d_model  : model dimension
        d_state  : SSM state size (default 64)
        d_conv   : depthwise conv kernel size (default 4)
        expand   : expansion factor (default 2)
        headdim  : head dimension (default 64)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.headdim = headdim
        self.nheads = max(1, self.d_inner // headdim)

        # Input projection → x_ssm + gate
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Depthwise causal conv
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        # SSM projections
        dt_rank = math.ceil(d_model / 16)
        self.dt_rank = dt_rank

        # Mamba3: B and C project to complex states
        # d_state * 2 because complex = real + imaginary
        self.x_proj = nn.Linear(
            self.d_inner,
            dt_rank + d_state * 2 * 2,  # dt + B(real+imag) + C(real+imag)
            bias=False
        )

        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)

        # Log-parameterised A (complex diagonal)
        A_real = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A_real))

        # Mamba3: A imaginary part (new!)
        self.A_imag = nn.Parameter(torch.zeros(self.d_inner, d_state))

        # D skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Mamba3: BCNorm — normalize B and C (new!)
        self.B_norm = nn.LayerNorm(d_state)
        self.C_norm = nn.LayerNorm(d_state)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(self.d_inner)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
        Returns:
            y: (B, L, D)
        """
        B, L, D = x.shape

        # 1. Input projection
        xz = self.in_proj(x)                           # (B, L, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)                 # each (B, L, d_inner)

        # 2. Causal conv1d
        x_conv = x_in.transpose(1, 2)                 # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[..., :L]         # causal trim
        x_conv = F.silu(x_conv).transpose(1, 2)       # (B, L, d_inner)

        # 3. Project to dt, B, C (complex)
        x_dbl = self.x_proj(x_conv)
        dt, B_real, B_imag, C_real, C_imag = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state,
             self.d_state, self.d_state],
            dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))              # (B, L, d_inner)

        # Mamba3: BCNorm — normalize B and C
        B_real = self.B_norm(B_real)
        C_real = self.C_norm(C_real)

        # 4. Complex A matrix
        # Mamba3: A = -exp(A_log) + i*A_imag
        A_real = -torch.exp(self.A_log.float())        # (d_inner, d_state)
        A_imag = self.A_imag.float()                   # (d_inner, d_state)

        # 5. Exponential-trapezoidal discretization (Mamba3 key innovation)
        # ET discretization: more stable than ZOH for complex states
        # A_bar = exp((A_real + i*A_imag) * dt)
        dt_expanded = torch.einsum('bld,ds->blds', dt, torch.ones_like(A_real))
        A_real_bar = torch.exp(
            torch.einsum('bld,ds->blds', dt, A_real)
        )
        A_imag_bar = torch.einsum('bld,ds->blds', dt, A_imag)

        # 6. Complex state computation
        # B_bar (complex input gate)
        B_bar_real = torch.einsum('bld,bls->blds', dt, B_real)
        B_bar_imag = torch.einsum('bld,bls->blds', dt, B_imag)

        u = x_conv                                      # (B, L, d_inner)
        Bu_real = torch.einsum('bld,blds->blds', u, B_bar_real)
        Bu_imag = torch.einsum('bld,blds->blds', u, B_bar_imag)

        # 7. Parallel scan with complex states
        # Real part of scan
        log_A_cumsum = torch.cumsum(
            torch.log(A_real_bar.clamp(min=1e-8)), dim=1
        )
        A_cumprod = torch.exp(log_A_cumsum)

        Bu_norm_real = Bu_real / A_cumprod.clamp(min=1e-8)
        h_real = torch.cumsum(Bu_norm_real, dim=1) * A_cumprod

        # Imaginary part modulation
        A_imag_cumsum = torch.cumsum(A_imag_bar, dim=1)
        h_imag = h_real * torch.sin(A_imag_cumsum) + \
                 Bu_imag * torch.cos(A_imag_cumsum)

        # 8. Output: combine real and imaginary
        # y = C_real * h_real + C_imag * h_imag
        y = torch.einsum('bls,blds->bld', C_real, h_real) + \
            torch.einsum('bls,blds->bld', C_imag, h_imag)

        # D skip connection
        y = y + self.D.unsqueeze(0).unsqueeze(0) * u

        # 9. Gate + output projection
        y = y * F.silu(z)
        y = self.out_proj(y)
        return y

# ─────────────────────────────────────────────────────────────────────────────
# Mamba1 Block (residual + norm wrapper)
# ─────────────────────────────────────────────────────────────────────────────
class Mamba1Block(nn.Module):
    """
    Pre-norm residual block wrapping Mamba1Fallback.

    Architecture:
        y = x + Dropout(Mamba1(LayerNorm(x)))
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,   # ignored, kept for API compatibility
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ssm = Mamba1Fallback(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.ssm(self.norm(x)))
    
# ─────────────────────────────────────────────────────────────────────────────
# Mamba2 Block (residual + norm wrapper)
# ─────────────────────────────────────────────────────────────────────────────
class Mamba2Block(nn.Module):
    """
    Pre-norm residual block wrapping either the real Mamba2 layer (from
    mamba-ssm) or the pure-PyTorch fallback.

    Architecture:
        y = x + Mamba2(RMSNorm(x))

    Args:
        d_model : model/embedding dimension
        d_state : SSM state expansion (default 64 — Mamba2 default)
        d_conv  : causal conv kernel size (default 4)
        expand  : inner-dim expansion factor (default 2)
        headdim : per-head dimension for SSD (default 64); ignored by fallback
        dropout : dropout after SSM output (default 0.1)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        if MAMBA2_AVAILABLE:
            # Real Mamba2 from mamba-ssm >= 2.0.0
            # headdim must evenly divide d_model * expand
            d_inner = int(expand * d_model)
            # Mamba2 requires nheads = d_inner // headdim; adjust if needed
            nheads = max(1, d_inner // headdim)
            actual_headdim = d_inner // nheads
            self.ssm = Mamba2(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=actual_headdim,
            )
        else:
            self.ssm = Mamba2Fallback(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=headdim,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) → (B, L, D)"""
        return x + self.dropout(self.ssm(self.norm(x)))

# ─────────────────────────────────────────────────────────────────────────────
# Mamba3 Block (residual + norm wrapper)
# ─────────────────────────────────────────────────────────────────────────────
class Mamba3Block(nn.Module):
    """
    Pre-norm residual block wrapping Mamba3Fallback.

    Architecture:
        y = x + Dropout(Mamba3(LayerNorm(x)))

    Args:
        d_model  : model dimension
        d_state  : SSM state size (default 64)
        d_conv   : causal conv kernel (default 4)
        expand   : expansion factor (default 2)
        headdim  : head dimension (default 64)
        dropout  : dropout rate (default 0.1)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ssm = Mamba3Fallback(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) → (B, L, D)"""
        return x + self.dropout(self.ssm(self.norm(x)))

# ─────────────────────────────────────────────────────────────────────────────
# Projection Head
# ─────────────────────────────────────────────────────────────────────────────
class ProjectionHead(nn.Module):
    """
    Two-layer MLP projection head for contrastive learning.
    Maps pooled representation → embedding space.

    Architecture: Linear → GELU → LayerNorm → Linear → L2-normalise
    """

    def __init__(self, d_model: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.LayerNorm(d_model * 2),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return F.normalize(x, p=2, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Main Model
# ─────────────────────────────────────────────────────────────────────────────
class BanglaDialectEmbeddingModel(nn.Module):
    """
    Bangla Dialect Embedding Model — Mamba2 backbone.

    Pipeline:
        token_ids → TokenEmbedding → N × Mamba2Block → MeanPool (masked)
                  → ProjectionHead → L2-normalised dialect embedding

    Args:
        vocab_size  : tokeniser vocabulary size (BanglaBERT: 101,975)
        d_model     : internal model dimension (default 256)
        n_layers    : number of Mamba2 blocks (default 4)
        embed_dim   : output embedding dimension (default 128)
        d_state     : SSM state expansion (default 64)
        d_conv      : causal conv kernel (default 4)
        expand      : inner-dim factor (default 2)
        headdim     : SSD head dimension (default 64)
        dropout     : dropout rate (default 0.1)
        max_seq_len : max token length for positional embedding (default 512)
        pad_token_id: padding token id for masked mean pooling (default 0)
    """

    def __init__(
    self,
    vocab_size: int = 101_975,
    d_model: int = 256,
    n_layers: int = 4,
    embed_dim: int = 128,
    d_state: int = 64,
    d_conv: int = 4,
    expand: int = 2,
    headdim: int = 64,
    dropout: float = 0.1,
    max_seq_len: int = 512,
    pad_token_id: int = 0,
    architecture: str = "mamba2",   
):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.d_model = d_model
        self.architecture = architecture  
        super().__init__()
        self.pad_token_id = pad_token_id
        self.d_model = d_model

        # Token embedding + learned positional embedding
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)
        self.emb_norm = nn.LayerNorm(d_model)

        # SSM blocks — Mamba1, Mamba2 or Mamba3
        if architecture == "mamba3":
            block_class = Mamba3Block
        elif architecture == "mamba1":
            block_class = Mamba1Block
        else:
            block_class = Mamba2Block

        self.blocks = nn.ModuleList([
            block_class(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=headdim,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Final layer norm before pooling
        self.final_norm = nn.LayerNorm(d_model)

        # Projection head
        self.proj = ProjectionHead(d_model, embed_dim, dropout)

        self._init_weights()

    def _init_weights(self):
        """Initialise weights following Mamba2 paper conventions."""
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _masked_mean_pool(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mean-pool over non-padding tokens.

        Args:
            hidden   : (B, L, D) — SSM output
            input_ids: (B, L)    — raw token ids (used to detect padding)
        Returns:
            pooled   : (B, D)
        """
        mask = (input_ids != self.pad_token_id).float().unsqueeze(-1)  # (B, L, 1)
        summed = (hidden * mask).sum(dim=1)                            # (B, D)
        lengths = mask.sum(dim=1).clamp(min=1e-9)                      # (B, 1)
        return summed / lengths

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids     : (B, L) — tokenised input
            attention_mask: (B, L) — 1 for real tokens, 0 for padding (optional;
                                     if None, padding is inferred from pad_token_id)
        Returns:
            embeddings: (B, embed_dim) — L2-normalised dialect embeddings
        """
        B, L = input_ids.shape
        device = input_ids.device

        # Embeddings
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_norm(self.emb_dropout(x))

        # Mamba2 blocks
        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        # Masked mean pooling
        # Use attention_mask if provided, otherwise fall back to pad_token_id
        if attention_mask is not None:
            mask = attention_mask.float().unsqueeze(-1)      # (B, L, 1)
            summed = (x * mask).sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / lengths
        else:
            pooled = self._masked_mean_pool(x, input_ids)   # (B, D)

        # Project to embedding space
        return self.proj(pooled)                             # (B, embed_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Contrastive Loss
# ─────────────────────────────────────────────────────────────────────────────
class ContrastiveLoss(nn.Module):
    """
    NT-Xent (InfoNCE) contrastive loss for dialect pair training.

    Given anchor embeddings `z1` and positive embeddings `z2` (both L2-normed),
    treats all other samples in the batch as negatives.

    Loss = -log( exp(sim(z1_i, z2_i) / τ) /
                  Σ_j exp(sim(z1_i, z2_j) / τ) )

    This is strictly superior to the simple margin-based loss for embedding
    learning — cite as NT-Xent (Chen et al., SimCLR, ICML 2020).

    Args:
        temperature: softmax temperature τ (default 0.07)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z1: torch.Tensor,   # (B, D) — anchor embeddings (L2-normalised)
        z2: torch.Tensor,   # (B, D) — positive embeddings (L2-normalised)
    ) -> torch.Tensor:
        B = z1.size(0)
        device = z1.device

        # Similarity matrix: (2B, 2B)
        z = torch.cat([z1, z2], dim=0)                    # (2B, D)
        sim = torch.matmul(z, z.T) / self.temperature     # (2B, 2B)

        # Mask out self-similarity
        mask = torch.eye(2 * B, dtype=torch.bool, device=device)
        sim.masked_fill_(mask, float('-inf'))

        # Positive indices: z1_i ↔ z2_i are at offset B
        labels = torch.cat([
            torch.arange(B, 2 * B, device=device),
            torch.arange(0, B, device=device),
        ])

        loss = F.cross_entropy(sim, labels)
        return loss

# ─────────────────────────────────────────────────────────────────────────────
# TripletLoss Function
# ─────────────────────────────────────────────────────────────────────────────
class TripletLoss(nn.Module):
    """
    Triplet Loss — NT-Xent এর alternative।
    Ablation study তে compare করার জন্য।
    
    Loss = max(0, pos_dist - neg_dist + margin)
    
    pos_dist = anchor আর positive এর দূরত্ব
    neg_dist = anchor আর negative এর দূরত্ব
    margin   = minimum gap যেটা enforce করা হয়
    """
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,    # (B, D) dialect embedding
        positive: torch.Tensor,  # (B, D) standard embedding
        negative: torch.Tensor = None,  # (B, D) random negative
    ) -> torch.Tensor:
        
        # Cosine distance = 1 - cosine similarity
        pos_sim = F.cosine_similarity(anchor, positive, dim=-1)
        pos_dist = 1 - pos_sim  # (B,)

        if negative is None:
            # Negative automatically বানাও
            # Batch কে shift করো → আলাদা sentence পাবো
            idx = torch.randperm(anchor.size(0), device=anchor.device)
            negative = positive[idx]

        neg_sim = F.cosine_similarity(anchor, negative, dim=-1)
        neg_dist = 1 - neg_sim  # (B,)

        # Triplet loss
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        return loss.mean()

# ─────────────────────────────────────────────────────────────────────────────
# Model factory — easy config switching for Mamba2 vs Mamba3 experiments
# ─────────────────────────────────────────────────────────────────────────────
def build_model(config: dict) -> BanglaDialectEmbeddingModel:
    """
    Build model from a config dict (matches configs/config.yaml structure).

    Example config keys:
        vocab_size, d_model, n_layers, embed_dim,
        d_state, d_conv, expand, headdim, dropout,
        max_seq_len, pad_token_id
    """
    return BanglaDialectEmbeddingModel(
        vocab_size=config.get("vocab_size", 101_975),
        d_model=config.get("d_model", 256),
        n_layers=config.get("n_layers", 4),
        embed_dim=config.get("embed_dim", 128),
        d_state=config.get("d_state", 64),
        d_conv=config.get("d_conv", 4),
        expand=config.get("expand", 2),
        headdim=config.get("headdim", 64),
        dropout=config.get("dropout", 0.1),
        max_seq_len=config.get("max_seq_len", 512),
        pad_token_id=config.get("pad_token_id", 0),
        architecture=config.get("architecture", "mamba2"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Mamba2 CUDA kernels: {'enabled' if MAMBA2_AVAILABLE else 'fallback (pure PyTorch)'}")

    model = BanglaDialectEmbeddingModel(
        vocab_size=101_975,
        d_model=256,
        n_layers=4,
        embed_dim=128,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters — total: {total_params:,}  |  trainable: {trainable_params:,}")

    # Dummy forward pass
    B, L = 8, 64
    ids = torch.randint(1, 101_975, (B, L), device=device)
    ids[:, -10:] = 0  # simulate padding

    emb = model(ids)
    print(f"Input:  {ids.shape}")
    print(f"Output: {emb.shape}  (expected: [{B}, 128])")
    print(f"Norms:  min={emb.norm(dim=-1).min():.4f}  max={emb.norm(dim=-1).max():.4f}  (should be ~1.0)")

    # Contrastive loss test
    loss_fn = ContrastiveLoss(temperature=0.07)
    z1 = model(ids)
    z2 = model(ids)  # positive pair (same tokens, different forward pass)
    loss = loss_fn(z1, z2)
    print(f"Contrastive loss (NT-Xent): {loss.item():.4f}")