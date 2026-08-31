import torch
import coremltools as ct
import sys
import os

sys.path.insert(0, '/Users/tariqulislam/bangla-dialect-embedding-model')
from src.model import StudentDialectModel

print(f"coremltools version: {ct.__version__}")

# ── Load Student Model ──
student = StudentDialectModel(
    vocab_size=101975, d_model=64, n_layers=2, embed_dim=128, dropout=0.1,
)
ckpt = torch.load(
    '/Users/tariqulislam/bangla-dialect-models/student_model.pt',
    map_location='cpu'
)
student.load_state_dict(ckpt['model_state_dict'])
student.eval()
print("Student loaded")

SEQ_LEN = 128

# ── Wrapper with fixed seq_len (avoid dynamic shape ops) ──
class TraceableWrapper(torch.nn.Module):
    def __init__(self, model, seq_len):
        super().__init__()
        self.model = model
        # Precompute fixed position ids as a buffer (not from input.shape)
        self.register_buffer(
            "positions",
            torch.arange(seq_len).unsqueeze(0)
        )

    def forward(self, input_ids, attention_mask):
        B = input_ids.shape[0]
        x = self.model.token_emb(input_ids) + self.model.pos_emb(
            self.positions.expand(B, -1)
        )
        x = self.model.emb_norm(self.model.emb_dropout(x))

        for block in self.model.blocks:
            x = block(x)

        x = self.model.final_norm(x)

        mask = attention_mask.float().unsqueeze(-1)
        summed = (x * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / lengths

        return self.model.proj(pooled)

wrapped = TraceableWrapper(student, SEQ_LEN)
wrapped.eval()

dummy_ids = torch.randint(1, 101975, (1, SEQ_LEN))
dummy_mask = torch.ones(1, SEQ_LEN, dtype=torch.long)

print("Tracing model...")
traced_model = torch.jit.trace(wrapped, (dummy_ids, dummy_mask))
print("Model traced")

print("Converting to CoreML...")
mlmodel = ct.convert(
    traced_model,
    inputs=[
        ct.TensorType(name="input_ids", shape=(1, SEQ_LEN), dtype=int),
        ct.TensorType(name="attention_mask", shape=(1, SEQ_LEN), dtype=int),
    ],
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.ALL,
    minimum_deployment_target=ct.target.iOS15,
)

save_path = '/Users/tariqulislam/bangla-dialect-models/student_coreml.mlpackage'
mlmodel.save(save_path)

size = sum(
    os.path.getsize(os.path.join(dirpath, f))
    for dirpath, _, files in os.walk(save_path)
    for f in files
) / 1e6
print(f"Student CoreML saved: {size:.1f} MB")