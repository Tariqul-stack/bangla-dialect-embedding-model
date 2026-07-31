import torch
import torch.nn as nn
import yaml

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class BanglaDialectEmbeddingModel(nn.Module):
    def __init__(self, config_path="configs/config.yaml"):
        super().__init__()
        config = load_config(config_path)
        
        self.hidden_dim = config["model"]["hidden_dim"]
        self.num_layers = config["model"]["num_layers"]
        self.dropout = config["model"]["dropout"]

        # Placeholder — Mamba/SSM architecture Sunday te decide hobe
        self.embedding = nn.Embedding(32000, self.hidden_dim)
        self.layers = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.hidden_dim)
            for _ in range(self.num_layers)
        ])
        self.dropout_layer = nn.Dropout(self.dropout)
        self.output = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = torch.relu(layer(x))
            x = self.dropout_layer(x)
        x = self.output(x)
        # Mean pooling to get sentence embedding
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            x = x.mean(dim=1)
        return x