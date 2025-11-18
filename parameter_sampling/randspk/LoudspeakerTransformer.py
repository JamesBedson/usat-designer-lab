import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class LoudspeakerTransformer(nn.Module):
    def __init__(self, d_model=16, nhead=2, num_layers=1, dim_feedforward=32, output_dim=100):
        super().__init__()
        self.embedding = nn.Linear(3, d_model)  # (x, y, z) → d_model

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, x):  # x: (B, N, 3)
        x = self.embedding(x)      # → (B, N, d_model)
        x = self.encoder(x)        # → (B, N, d_model)
        x = x.mean(dim=1)          # mean over speakers → (B, d_model)
        return self.output_layer(x)  # → (B, output_dim)
    
# Example: batch of 2 samples, each with 5 loudspeakers
batch_size = 2
num_points = 5  # Number of speakers
input_dim = 3   # (x, y, z)

# Input shape: (num_points, batch_size, 3)
x = torch.randn(batch_size, num_points, input_dim)

# Create the model
loudspeaker_encoder = LoudspeakerTransformer()

# Forward pass
out = loudspeaker_encoder(x)  # Output shape: (batch_size, output_dim)
print("Output shape:", out.shape)