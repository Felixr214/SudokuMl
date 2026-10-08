import torch
import torch.nn as nn
import torch.nn.functional as F

class SudokuTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1):
        super(SudokuTransformer, self).__init__()

        self.d_model = d_model

        # 1. Digit Value Embedding (Tokens 0-9 -> d_model vector)
        self.val_embed = nn.Embedding(num_embeddings=10, embedding_dim=d_model)

        # 2. 2D Spatial Coordinate Embeddings
        self.row_embed = nn.Embedding(num_embeddings=9, embedding_dim=d_model)
        self.col_embed = nn.Embedding(num_embeddings=9, embedding_dim=d_model)
        self.box_embed = nn.Embedding(num_embeddings=9, embedding_dim=d_model)

        # Pre-compute fixed spatial coordinate indices for 81 cells
        idx = torch.arange(81)
        rows = idx // 9
        cols = idx % 9
        boxes = (rows // 3) * 3 + (cols // 3)

        # Register as non-trainable buffers
        self.register_buffer('rows', rows)
        self.register_buffer('cols', cols)
        self.register_buffer('boxes', boxes)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. Output Projection Head
        # Predicts 9 logits per cell corresponding to target digits 1 through 9
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 9)
        )

    def forward(self, x):
        """
        x: Tensor of shape (Batch, 9, 9) with int values 0-9
        Returns: Logits of shape (Batch, 9, 9, 9)
        """
        batch_size = x.size(0)

        # embedding

        x_flat = x.view(batch_size, 81)
        val_e = self.val_embed(x_flat)
        pos_e = self.row_embed(self.rows) + self.col_embed(self.cols) + self.box_embed(self.boxes)
        h = val_e + pos_e.unsqueeze(0)  # (B, 81, d_model)

        # Pass through Transformer Stack
        h = self.transformer(h)  # (B, 81, d_model)

        # Predict logits per cell -> (B, 81, 9)
        logits = self.head(h)

        # Reshape back to grid format -> (B, 9, 9, 9)
        return logits.view(batch_size, 9, 9, 9)