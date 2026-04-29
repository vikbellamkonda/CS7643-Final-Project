"""
LSTMmodel.py
--------
Bidirectional LSTM for realized volatility forecasting.
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(
        self,
        num_stocks: int,
        embed_dim: int = 16,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.stock_embedding = nn.Linear(num_stocks, embed_dim, bias=False)

        self.lstm = nn.LSTM(
            input_size=1 + embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x: torch.Tensor, stock_id: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        emb = self.stock_embedding(stock_id)
        emb_expanded = emb.unsqueeze(1).expand(-1, seq_len, -1)

        lstm_input = torch.cat([x, emb_expanded], dim=-1)

        out, _ = self.lstm(lstm_input)

        last_hidden = out[:, -1, :]

        pred = self.fc(self.dropout(last_hidden)).squeeze(-1)

        return pred