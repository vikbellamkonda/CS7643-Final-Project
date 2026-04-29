"""
RNNmodel.py
--------
Vanilla RNN for realized volatility forecasting.

Architecture:
    Input:   (batch, lookback, 1)  -- univariate log-returns
    Stock ID: (batch, num_stocks)  -- one-hot stock embedding
    Output:  (batch,)              -- predicted log realized volatility

The one-hot vector is projected to a learned embedding and concatenated
with the return sequence at every timestep, so the RNN sees both the
price signal and the stock identity simultaneously.
"""

import torch
import torch.nn as nn


class VanillaRNN(nn.Module):
    """
    Single-layer Vanilla RNN volatility forecaster.

    Parameters
    ----------
    num_stocks   : int   – number of unique stocks (one-hot dim)
    embed_dim    : int   – projection size for the stock embedding
    hidden_size  : int   – RNN hidden state dimension
    num_layers   : int   – stacked RNN layers (1 = vanilla)
    dropout      : float – dropout on the final linear layer (0 = off)
    """

    def __init__(
        self,
        num_stocks: int,
        embed_dim: int = 16,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # --- Stock identity projection -----------------------------------
        # Maps the one-hot (num_stocks,) → (embed_dim,)
        # Linear with no bias keeps the embedding space clean.
        self.stock_embedding = nn.Linear(num_stocks, embed_dim, bias=False)

        # --- Vanilla RNN -------------------------------------------------
        # input_size = 1 (return) + embed_dim (stock identity)
        # The embedding is broadcast and concatenated at every timestep
        # inside forward(), so the RNN always knows which stock it is
        # reading — important for a pooled multi-stock model.
        self.rnn = nn.RNN(
            input_size=1 + embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,   # (batch, seq, feature)
            nonlinearity="tanh",
        )

        # --- Output head -------------------------------------------------
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, stock_id: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x        : (batch, lookback, 1)   log-return sequence
        stock_id : (batch, num_stocks)    one-hot stock vector

        Returns
        -------
        pred : (batch,)  predicted log realized volatility
        """
        batch_size, seq_len, _ = x.shape

        # Project one-hot → embedding: (batch, embed_dim)
        emb = self.stock_embedding(stock_id)           # (B, E)

        # Expand embedding across the time dimension so it can be
        # concatenated with each timestep of the return sequence.
        emb_expanded = emb.unsqueeze(1).expand(-1, seq_len, -1)  # (B, T, E)

        # Concatenate returns + embedding at every step: (B, T, 1+E)
        rnn_input = torch.cat([x, emb_expanded], dim=-1)

        # RNN forward — we only need the final hidden state.
        # out: (B, T, hidden_size) | h_n: (num_layers, B, hidden_size)
        out, _ = self.rnn(rnn_input)

        # Take the last timestep's output as the sequence summary.
        last_hidden = out[:, -1, :]        # (B, hidden_size)

        # Project to scalar prediction
        pred = self.fc(self.dropout(last_hidden)).squeeze(-1)  # (B,)
        return pred
