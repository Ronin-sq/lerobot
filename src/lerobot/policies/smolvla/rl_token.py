"""RL-token encoder and autoregressive reconstruction decoder."""

from __future__ import annotations

import torch
import torch.nn as nn


class RLEncoder(nn.Module):
    """Compress a token sequence into the hidden state of a learned ``e_rl`` token."""

    def __init__(self, dim: int, depth: int = 2, heads: int = 8):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads})")
        self.dim = dim
        self.e_rl = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=4 * dim,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=depth,
            norm=nn.LayerNorm(dim),
        )

    def forward(
        self,
        z: torch.Tensor
    ) -> torch.Tensor:
        if z.ndim != 3:
            raise ValueError(f"Expected z with shape [B, L, D], got {tuple(z.shape)}")
        batch_size, _, dim = z.shape
        if dim != self.dim:
            raise ValueError(f"Expected hidden dimension {self.dim}, got {dim}")
       

        e_rl = self.e_rl.expand(batch_size, -1, -1)
        x = torch.cat([z, e_rl], dim=1)

       
        # The paper's extractor uses unrestricted self-attention over the
        # retained VLA tokens; only padding positions are masked.
        encoded = self.encoder(x)
        return encoded[:, -1]


class RLDecoder(nn.Module):
    """Causal decoder for ``[z_rl, z_bar_1, ..., z_bar_{i-1}]``."""

    def __init__(self, dim: int, depth: int = 4, heads: int = 8):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads})")
        self.dim = dim

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=4 * dim,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=depth,
            norm=nn.LayerNorm(dim),
        )
        self.output_proj = nn.Linear(dim, dim)

    def forward(
        self,
        z_rl: torch.Tensor,
        z_bar: torch.Tensor
        # valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if z_rl.ndim != 2 or z_bar.ndim != 3:
            raise ValueError("Expected z_rl=[B, D] and z_bar=[B, M, D]")
        batch_size, seq_len, dim = z_bar.shape
        if z_rl.shape != (batch_size, dim):
            raise ValueError(
                f"Expected z_rl shape {(batch_size, dim)}, got {tuple(z_rl.shape)}"
            )
        if dim != self.dim:
            raise ValueError(f"Expected hidden dimension {self.dim}, got {dim}")
        
        # Position i predicts z_bar_i. z_rl is the first causal prefix token.
        decoder_input = torch.cat([z_rl.unsqueeze(1), z_bar[:, :-1]], dim=1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=z_bar.device
        )

        hidden = self.decoder(
            src=decoder_input,
            mask=causal_mask,
        )
        return self.output_proj(hidden)


def rl_token_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    # valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean per-token squared L2 loss from the RL-token objective."""
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError(
            "prediction and target must have the same shape [B, M, D], "
            f"got {tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    token_loss = (prediction - target).pow(2).sum(dim=-1)
   
    return token_loss.mean()

class RLTokenExtractor(nn.Module):
    """Trainable Encoder/Decoder pair for frozen SmolVLA representations."""

    def __init__(
        self,
        dim: int,
        encoder_depth: int = 2,
        decoder_depth: int = 4,
        heads: int = 8,
    ):
        super().__init__()
        self.encoder = RLEncoder(dim, depth=encoder_depth, heads=heads)
        self.decoder = RLDecoder(dim, depth=decoder_depth, heads=heads)

    def forward(
        self,
        z: torch.Tensor,
        # valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(z_rl, prediction, reconstruction_loss)``."""
        z_bar = z.detach()
        z_rl = self.encoder(z_bar)
        prediction = self.decoder(z_rl, z_bar)
        loss = rl_token_reconstruction_loss(
            prediction, z_bar
        )
        return z_rl, prediction, loss
