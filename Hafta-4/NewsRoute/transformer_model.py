import torch
import torch.nn as nn
import math

MAX_LENGTH = 64
PAD_ID = 0


def positional_encoding(seq_len, d_model):

    positions = torch.arange(seq_len).unsqueeze(1)

    div_term = torch.exp(
        torch.arange(0, d_model, 2).float()
        * (-math.log(10000.0) / d_model)
    )

    angles = positions * div_term

    PE = torch.zeros(seq_len, d_model)

    PE[:, 0::2] = torch.sin(angles)
    PE[:, 1::2] = torch.cos(angles)

    return PE


class NewsRouteTransformer(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model=64,
        num_heads=4,
        num_layers=1,
        num_classes=4
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=PAD_ID
        )

        self.register_buffer(
            "pe",
            positional_encoding(MAX_LENGTH, d_model)
        )

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Linear(
            in_features=d_model,
            out_features=num_classes
        )

    def forward(self, input_ids):

        padding_mask = input_ids == PAD_ID

        x = self.embedding(input_ids)

        x = x + self.pe[:input_ids.size(1)].unsqueeze(0)

        x = self.encoder(
            src=x,
            src_key_padding_mask=padding_mask
        )

        valid_mask = input_ids != PAD_ID

        mask = valid_mask.unsqueeze(-1).float()

        mask_x = x * mask

        token_sum = mask_x.sum(dim=1)

        token_count = mask.sum(dim=1).clamp(min=1)

        pooled = token_sum / token_count

        logits = self.classifier(pooled)

        return logits