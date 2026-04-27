import torch.nn as nn
import torch.nn.functional as F
import math
import torch

class LightTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, nhead, num_layers, dim_feedforward, output_dim, dropout=0.1, padding_idx=0, threshold=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.positional_encoding = PositionalEncoding(embedding_dim)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.threshold = threshold

    def forward(self, src, training=True):
        embedded = self.dropout(self.positional_encoding(self.embedding(src)))
        encoder_output = self.transformer_encoder(embedded)
        mask = (src != 0).float().unsqueeze(-1)
        masked_output = encoder_output * mask
        last_valid_output = torch.sum(masked_output, dim=1) / torch.sum(mask, dim=1)
        output = torch.sigmoid(self.fc(last_valid_output))
        
        if not training: 
            output = (output > self.threshold).float()
        
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)