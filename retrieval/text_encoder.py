import torch
import torch.nn as nn
import math
import os
import pickle

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

class TextEncoder:
    def __init__(self, model_path='trained_model/best_model.pth', vocab_path='trained_model/vocab.pkl'):
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"{vocab_path}")
        with open(vocab_path, 'rb') as f:
            vocab_data = pickle.load(f)
        self.stoi = vocab_data['stoi']
        self.itos = vocab_data['itos']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.vocab_size = len(self.itos)
        self.embedding_dim = 128
        self.nhead = 2
        self.num_layers = 2
        self.dim_feedforward = 256
        self.output_dim = 4
        self.dropout = 0.1
        self.padding_idx = self.stoi['<pad>']
        
        self.model = LightTransformerEncoder(
            self.vocab_size,
            self.embedding_dim,
            self.nhead,
            self.num_layers,
            self.dim_feedforward,
            self.output_dim,
            self.dropout,
            self.padding_idx
        ).to(self.device)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
    
    def encode(self, text, max_length=None):
        tokens = text.lower().split()
        indexed_tokens = [self.stoi.get(token, self.stoi['<unk>']) for token in tokens]
        
        text_tensor = torch.tensor(indexed_tokens).unsqueeze(0).to(self.device)

        if max_length is not None:
            if text_tensor.size(1) < max_length:
                padding = torch.zeros(1, max_length - text_tensor.size(1), dtype=torch.long).to(self.device)
                text_tensor = torch.cat((text_tensor, padding), dim=1)
            else:
                text_tensor = text_tensor[:, :max_length]
        
        with torch.no_grad():
            encoded_vector = self.model(text_tensor, training=False)
        
        return encoded_vector.cpu().numpy()[0]

def save_vocab(vocab_path='trained_model/vocab.pkl'):

    from src.dataset import TextCodeDataset
    
    train_dataset = TextCodeDataset('dataset/train')
    
    vocab_data = {
        'stoi': train_dataset.stoi,
        'itos': train_dataset.itos
    }
    
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab_data, f)
    
    print(f"{vocab_path}")

if __name__ == "__main__":
    save_vocab()