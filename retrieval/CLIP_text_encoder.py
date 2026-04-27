import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPModel

class CLIPTextEncoder(nn.Module):
    def __init__(self, model_name='openai/clip-vit-base-patch32', out_channels=4096):
        super(CLIPTextEncoder, self).__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.clip_model = CLIPModel.from_pretrained(model_name)
        
        self.text_encoder = self.clip_model.text_model
        
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
        self.embed_dim = self.clip_model.config.text_config.hidden_size
        self.out_channels = out_channels
        
        if self.embed_dim != self.out_channels:
            self.proj = nn.Linear(self.embed_dim, self.out_channels)
        else:
            self.proj = nn.Identity()

    def forward(self, text_list, device):
        inputs = self.tokenizer(text_list, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
            
        pooled_output = outputs.pooler_output
        projected_output = self.proj(pooled_output)
        
        return projected_output

    def get_text_embeddings(self, prompts, batch_size, device):
        text_features = self.forward(prompts, device)
        text_features = text_features.unsqueeze(0).expand(batch_size, -1, -1)
        return text_features