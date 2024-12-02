import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBackbone(nn.Module):
    def __init__(self, state_dim, embed_dim, num_heads, num_layers, max_seq_length):
        """
        Transformer-based backbone for processing environment states.
        
        Args:
            state_dim (int): Dimension of the input state (e.g., flattened grid size).
            embed_dim (int): Embedding dimension for the Transformer.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of Transformer encoder layers.
            max_seq_length (int): Maximum sequence length (e.g., grid size or flattened input).
        """
        super(TransformerBackbone, self).__init__()
        self.embedding = nn.Linear(state_dim, embed_dim)
        self.positional_encoding = self._generate_positional_encoding(max_seq_length, embed_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, num_heads),
            num_layers
        )
        self.output_layer = nn.Linear(embed_dim, embed_dim)

    def _generate_positional_encoding(self, max_seq_length, embed_dim):
        """
        Generate sinusoidal positional encodings for the Transformer.
        """
        position = torch.arange(0, max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_seq_length, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        """
        Forward pass for the Transformer backbone.
        
        Args:
            x (Tensor): Input state tensor of shape (batch_size, seq_length, state_dim).
        
        Returns:
            Tensor: Processed state representations of shape (batch_size, embed_dim).
        """
        batch_size, seq_length, _ = x.size()
        x = self.embedding(x)  # Project state to embedding dimension
        x = x + self.positional_encoding[:, :seq_length, :].to(x.device)  # Add positional encodings
        x = self.encoder(x)  # Apply Transformer encoder
        x = self.output_layer(x.mean(dim=1))  # Pool over sequence dimension and project
        return x
