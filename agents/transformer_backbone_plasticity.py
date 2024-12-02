import torch
import torch.nn as nn

class TransformerWithPlasticity(nn.Module):
    def __init__(self, state_dim, embed_dim, num_heads, num_layers, max_seq_length):
        """
        Transformer with plasticity injection in the head layer.
        
        Args:
            state_dim (int): Input state dimension.
            embed_dim (int): Embedding dimension for the Transformer.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of layers in the Transformer.
            max_seq_length (int): Maximum sequence length (e.g., flattened grid size).
        """
        super(TransformerWithPlasticity, self).__init__()
        
        # Encoder (shared part, frozen during plasticity injection)
        self.embedding = nn.Linear(state_dim, embed_dim)
        self.positional_encoding = self._generate_positional_encoding(max_seq_length, embed_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, num_heads),
            num_layers
        )
        
        # Head layers for predictions
        self.head = nn.Linear(embed_dim, embed_dim)  # Standard head
        
        # Plasticity injection: New head parameters
        self.head_prime_1 = nn.Linear(embed_dim, embed_dim)  # Residual learning parameters (trainable)
        self.head_prime_2 = nn.Linear(embed_dim, embed_dim)  # Frozen parameters

        self.freeze_head()  # Freeze the second copy of the head

    def _generate_positional_encoding(self, max_seq_length, embed_dim):
        position = torch.arange(0, max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_seq_length, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        """
        Forward pass for the Transformer with plasticity injection.
        
        Args:
            x (Tensor): Input state tensor of shape (batch_size, seq_length, state_dim).
        
        Returns:
            Tensor: The agent's predictions after plasticity injection.
        """
        batch_size, seq_length, _ = x.size()
        x = self.embedding(x)  # Project state to embedding dimension
        x = x + self.positional_encoding[:, :seq_length, :].to(x.device)  # Add positional encodings
        x = self.encoder(x)  # Apply Transformer encoder
        
        # Standard head output
        head_output = self.head(x.mean(dim=1))

        # Plasticity injection (additional learnable residual)
        head_prime_1_output = self.head_prime_1(x.mean(dim=1))  # Residual head (trainable)
        head_prime_2_output = self.head_prime_2(x.mean(dim=1))  # Frozen head
        
        # Final output with plasticity injection
        output = head_output + head_prime_1_output - head_prime_2_output
        return output

    def freeze_head(self):
        """
        Freeze the parameters of the second copy of the head (θ′2).
        """
        for param in self.head_prime_2.parameters():
            param.requires_grad = False
