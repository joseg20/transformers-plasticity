import pytest
import torch
from agents.transformer_backbone import TransformerBackbone

# Test Transformer initialization
def test_transformer_initialization():
    state_dim = 5  # Assume 5-dimensional input state
    embed_dim = 128
    num_heads = 4
    num_layers = 2
    max_seq_length = 25  # Maximum sequence length (flattened grid)

    transformer = TransformerBackbone(
        state_dim=state_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_length=max_seq_length
    )

    # Check that transformer model has the correct architecture
    assert isinstance(transformer.embedding, torch.nn.Linear)
    assert isinstance(transformer.encoder, torch.nn.TransformerEncoder)
    assert isinstance(transformer.output_layer, torch.nn.Linear)

# Test Transformer forward pass
def test_transformer_forward():
    state_dim = 5
    embed_dim = 128
    num_heads = 4
    num_layers = 2
    max_seq_length = 25
    batch_size = 4

    transformer = TransformerBackbone(
        state_dim=state_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_length=max_seq_length
    )

    # Simulate a batch of input states (batch_size, max_seq_length, state_dim)
    input_data = torch.randn(batch_size, max_seq_length, state_dim)
    output = transformer(input_data)

    # The output should be of shape (batch_size, embed_dim)
    assert output.shape == (batch_size, embed_dim)
