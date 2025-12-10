"""
Test Suite for MiniGPT-Shakespeare
===================================

Verifies implementation correctness by:
1. Testing individual components in isolation
2. Comparing against PyTorch's built-in layers where applicable
3. Testing gradient flow through the entire model
4. Verifying shapes at each stage

Run with: python -m pytest test_minigpt.py -v
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from minigpt import (
    GPTConfig,
    LayerNorm,
    CausalSelfAttention,
    MLP,
    TransformerBlock,
    GPT,
    get_lr
)


# =============================================================================
# TEST CONFIGURATION
# =============================================================================

@pytest.fixture
def config():
    """Small config for fast testing."""
    return GPTConfig(
        vocab_size=50,
        block_size=32,
        n_layer=2,
        n_head=4,
        n_embd=64,
        dropout=0.0,  # Disable dropout for deterministic tests
        batch_size=4
    )


@pytest.fixture
def device():
    """Use GPU if available."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# LAYER NORM TESTS
# =============================================================================

class TestLayerNorm:
    """Tests for our LayerNorm implementation."""
    
    def test_output_shape(self, config):
        """LayerNorm should preserve input shape."""
        ln = LayerNorm(config.n_embd)
        x = torch.randn(2, 10, config.n_embd)
        out = ln(x)
        assert out.shape == x.shape
    
    def test_normalized_statistics(self, config):
        """Output should have mean≈0 and var≈1 along feature dimension."""
        ln = LayerNorm(config.n_embd)
        x = torch.randn(2, 10, config.n_embd) * 10 + 5  # Non-normalized input
        out = ln(x)
        
        # Check statistics along last dimension
        mean = out.mean(dim=-1)
        var = out.var(dim=-1, unbiased=False)
        
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
        assert torch.allclose(var, torch.ones_like(var), atol=1e-5)
    
    def test_matches_pytorch(self, config):
        """Our LayerNorm should match PyTorch's implementation."""
        torch.manual_seed(42)
        
        our_ln = LayerNorm(config.n_embd)
        pytorch_ln = nn.LayerNorm(config.n_embd)
        
        # Copy weights
        pytorch_ln.weight.data = our_ln.weight.data.clone()
        pytorch_ln.bias.data = our_ln.bias.data.clone()
        
        x = torch.randn(2, 10, config.n_embd)
        
        our_out = our_ln(x)
        pytorch_out = pytorch_ln(x)
        
        assert torch.allclose(our_out, pytorch_out, atol=1e-5)
    
    def test_gradient_flow(self, config):
        """Gradients should flow through LayerNorm."""
        ln = LayerNorm(config.n_embd)
        x = torch.randn(2, 10, config.n_embd, requires_grad=True)
        
        out = ln(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert ln.weight.grad is not None
        assert ln.bias.grad is not None


# =============================================================================
# CAUSAL SELF-ATTENTION TESTS
# =============================================================================

class TestCausalSelfAttention:
    """Tests for the multi-head causal self-attention."""
    
    def test_output_shape(self, config):
        """Attention should preserve input shape."""
        attn = CausalSelfAttention(config)
        x = torch.randn(2, config.block_size, config.n_embd)
        out = attn(x)
        assert out.shape == x.shape
    
    def test_causality(self, config):
        """Each position should only attend to previous positions."""
        attn = CausalSelfAttention(config)
        attn.eval()  # Disable dropout
        
        # Create input where each position has unique values
        x = torch.randn(1, 8, config.n_embd)
        out1 = attn(x)
        
        # Modify a future token
        x_modified = x.clone()
        x_modified[0, 7, :] = torch.randn(config.n_embd) * 100
        out2 = attn(x_modified)
        
        # Outputs for positions 0-6 should be identical
        # because they can't see position 7
        assert torch.allclose(out1[0, :7, :], out2[0, :7, :], atol=1e-5)
        
        # Position 7 should be different
        assert not torch.allclose(out1[0, 7, :], out2[0, 7, :])
    
    def test_variable_sequence_length(self, config):
        """Should handle sequences shorter than block_size."""
        attn = CausalSelfAttention(config)
        
        # Shorter sequence
        x = torch.randn(2, 16, config.n_embd)  # 16 < block_size
        out = attn(x)
        assert out.shape == (2, 16, config.n_embd)
    
    def test_gradient_flow(self, config):
        """Gradients should flow through attention."""
        attn = CausalSelfAttention(config)
        x = torch.randn(2, 10, config.n_embd, requires_grad=True)
        
        out = attn(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert attn.c_attn.weight.grad is not None
        assert attn.c_proj.weight.grad is not None


# =============================================================================
# MLP TESTS
# =============================================================================

class TestMLP:
    """Tests for the feed-forward network."""
    
    def test_output_shape(self, config):
        """MLP should preserve input shape."""
        mlp = MLP(config)
        x = torch.randn(2, 10, config.n_embd)
        out = mlp(x)
        assert out.shape == x.shape
    
    def test_hidden_dimension(self, config):
        """Hidden layer should be 4x embedding dimension."""
        mlp = MLP(config)
        assert mlp.c_fc.out_features == 4 * config.n_embd
        assert mlp.c_proj.in_features == 4 * config.n_embd
    
    def test_nonlinearity(self, config):
        """MLP should be nonlinear (not just a linear transformation)."""
        mlp = MLP(config)
        mlp.eval()
        
        x1 = torch.randn(1, 1, config.n_embd)
        x2 = torch.randn(1, 1, config.n_embd)
        
        out1 = mlp(x1)
        out2 = mlp(x2)
        out_sum = mlp(x1 + x2)
        
        # For a linear function: f(x1 + x2) = f(x1) + f(x2)
        # This should NOT hold for our MLP due to GELU
        assert not torch.allclose(out_sum, out1 + out2, atol=1e-3)


# =============================================================================
# TRANSFORMER BLOCK TESTS
# =============================================================================

class TestTransformerBlock:
    """Tests for a single transformer block."""
    
    def test_output_shape(self, config):
        """Block should preserve input shape."""
        block = TransformerBlock(config)
        x = torch.randn(2, 10, config.n_embd)
        out = block(x)
        assert out.shape == x.shape
    
    def test_residual_connection(self, config):
        """Output should be close to input for initialized model."""
        # With small random weights and residual connections,
        # output shouldn't be too far from input
        torch.manual_seed(42)
        block = TransformerBlock(config)
        block.eval()
        
        x = torch.randn(2, 10, config.n_embd)
        out = block(x)
        
        # The difference should be relatively small due to residuals
        diff = (out - x).abs().mean()
        x_mean = x.abs().mean()
        
        # Difference should be same order of magnitude, not huge
        assert diff < x_mean * 10
    
    def test_gradient_flow(self, config):
        """Gradients should flow through the block."""
        block = TransformerBlock(config)
        x = torch.randn(2, 10, config.n_embd, requires_grad=True)
        
        out = block(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        # Check all submodule gradients
        for name, param in block.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


# =============================================================================
# FULL GPT MODEL TESTS
# =============================================================================

class TestGPT:
    """Tests for the complete GPT model."""
    
    def test_output_shape(self, config):
        """Model should output correct logit shape."""
        model = GPT(config)
        idx = torch.randint(0, config.vocab_size, (2, 10))
        
        logits, _ = model(idx)
        assert logits.shape == (2, 10, config.vocab_size)
    
    def test_loss_computation(self, config):
        """Model should compute loss when targets provided."""
        model = GPT(config)
        idx = torch.randint(0, config.vocab_size, (2, 10))
        targets = torch.randint(0, config.vocab_size, (2, 10))
        
        logits, loss = model(idx, targets)
        
        assert loss is not None
        assert loss.ndim == 0  # Scalar
        assert loss.item() > 0  # Loss should be positive
    
    def test_loss_decreases_with_training(self, config, device):
        """Loss should decrease after a few gradient updates."""
        model = GPT(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Fixed batch for consistent comparison
        idx = torch.randint(0, config.vocab_size, (4, 16)).to(device)
        targets = torch.randint(0, config.vocab_size, (4, 16)).to(device)
        
        # Initial loss
        _, initial_loss = model(idx, targets)
        initial_loss_val = initial_loss.item()
        
        # Train for a few steps
        for _ in range(50):
            optimizer.zero_grad()
            _, loss = model(idx, targets)
            loss.backward()
            optimizer.step()
        
        # Final loss should be lower
        _, final_loss = model(idx, targets)
        assert final_loss.item() < initial_loss_val * 0.5  # At least 50% reduction
    
    def test_weight_tying(self, config):
        """Token embedding and output head should share weights."""
        model = GPT(config)
        
        # These should be the exact same tensor
        assert model.transformer.wte.weight is model.lm_head.weight
    
    def test_generation(self, config, device):
        """Model should generate sequences."""
        model = GPT(config).to(device)
        
        # Start with a single token
        idx = torch.randint(0, config.vocab_size, (1, 1)).to(device)
        
        # Generate 10 new tokens
        generated = model.generate(idx, max_new_tokens=10)
        
        assert generated.shape == (1, 11)  # 1 original + 10 new
        assert (generated >= 0).all()
        assert (generated < config.vocab_size).all()
    
    def test_generation_respects_temperature(self, config, device):
        """Higher temperature should produce more varied outputs."""
        model = GPT(config).to(device)
        model.eval()
        
        torch.manual_seed(42)
        idx = torch.zeros(1, 1, dtype=torch.long, device=device)
        
        # Generate multiple times with low temperature
        low_temp_outputs = []
        for _ in range(5):
            torch.manual_seed(42)
            out = model.generate(idx.clone(), max_new_tokens=20, temperature=0.1)
            low_temp_outputs.append(out[0].tolist())
        
        # Generate multiple times with high temperature
        high_temp_outputs = []
        for seed in range(5):
            torch.manual_seed(seed)
            out = model.generate(idx.clone(), max_new_tokens=20, temperature=2.0)
            high_temp_outputs.append(out[0].tolist())
        
        # Low temperature outputs should be more similar (less variance)
        low_temp_unique = len(set(tuple(x) for x in low_temp_outputs))
        high_temp_unique = len(set(tuple(x) for x in high_temp_outputs))
        
        # With same seed, low temp should be nearly identical
        # High temp with different seeds should vary more
        assert high_temp_unique >= low_temp_unique
    
    def test_parameter_count(self, config):
        """Parameter count should match expected value."""
        model = GPT(config)
        
        # Calculate expected parameters
        n_embd = config.n_embd
        vocab_size = config.vocab_size
        block_size = config.block_size
        n_layer = config.n_layer
        
        # Token embedding (shared with lm_head, so count once)
        tok_emb = vocab_size * n_embd
        
        # Position embedding
        pos_emb = block_size * n_embd
        
        # Per transformer block:
        # - LayerNorm x 2: 2 * (n_embd + n_embd) weights and biases
        # - Attention: c_attn (n_embd -> 3*n_embd) + c_proj (n_embd -> n_embd)
        # - MLP: c_fc (n_embd -> 4*n_embd) + c_proj (4*n_embd -> n_embd)
        ln_params = 2 * 2 * n_embd  # 2 LayerNorms, each with weight + bias
        attn_params = n_embd * 3 * n_embd + n_embd * n_embd  # c_attn + c_proj (no bias)
        mlp_params = n_embd * 4 * n_embd + 4 * n_embd * n_embd  # c_fc + c_proj (no bias)
        block_params = ln_params + attn_params + mlp_params
        
        # Final LayerNorm
        final_ln = 2 * n_embd
        
        expected = tok_emb + pos_emb + n_layer * block_params + final_ln
        actual = model.get_num_params()
        
        # Should be close (exact match depends on bias usage)
        assert abs(actual - expected) < expected * 0.1  # Within 10%


# =============================================================================
# LEARNING RATE SCHEDULE TESTS
# =============================================================================

class TestLRSchedule:
    """Tests for the learning rate schedule."""
    
    def test_warmup(self, config):
        """LR should increase linearly during warmup."""
        config.warmup_iters = 100
        
        lr_0 = get_lr(0, config)
        lr_50 = get_lr(50, config)
        lr_100 = get_lr(100, config)
        
        assert lr_0 == 0.0
        assert abs(lr_50 - config.learning_rate * 0.5) < 1e-6
        assert abs(lr_100 - config.learning_rate) < 1e-6
    
    def test_cosine_decay(self, config):
        """LR should follow cosine decay after warmup."""
        config.warmup_iters = 100
        config.lr_decay_iters = 1000
        
        lr_warmup_end = get_lr(100, config)
        lr_mid = get_lr(550, config)  # Middle of decay
        lr_end = get_lr(1000, config)
        
        # At warmup end: full learning rate
        assert abs(lr_warmup_end - config.learning_rate) < 1e-6
        
        # At midpoint: approximately halfway between max and min
        expected_mid = (config.learning_rate + config.min_lr) / 2
        assert abs(lr_mid - expected_mid) < config.learning_rate * 0.1
        
        # At end: minimum learning rate
        assert abs(lr_end - config.min_lr) < 1e-6
    
    def test_min_lr_floor(self, config):
        """LR should not go below min_lr."""
        config.lr_decay_iters = 1000
        
        lr_after = get_lr(2000, config)
        assert lr_after == config.min_lr


# =============================================================================
# GRADIENT ACCUMULATION CONCEPTUAL TEST
# =============================================================================

class TestGradientAccumulation:
    """Test that gradient accumulation works correctly."""
    
    def test_accumulated_equals_large_batch(self, config, device):
        """
        Accumulated gradients should equal gradients from a large batch.
        
        This is the core principle: 4 micro-batches of size 4 should give
        the same gradients as 1 batch of size 16.
        """
        torch.manual_seed(42)
        model1 = GPT(config).to(device)
        
        # Clone model for fair comparison
        torch.manual_seed(42)
        model2 = GPT(config).to(device)
        
        # Create data
        large_batch_x = torch.randint(0, config.vocab_size, (16, 10)).to(device)
        large_batch_y = torch.randint(0, config.vocab_size, (16, 10)).to(device)
        
        # Method 1: Single large batch
        model1.zero_grad()
        _, loss1 = model1(large_batch_x, large_batch_y)
        loss1.backward()
        
        # Store gradients
        grads_large = {name: p.grad.clone() for name, p in model1.named_parameters() if p.grad is not None}
        
        # Method 2: 4 accumulated micro-batches of size 4
        model2.zero_grad()
        for i in range(4):
            micro_x = large_batch_x[i*4:(i+1)*4]
            micro_y = large_batch_y[i*4:(i+1)*4]
            _, loss2 = model2(micro_x, micro_y)
            (loss2 / 4).backward()  # Scale by 1/num_accumulation_steps
        
        # Compare gradients
        for name, p in model2.named_parameters():
            if p.grad is not None:
                assert torch.allclose(p.grad, grads_large[name], atol=1e-5), \
                    f"Gradient mismatch for {name}"


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
