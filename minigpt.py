"""
MiniGPT-Shakespeare: A Complete GPT Language Model Implementation
==================================================================

This is a from-scratch GPT implementation trained on Tiny Shakespeare.
Built to demonstrate deep understanding of the transformer architecture.

Architecture Overview:
---------------------
Input tokens → Token Embedding + Positional Embedding
            → N × Transformer Blocks
            → LayerNorm → Linear Head → Logits

Each Transformer Block:
    x → LayerNorm → Multi-Head Attention → + x (residual)
      → LayerNorm → Feed-Forward MLP → + x (residual)

Key Concepts Implemented:
- Causal (autoregressive) self-attention
- Pre-norm architecture (LayerNorm before attention/MLP, not after)
- Gradient accumulation for simulating larger batch sizes
- Weight tying between token embedding and output head

Author: Ajay
Reference: Karpathy's NanoGPT, "Attention Is All You Need" paper
"""

import math
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Optional: wandb for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed. Run 'pip install wandb' for experiment tracking.")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class GPTConfig:
    """
    Configuration for the GPT model.
    
    These hyperparameters define the model architecture.
    The defaults are sized for training on a laptop/small GPU.
    """
    # Model architecture
    vocab_size: int = 65          # Character-level for Shakespeare (unique chars)
    block_size: int = 256         # Maximum context length (sequence length)
    n_layer: int = 6              # Number of transformer blocks
    n_head: int = 6               # Number of attention heads
    n_embd: int = 384             # Embedding dimension (must be divisible by n_head)
    dropout: float = 0.2          # Dropout probability
    
    # Training hyperparameters
    batch_size: int = 64          # Micro-batch size (fits in memory)
    gradient_accumulation_steps: int = 4  # Effective batch = batch_size * this
    learning_rate: float = 3e-4   # Peak learning rate
    max_iters: int = 5000         # Total training iterations
    warmup_iters: int = 100       # Linear warmup steps
    lr_decay_iters: int = 5000    # Should be ~= max_iters
    min_lr: float = 3e-5          # Minimum learning rate (10% of peak)
    weight_decay: float = 0.1     # L2 regularization
    
    # Evaluation
    eval_interval: int = 250      # How often to evaluate
    eval_iters: int = 200         # Batches to average for eval loss
    
    # Logging
    use_wandb: bool = True        # Enable wandb logging
    wandb_project: str = "minigpt-shakespeare"
    wandb_run_name: str = "gpt-tiny"
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.n_embd % self.n_head == 0, \
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"


# =============================================================================
# MODEL COMPONENTS
# =============================================================================

class LayerNorm(nn.Module):
    """
    Layer Normalization with optional bias.
    
    Why LayerNorm over BatchNorm?
    - BatchNorm normalizes across the batch dimension
    - LayerNorm normalizes across the feature dimension
    - For sequences of varying lengths, LayerNorm is more stable
    - No dependency on batch statistics at inference time
    
    The operation:
        y = (x - mean) / sqrt(var + eps) * gamma + beta
    
    Where gamma (weight) and beta (bias) are learned parameters.
    """
    
    def __init__(self, ndim: int, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ndim))   # gamma, initialized to 1
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None  # beta
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize over the last dimension (features)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Scale and shift with learned parameters
        out = self.weight * x_norm
        if self.bias is not None:
            out = out + self.bias
        return out


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention.
    
    This is the core of the transformer. Each token attends to all previous
    tokens (including itself) but cannot see future tokens. This causality
    is enforced by masking.
    
    The attention mechanism:
        1. Project input to Q, K, V (query, key, value)
        2. Split into multiple heads
        3. Compute attention scores: softmax(QK^T / sqrt(d_k))
        4. Apply causal mask (set future positions to -inf before softmax)
        5. Weighted sum of values
        6. Concatenate heads and project back
    
    Why multiple heads?
    - Different heads can learn different types of relationships
    - One head might learn syntactic patterns, another semantic ones
    - Increases model capacity without increasing depth
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        
        # Combined projection for Q, K, V (more efficient than 3 separate)
        # Output size is 3 * n_embd because we compute Q, K, V together
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Causal mask: lower triangular matrix
        # This is registered as a buffer (not a parameter) because it's not learned
        # Shape: (1, 1, block_size, block_size) for broadcasting
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of multi-head causal self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd)
        """
        B, T, C = x.shape  # batch, sequence length, embedding dim
        
        # Compute Q, K, V in one matrix multiplication
        # Shape: (B, T, 3 * n_embd) -> split into 3 tensors of (B, T, n_embd)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention
        # (B, T, n_embd) -> (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Attention scores: (B, n_head, T, head_dim) @ (B, n_head, head_dim, T)
        # Result shape: (B, n_head, T, T)
        # Scale by 1/sqrt(d_k) to prevent softmax saturation
        scale = 1.0 / math.sqrt(self.head_dim)
        att = (q @ k.transpose(-2, -1)) * scale
        
        # Apply causal mask: set future positions to -inf
        # After softmax, -inf becomes 0, so future tokens don't contribute
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        
        # Softmax over the last dimension (keys)
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Weighted sum of values
        # (B, n_head, T, T) @ (B, n_head, T, head_dim) -> (B, n_head, T, head_dim)
        out = att @ v
        
        # Reshape back: (B, n_head, T, head_dim) -> (B, T, n_head, head_dim) -> (B, T, n_embd)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection and dropout
        out = self.resid_dropout(self.c_proj(out))
        
        return out


class MLP(nn.Module):
    """
    Feed-Forward Network (MLP) used in each transformer block.
    
    Architecture: Linear -> GELU -> Linear
    
    The hidden dimension is traditionally 4x the embedding dimension.
    This expansion allows the network to learn more complex transformations.
    
    Why GELU over ReLU?
    - GELU (Gaussian Error Linear Unit) is smoother than ReLU
    - Better gradient flow (no "dead neurons" like ReLU)
    - Empirically works better for transformers
    - GELU(x) ≈ x * sigmoid(1.702 * x)
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        # First linear: expand from n_embd to 4 * n_embd
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        # Second linear: project back to n_embd
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x)  # Using the exact GELU, not approximation
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    A single transformer block.
    
    This implements the Pre-Norm architecture (used in GPT-2 and later):
        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))
    
    Why Pre-Norm instead of Post-Norm (original transformer)?
    - More stable training, especially for deep networks
    - Gradients flow more easily through residual connections
    - The original transformer had vanishing gradient issues at depth
    
    The residual connections are crucial:
    - They allow gradients to flow directly through the network
    - Each block can learn a "refinement" of the representation
    - Without them, deep networks are very hard to train
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention with residual connection
        x = x + self.attn(self.ln_1(x))
        # MLP with residual connection
        x = x + self.mlp(self.ln_2(x))
        return x


# =============================================================================
# THE FULL GPT MODEL
# =============================================================================

class GPT(nn.Module):
    """
    The complete GPT Language Model.
    
    Architecture:
        1. Token Embedding: vocab_size -> n_embd
        2. Positional Embedding: block_size -> n_embd
        3. Dropout
        4. N × TransformerBlock
        5. LayerNorm
        6. Linear Head: n_embd -> vocab_size (tied with token embedding)
    
    Weight Tying:
    The output projection (lm_head) shares weights with the token embedding.
    This makes sense because:
    - Token embedding: "what vector represents this token?"
    - Output projection: "what token does this vector represent?"
    These are conceptually inverse operations, so sharing weights works well
    and reduces parameter count significantly.
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict({
            # Token embeddings: map token IDs to vectors
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            # Positional embeddings: learned (not sinusoidal like original transformer)
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            # Dropout after embeddings
            'drop': nn.Dropout(config.dropout),
            # Stack of transformer blocks
            'h': nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            # Final layer norm (Pre-Norm architecture requires this at the end)
            'ln_f': LayerNorm(config.n_embd),
        })
        
        # Output head: project from embedding space to vocabulary
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying: share weights between token embedding and output head
        # This is done by making them point to the same tensor
        self.transformer.wte.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply special scaled initialization to residual projections
        # This helps with training stability for deep networks
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                # Scale by 1/sqrt(2 * n_layer) for residual connections
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        
        # Print model size
        n_params = self.get_num_params()
        print(f"Model initialized with {n_params / 1e6:.2f}M parameters")
    
    def _init_weights(self, module):
        """
        Initialize weights using the GPT-2 scheme.
        
        - Linear layers: Normal(0, 0.02)
        - Embeddings: Normal(0, 0.02)
        - LayerNorm: weight=1, bias=0
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def get_num_params(self, non_embedding: bool = False) -> int:
        """
        Return the number of parameters in the model.
        
        Args:
            non_embedding: If True, subtract position embeddings (for comparison
                          with models that use sinusoidal embeddings)
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
    
    def forward(
        self, 
        idx: torch.Tensor, 
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the GPT model.
        
        Args:
            idx: Input token indices, shape (batch_size, seq_len)
            targets: Target token indices for loss computation, shape (batch_size, seq_len)
        
        Returns:
            logits: Output logits, shape (batch_size, seq_len, vocab_size)
            loss: Cross-entropy loss if targets provided, else None
        """
        device = idx.device
        B, T = idx.shape
        
        assert T <= self.config.block_size, \
            f"Sequence length {T} exceeds block size {self.config.block_size}"
        
        # Get position indices: [0, 1, 2, ..., T-1]
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        
        # Forward through embeddings
        tok_emb = self.transformer.wte(idx)      # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)      # (T, n_embd) -> broadcasts to (B, T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Forward through transformer blocks
        for block in self.transformer.h:
            x = block(x)
        
        # Final layer norm
        x = self.transformer.ln_f(x)
        
        # Compute logits
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Flatten for cross-entropy: (B*T, vocab_size) and (B*T,)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1  # Ignore padding tokens if any
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate new tokens autoregressively.
        
        Args:
            idx: Starting token indices, shape (batch_size, seq_len)
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k most likely tokens
        
        Returns:
            Generated token indices, shape (batch_size, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop to block_size if sequence is too long
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            
            # Get logits for the last position only
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat([idx, idx_next], dim=1)
        
        return idx


# =============================================================================
# DATASET
# =============================================================================

class ShakespeareDataset(Dataset):
    """
    Character-level Shakespeare dataset.
    
    This loads the Tiny Shakespeare dataset and provides random chunks
    of text for training. Each sample is a sequence of block_size + 1
    characters (input and target).
    """
    
    def __init__(self, data_path: str, block_size: int, split: str = 'train'):
        """
        Args:
            data_path: Path to the text file
            block_size: Maximum sequence length
            split: 'train' or 'val'
        """
        self.block_size = block_size
        
        # Load text
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Create character-level vocabulary
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        # Encode all text
        data = torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)
        
        # Split into train/val (90/10)
        n = int(0.9 * len(data))
        self.data = data[:n] if split == 'train' else data[n:]
        
        print(f"{split} dataset: {len(self.data):,} characters, vocab size: {self.vocab_size}")
    
    def __len__(self):
        # Number of possible starting positions
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        # Get a chunk of block_size + 1 characters
        chunk = self.data[idx:idx + self.block_size + 1]
        x = chunk[:-1]  # Input: all but last
        y = chunk[1:]   # Target: all but first (shifted by 1)
        return x, y
    
    def encode(self, text: str) -> torch.Tensor:
        """Convert text to tensor of token indices."""
        return torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)
    
    def decode(self, indices: torch.Tensor) -> str:
        """Convert tensor of token indices back to text."""
        return ''.join([self.itos[i] for i in indices.tolist()])


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def get_lr(it: int, config: GPTConfig) -> float:
    """
    Learning rate schedule with linear warmup and cosine decay.
    
    The schedule:
    1. Linear warmup from 0 to learning_rate over warmup_iters
    2. Cosine decay from learning_rate to min_lr over remaining iterations
    
    Why this schedule?
    - Warmup: Prevents early training instability when gradients are large
    - Cosine decay: Smooth decrease allows fine-tuning at the end
    """
    # Linear warmup
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    
    # After decay period, return minimum
    if it > config.lr_decay_iters:
        return config.min_lr
    
    # Cosine decay
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # Goes from 1 to 0
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


@torch.no_grad()
def estimate_loss(
    model: GPT,
    train_dataset: ShakespeareDataset,
    val_dataset: ShakespeareDataset,
    config: GPTConfig,
    device: torch.device
) -> dict:
    """
    Estimate loss on train and val sets by averaging over multiple batches.
    
    We do this because a single batch might not be representative.
    """
    model.eval()
    out = {}
    
    for split, dataset in [('train', train_dataset), ('val', val_dataset)]:
        losses = []
        for _ in range(config.eval_iters):
            # Random batch
            indices = torch.randint(len(dataset), (config.batch_size,))
            x = torch.stack([dataset[i][0] for i in indices]).to(device)
            y = torch.stack([dataset[i][1] for i in indices]).to(device)
            
            _, loss = model(x, y)
            losses.append(loss.item())
        
        out[split] = sum(losses) / len(losses)
    
    model.train()
    return out


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def train(config: GPTConfig, data_path: str = 'data/input.txt'):
    """
    Main training function with gradient accumulation.
    
    Gradient Accumulation Explained:
    --------------------------------
    When you can't fit large batches in memory, you can simulate them by:
    1. Running multiple small batches (micro-batches)
    2. Accumulating gradients without updating weights
    3. Averaging gradients and updating once per N micro-batches
    
    Mathematically equivalent to a large batch:
    - Large batch of 256: compute loss, backprop, update
    - 4 micro-batches of 64: compute loss 4 times, accumulate gradients, update once
    
    The key insight: gradients are additive, so we can sum them up.
    We divide by gradient_accumulation_steps to get the mean.
    """
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 
                          'mps' if False else 'cpu')
    print(f"Using device: {device}")
    
    # Enable TF32 for faster training on Ampere GPUs
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Load datasets
    train_dataset = ShakespeareDataset(data_path, config.block_size, split='train')
    val_dataset = ShakespeareDataset(data_path, config.block_size, split='val')
    
    # Update vocab size from actual data
    config.vocab_size = train_dataset.vocab_size
    
    # Create model
    model = GPT(config).to(device)
    
    # Optimizer with weight decay
    # We separate parameters into those that should have weight decay and those that shouldn't
    # Weight decay should NOT be applied to biases, LayerNorm, and embeddings
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.dim() >= 2:  # Weight matrices
                decay_params.append(param)
            else:  # Biases, LayerNorm parameters
                no_decay_params.append(param)
    
    optim_groups = [
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    # AdamW is Adam with proper weight decay (not L2 regularization)
    optimizer = torch.optim.AdamW(optim_groups, lr=config.learning_rate, betas=(0.9, 0.95))
    
    # Initialize wandb
    if config.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=vars(config)
        )
    
    # Training loop
    print(f"\nStarting training for {config.max_iters} iterations...")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"Gradient accumulation steps: {config.gradient_accumulation_steps}")
    print("-" * 50)
    
    model.train()
    start_time = time.time()
    
    for iter_num in range(config.max_iters):
        # Update learning rate
        lr = get_lr(iter_num, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluation
        if iter_num % config.eval_interval == 0:
            losses = estimate_loss(model, train_dataset, val_dataset, config, device)
            elapsed = time.time() - start_time
            print(f"iter {iter_num:5d} | train loss {losses['train']:.4f} | "
                  f"val loss {losses['val']:.4f} | lr {lr:.2e} | time {elapsed:.1f}s")
            
            if config.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    'iter': iter_num,
                    'train/loss': losses['train'],
                    'val/loss': losses['val'],
                    'lr': lr,
                })
            
            # Generate sample
            if iter_num > 0:
                model.eval()
                prompt = train_dataset.encode("\n").unsqueeze(0).to(device)
                generated = model.generate(prompt, max_new_tokens=100, temperature=0.8)
                sample = train_dataset.decode(generated[0])
                print(f"Sample: {sample[:200]}...")
                model.train()
        
        # Gradient accumulation loop
        optimizer.zero_grad()
        accumulated_loss = 0.0
        
        for micro_step in range(config.gradient_accumulation_steps):
            # Get random batch
            indices = torch.randint(len(train_dataset), (config.batch_size,))
            x = torch.stack([train_dataset[i][0] for i in indices]).to(device)
            y = torch.stack([train_dataset[i][1] for i in indices]).to(device)
            
            # Forward pass
            _, loss = model(x, y)
            
            # Scale loss by accumulation steps (so gradients are averaged)
            loss = loss / config.gradient_accumulation_steps
            accumulated_loss += loss.item()
            
            # Backward pass (accumulates gradients)
            loss.backward()
        
        # Gradient clipping (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights (only once per gradient_accumulation_steps micro-batches)
        optimizer.step()
        
        # Log accumulated loss
        if config.use_wandb and WANDB_AVAILABLE and iter_num % 10 == 0:
            wandb.log({'train/loss_step': accumulated_loss, 'iter': iter_num})
    
    # Final evaluation
    losses = estimate_loss(model, train_dataset, val_dataset, config, device)
    print(f"\nFinal: train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")
    
    # Save model
    save_path = 'minigpt_shakespeare.pt'
    torch.save({
        'model': model.state_dict(),
        'config': config,
        'vocab': {'stoi': train_dataset.stoi, 'itos': train_dataset.itos}
    }, save_path)
    print(f"Model saved to {save_path}")
    
    if config.use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    return model, train_dataset


# =============================================================================
# INFERENCE / GENERATION
# =============================================================================

def generate_text(
    checkpoint_path: str = 'minigpt_shakespeare.pt',
    prompt: str = "\n",
    max_tokens: int = 500,
    temperature: float = 0.8,
    top_k: int = 40
) -> str:
    """Load a trained model and generate text."""
    device = torch.device('cuda' if torch.cuda.is_available() else 
                          'mps' if False else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    vocab = checkpoint['vocab']
    
    # Recreate model
    model = GPT(config).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Encode prompt
    stoi = vocab['stoi']
    itos = vocab['itos']
    idx = torch.tensor([[stoi[ch] for ch in prompt]], dtype=torch.long, device=device)
    
    # Generate
    with torch.no_grad():
        generated = model.generate(idx, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)
    
    # Decode
    return ''.join([itos[i] for i in generated[0].tolist()])


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='MiniGPT Shakespeare')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'generate'])
    parser.add_argument('--data', type=str, default='data/input.txt', help='Path to training data')
    parser.add_argument('--checkpoint', type=str, default='minigpt_shakespeare.pt')
    parser.add_argument('--prompt', type=str, default='\n', help='Generation prompt')
    parser.add_argument('--max_tokens', type=int, default=500)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    args = parser.parse_args()
    
    if args.mode == 'train':
        config = GPTConfig()
        if args.no_wandb:
            config.use_wandb = False
        train(config, args.data)
    else:
        text = generate_text(
            args.checkpoint,
            args.prompt,
            args.max_tokens,
            args.temperature
        )
        print(text)
