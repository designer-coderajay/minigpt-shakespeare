# MiniGPT-Shakespeare

A from-scratch implementation of the GPT language model architecture, trained on Shakespeare's complete works.

**This is not a library wrapper.** Every component, from LayerNorm to the training loop with gradient accumulation, is implemented from first principles to demonstrate deep understanding of transformer architectures.

## Why This Project Exists

Using `model = GPT.from_pretrained('gpt2')` tells you nothing about whether someone understands transformers. This implementation proves I can:

1. **Build the architecture** — Positional embeddings, multi-head causal attention, transformer blocks, the whole stack
2. **Train it properly** — Learning rate scheduling, weight decay, gradient accumulation for memory-constrained hardware
3. **Debug it systematically** — Comprehensive test suite comparing against PyTorch's implementations

## Architecture

```
Input Tokens
    │
    ▼
┌─────────────────────────┐
│ Token Embedding (tied)  │──────────────────────────┐
│ + Positional Embedding  │                          │
└─────────────────────────┘                          │
    │                                                │
    ▼                                                │
┌─────────────────────────┐                          │
│   Transformer Block     │                          │
│  ┌───────────────────┐  │                          │
│  │ LayerNorm         │  │                          │
│  │ Multi-Head Attn   │◄─┼── Causal mask            │
│  │ + Residual        │  │                          │
│  ├───────────────────┤  │                          │
│  │ LayerNorm         │  │                          │
│  │ MLP (4x expand)   │  │                          │
│  │ + Residual        │  │                          │
│  └───────────────────┘  │                          │
└─────────────────────────┘                          │
    │ × N layers                                     │
    ▼                                                │
┌─────────────────────────┐                          │
│    Final LayerNorm      │                          │
└─────────────────────────┘                          │
    │                                                │
    ▼                                                │
┌─────────────────────────┐                          │
│   Linear Head (tied) ◄──┼──────────────────────────┘
│   → vocab_size logits   │   Weight tying
└─────────────────────────┘
```

## Key Implementation Details

### Gradient Accumulation

When you can't fit large batches in GPU memory, you simulate them:

```python
# Instead of one batch of 256:
for micro_step in range(4):  # 4 micro-batches of 64
    loss = model(micro_batch) / 4  # Scale loss
    loss.backward()  # Gradients accumulate
optimizer.step()  # Update once with averaged gradients
```

Mathematically identical to large-batch training. Essential for training on consumer hardware.

### Pre-Norm Architecture

Unlike the original transformer (LayerNorm after attention), GPT-2 and later use Pre-Norm:

```python
# Pre-Norm (used here)
x = x + attention(layernorm(x))

# Post-Norm (original transformer)  
x = layernorm(x + attention(x))
```

Pre-Norm has better gradient flow in deep networks.

### Weight Tying

The token embedding and output projection share the same weight matrix. The embedding says "what vector represents token X" and the output projection says "what token does this vector represent", conceptually inverse operations.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/YOUR_USERNAME/minigpt-shakespeare.git
cd minigpt-shakespeare
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt

# Download dataset
python download_data.py

# Run tests (verify implementation correctness)
pytest test_minigpt.py -v

# Train
python minigpt.py --mode train

# Generate text
python minigpt.py --mode generate --prompt "ROMEO:" --max_tokens 200
```

## Training Configuration

Default hyperparameters (tuned for laptop/small GPU):

| Parameter | Value | Notes |
|-----------|-------|-------|
| Layers | 6 | Number of transformer blocks |
| Heads | 6 | Attention heads per block |
| Embedding | 384 | Hidden dimension |
| Context | 256 | Maximum sequence length |
| Batch size | 64 | Micro-batch size |
| Gradient accum | 4 | Effective batch = 256 |
| Learning rate | 3e-4 | Peak LR with warmup |
| Dropout | 0.2 | Regularization |

~10M parameters. Trains in ~30 minutes on M1 MacBook Pro.

## Experiment Tracking

Integrated with Weights & Biases:

```bash
# First time: login to wandb
wandb login

# Training automatically logs to wandb
python minigpt.py --mode train

# Disable if you prefer
python minigpt.py --mode train --no_wandb
```

## Test Coverage

The test suite verifies:

- **LayerNorm**: Output statistics, matches PyTorch implementation
- **Attention**: Causality (future tokens invisible), variable sequence lengths  
- **MLP**: Shape preservation, nonlinearity verification
- **Full model**: Loss computation, generation, weight tying
- **Gradient accumulation**: Accumulated gradients = large batch gradients
- **LR schedule**: Warmup, cosine decay, minimum floor

```bash
pytest test_minigpt.py -v
```

## Sample Output

After training:

```
ROMEO:
What say'st thou? I have done thee wrong,
And I will kiss thy lips; haply some poison
Yet doth hang on them, to make me die
With a restorative.

JULIET:
Thy lips are warm!

PRINCE:
What, ho! you men, you beasts,
That quench the fire of your pernicious rage
With purple fountains issuing from your veins
```

Not perfect Shakespeare, but recognizable iambic pentameter and character dialogue structure after just 30 minutes of training.

## File Structure

```
minigpt-shakespeare/
├── minigpt.py          # Full implementation (model + training)
├── test_minigpt.py     # Comprehensive test suite
├── download_data.py    # Dataset download script
├── requirements.txt    # Dependencies
├── data/
│   └── input.txt       # Tiny Shakespeare dataset
└── README.md
```

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Original transformer paper
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2 paper
- [Karpathy's NanoGPT](https://github.com/karpathy/nanoGPT) — Reference implementation
- [makemore series](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) — Educational background

## License

MIT

---

*Part of my from-scratch ML portfolio. See also: [micrograd-numpy](https://github.com/YOUR_USERNAME/micrograd-numpy)*
# minigpt-shakespeare
