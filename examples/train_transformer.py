"""Train a small causal character language model using Needle end to end.

The bundled corpus makes this an offline runnable example. For real experiments,
pass --text path/to/corpus.txt. Attention and LayerNorm use differentiable Needle
operators throughout training, without the inference kernels' NumPy backward.
"""
import argparse
import math
from pathlib import Path
import time

import numpy as np
import needle as ndl
from needle import nn


class CharacterWindows(ndl.data.Dataset):
    """Non-overlapping input windows with one-character-shifted targets."""
    def __init__(self, tokens, length):
        super().__init__()
        self.tokens = np.asarray(tokens, dtype=np.int32)
        self.length = length
        if length <= 0 or len(self.tokens) <= length:
            raise ValueError("Each text split must contain more than sequence_length characters")

    def __len__(self):
        return (len(self.tokens) - 1) // self.length

    def __getitem__(self, index):
        positions = np.asarray(index)[..., None] * self.length + np.arange(self.length)
        return self.tokens[positions], self.tokens[positions + 1]


class CharacterTransformer(nn.Module):
    def __init__(self, vocab_size, sequence_length, device, dim=32, heads=2, layers=1):
        super().__init__()
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, dim, device=device)
        self.transformer = nn.Transformer(
            embedding_size=dim, hidden_size=4 * dim, num_layers=layers,
            num_head=heads, dim_head=dim // heads, sequence_len=sequence_length,
            batch_first=True, causal=True, dropout=0, device=device,
            use_flash_attn=False, use_layernorm=False,
        )
        self.norm = nn.LayerNorm1d(dim, device=device, use_layernorm=False)
        self.projection = nn.Linear(dim, vocab_size, device=device)

    def forward(self, tokens):
        hidden, _ = self.transformer(self.embedding(tokens))
        batch, length, dim = hidden.shape
        hidden = self.norm(hidden.reshape((batch * length, dim)))
        return self.projection(hidden).reshape((batch, length, self.vocab_size))


def run_epoch(model, loader, loss_fn, optimizer=None):
    model.train() if optimizer is not None else model.eval()
    loss_sum, token_count = 0.0, 0
    for inputs, targets in loader:
        if optimizer is not None:
            optimizer.reset_grad()
        logits = model(inputs)
        count = targets.shape[0] * targets.shape[1]
        loss = loss_fn(logits.reshape((count, model.vocab_size)), targets.reshape((count,)))
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        loss_sum += float(loss.numpy().item()) * count
        token_count += count
    return loss_sum / token_count


def generate(model, prompt, vocabulary, device, count):
    model.eval()
    indices = {character: i for i, character in enumerate(vocabulary)}
    if not prompt or any(c not in indices for c in prompt):
        raise ValueError("Prompt must be nonempty and contain only characters from the corpus")
    tokens = [indices[c] for c in prompt]
    for _ in range(count):
        context = np.array([tokens[-model.sequence_length:]], dtype=np.float32)
        logits = model(ndl.Tensor(context, device=device, requires_grad=False))
        tokens.append(int(logits.numpy()[0, -1].argmax()))
    return "".join(vocabulary[i] for i in tokens)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", type=Path, default=Path(__file__).parent / "data/tiny_corpus.txt")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default="the ")
    parser.add_argument("--generate", type=int, default=100)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.sequence_length, args.dim, args.heads, args.layers, args.lr) <= 0:
        parser.error("training sizes and learning rate must be positive")
    if args.dim % args.heads or args.generate < 0:
        parser.error("dim must be divisible by heads; generate must be nonnegative")
    text = args.text.read_text(encoding="utf-8")
    vocabulary = sorted(set(text))
    indices = {character: i for i, character in enumerate(vocabulary)}
    tokens = np.array([indices[c] for c in text], dtype=np.int32)
    split = int(0.9 * len(tokens))
    try:
        train_set = CharacterWindows(tokens[:split], args.sequence_length)
        valid_set = CharacterWindows(tokens[split:], args.sequence_length)
    except ValueError as error:
        parser.error(str(error))
    if args.generate and (not args.prompt or any(c not in indices for c in args.prompt)):
        parser.error("prompt must use characters present in the text")
    np.random.seed(args.seed)
    device = ndl.cuda() if args.device == "cuda" else ndl.cpu()
    if not device.enabled():
        parser.error(f"{args.device} backend is not built")
    if args.device == "cuda":
        device.synchronize()
    train_loader = ndl.data.DataLoader(train_set, args.batch_size, shuffle=True, device=device)
    valid_loader = ndl.data.DataLoader(valid_set, args.batch_size, device=device)
    model = CharacterTransformer(len(vocabulary), args.sequence_length, device,
                                 args.dim, args.heads, args.layers)
    optimizer = ndl.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.SoftmaxLoss()
    print(f"Needle character Transformer | {device} | vocab={len(vocabulary)} "
          f"train_windows={len(train_set)} valid_windows={len(valid_set)} | FP32", flush=True)
    initial = run_epoch(model, valid_loader, loss_fn)
    print(f"Initial: valid_loss={initial:.4f} perplexity={math.exp(initial):.2f}", flush=True)
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_loss = run_epoch(model, train_loader, loss_fn, optimizer)
        valid_loss = run_epoch(model, valid_loader, loss_fn)
        print(f"Epoch {epoch:02d}/{args.epochs}: train_loss={train_loss:.4f} "
              f"valid_loss={valid_loss:.4f} perplexity={math.exp(valid_loss):.2f} "
              f"time={time.perf_counter() - start:.1f}s", flush=True)
    if args.generate:
        print("\nGreedy sample:\n" + generate(model, args.prompt, vocabulary, device, args.generate))


if __name__ == "__main__":
    main()
