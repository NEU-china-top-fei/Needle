"""Train a 784 -> 128 -> 10 MLP with Needle, without PyTorch or manual gradients.

Usage (after building and setting PYTHONPATH):
    python examples/train_mnist.py --device cpu --download
    python examples/train_mnist.py --device cuda --epochs 5

NumPy is used only for data preparation, seeding, and reporting accuracy.
Forward, cross-entropy, backward, and parameter updates all use Needle.
"""
import argparse
import hashlib
from pathlib import Path
import shutil
import time
import urllib.request

import numpy as np
import needle as ndl
from needle import nn


# Public MNIST mirror and original file checksums (also used by torchvision).
MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def prepare_data(directory, download=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, checksum in FILES.items():
        path = directory / name
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"Missing {path}; run once with --download")
            print(f"Downloading {name}...", flush=True)
            temporary = path.with_suffix(".part")
            try:
                with urllib.request.urlopen(MNIST_URL + name, timeout=60) as response:
                    with temporary.open("wb") as output:
                        shutil.copyfileobj(response, output)
                if hashlib.md5(temporary.read_bytes()).hexdigest() != checksum:
                    raise ValueError(f"Checksum mismatch for {name}")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        if hashlib.md5(path.read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Checksum mismatch for {path}; remove it and download again")
    return (
        ndl.data.MNISTDataset(directory / "train-images-idx3-ubyte.gz",
                              directory / "train-labels-idx1-ubyte.gz"),
        ndl.data.MNISTDataset(directory / "t10k-images-idx3-ubyte.gz",
                              directory / "t10k-labels-idx1-ubyte.gz"),
    )


def make_model(device, hidden=128):
    return nn.Sequential(
        nn.Linear(784, hidden, device=device),
        nn.ReLU(),
        nn.Linear(hidden, 10, device=device),
    )


def train_epoch(model, loader, loss_fn, optimizer):
    model.train()
    total_loss, count = 0.0, 0
    for images, labels in loader:
        optimizer.reset_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.numpy().item()) * images.shape[0]
        count += images.shape[0]
    return total_loss / count


def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    for images, labels in loader:
        logits = model(images)
        total_loss += float(loss_fn(logits, labels).numpy().item()) * images.shape[0]
        correct += int((logits.numpy().argmax(axis=1) == labels.numpy()).sum())
        count += images.shape[0]
    return total_loss / count, correct / count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/mnist")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-limit", type=int, default=0, help="0 uses all 60,000 training examples")
    parser.add_argument("--test-limit", type=int, default=0, help="0 uses all 10,000 test examples")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.hidden, args.lr) <= 0:
        parser.error("epochs, batch size, hidden size and learning rate must be positive")
    if min(args.train_limit, args.test_limit) < 0:
        parser.error("data limits must be nonnegative")
    np.random.seed(args.seed)
    device = ndl.cuda() if args.device == "cuda" else ndl.cpu()
    if not device.enabled():
        parser.error(f"{args.device} backend is not built")
    if args.device == "cuda":
        device.synchronize()  # Fail early on an unavailable driver/device.
    train_set, test_set = prepare_data(args.data_dir, args.download)
    for dataset, limit in ((train_set, args.train_limit), (test_set, args.test_limit)):
        if limit:
            dataset.images, dataset.labels = dataset.images[:limit], dataset.labels[:limit]
    train_loader = ndl.data.DataLoader(train_set, args.batch_size, shuffle=True, device=device)
    test_loader = ndl.data.DataLoader(test_set, args.batch_size, device=device)
    model = make_model(device, args.hidden)
    loss_fn = nn.SoftmaxLoss()
    optimizer = ndl.optim.SGD(model.parameters(), lr=args.lr)
    print(f"Needle MLP 784-{args.hidden}-10 | {device} | "
          f"train={len(train_set)}, test={len(test_set)} | FP32", flush=True)
    loss, accuracy = evaluate(model, test_loader, loss_fn)
    print(f"Initial: test_loss={loss:.4f} test_accuracy={accuracy:.2%}", flush=True)
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer)
        test_loss, accuracy = evaluate(model, test_loader, loss_fn)
        print(f"Epoch {epoch:02d}/{args.epochs}: train_loss={train_loss:.4f} "
              f"test_loss={test_loss:.4f} test_accuracy={accuracy:.2%} "
              f"time={time.perf_counter() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
