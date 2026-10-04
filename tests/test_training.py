import gzip
import struct

import numpy as np
import pytest
import needle as ndl

from examples.train_mnist import evaluate, make_model, train_epoch
from examples.train_transformer import CharacterTransformer, CharacterWindows, generate, run_epoch


def write_idx(tmp_path, labels_count=3):
    images_path, labels_path = tmp_path / "images.gz", tmp_path / "labels.gz"
    pixels = np.arange(12, dtype=np.uint8).reshape(3, 2, 2)
    with gzip.open(images_path, "wb") as f:
        f.write(struct.pack(">IIII", 2051, 3, 2, 2) + pixels.tobytes())
    with gzip.open(labels_path, "wb") as f:
        f.write(struct.pack(">II", 2049, labels_count) + bytes(range(labels_count)))
    return images_path, labels_path, pixels


def test_mnist_dataset_and_device_batches(tmp_path):
    images, labels, pixels = write_idx(tmp_path)
    dataset = ndl.data.MNISTDataset(images, labels)
    assert len(dataset) == 3
    x, y = dataset[np.array([2, 0])]
    np.testing.assert_allclose(x, pixels[[2, 0]].reshape(2, 4) / 255)
    np.testing.assert_array_equal(y, [2, 0])
    assert x.dtype == np.float32
    batches = list(ndl.data.DataLoader(dataset, batch_size=2, device=ndl.cpu()))
    assert [x.shape for x, _ in batches] == [(2, 4), (1, 4)]
    for x, y in batches:
        assert x.device == y.device == ndl.cpu()
        assert not x.requires_grad and not y.requires_grad


def test_mnist_rejects_mismatched_labels(tmp_path):
    images, labels, _ = write_idx(tmp_path, labels_count=2)
    with pytest.raises(ValueError, match="labels"):
        ndl.data.MNISTDataset(images, labels)


def test_transforms_apply_per_image_without_mutating_dataset(tmp_path):
    images, labels, _ = write_idx(tmp_path)
    def transform(image):
        assert image.shape == (2, 2, 1)
        image[:] += 1
        return image
    dataset = ndl.data.MNISTDataset(images, labels, transforms=[transform])
    original = dataset.images.copy()
    batch, _ = dataset[[0, 1]]
    single, _ = dataset[0]
    np.testing.assert_allclose(batch[0], single)
    np.testing.assert_array_equal(dataset.images, original)


def test_loader_shuffle_covers_samples_once_each_epoch():
    data = ndl.data.NDArrayDataset(np.arange(17).reshape(17, 1), np.arange(17))
    loader = ndl.data.DataLoader(data, batch_size=5, shuffle=True, device=ndl.cpu())
    np.random.seed(3)
    epochs = [[int(v) for _, y in loader for v in y.numpy()] for _ in range(2)]
    assert len(loader) == 4
    assert sorted(epochs[0]) == sorted(epochs[1]) == list(range(17))
    assert epochs[0] != epochs[1]
    with pytest.raises(ValueError):
        ndl.data.DataLoader(data, batch_size=0)


def test_example_trains_through_needle_autograd_and_optimizer():
    np.random.seed(0)
    rng = np.random.default_rng(7)
    x = np.zeros((64, 784), dtype=np.float32)
    x[:, :2] = rng.normal(size=(64, 2))
    y = (x[:, 0] > 0).astype(np.uint8)
    dataset = ndl.data.NDArrayDataset(x, y)
    loader = ndl.data.DataLoader(dataset, batch_size=16, shuffle=True, device=ndl.cpu())
    model = make_model(ndl.cpu(), hidden=16)
    initial_weights = [p.numpy().copy() for p in model.parameters()]
    loss_fn = ndl.nn.SoftmaxLoss()
    optimizer = ndl.optim.SGD(model.parameters(), lr=0.3)
    before, _ = evaluate(model, loader, loss_fn)
    for _ in range(10):
        assert np.isfinite(train_epoch(model, loader, loss_fn, optimizer))
    after, accuracy = evaluate(model, loader, loss_fn)
    assert after < before * 0.3
    assert accuracy >= 0.9
    assert any(not np.array_equal(w, p.numpy()) for w, p in zip(initial_weights, model.parameters()))
    assert all(p.grad is not None and np.isfinite(p.grad.numpy()).all() for p in model.parameters())


def test_character_windows_shift_targets():
    dataset = CharacterWindows(np.arange(14), length=4)
    inputs, targets = dataset[[2, 0]]
    np.testing.assert_array_equal(inputs, [[8, 9, 10, 11], [0, 1, 2, 3]])
    np.testing.assert_array_equal(targets, inputs + 1)
    assert len(dataset) == 3


def test_transformer_is_causal_and_trains_all_parameters():
    np.random.seed(0)
    device = ndl.cpu()
    model = CharacterTransformer(3, sequence_length=6, device=device, dim=8, heads=2)
    model.eval()
    tokens = np.array([[0, 1, 2, 0, 1, 2]], dtype=np.float32)
    changed = tokens.copy()
    changed[:, 3:] = [2, 2, 0]
    before_suffix_change = model(ndl.Tensor(tokens, device=device)).numpy()
    after_suffix_change = model(ndl.Tensor(changed, device=device)).numpy()
    np.testing.assert_allclose(before_suffix_change[:, :3], after_suffix_change[:, :3],
                               rtol=1e-5, atol=1e-5)

    dataset = CharacterWindows(np.tile([0, 1, 2], 17), length=6)
    loader = ndl.data.DataLoader(dataset, batch_size=8, device=device)
    loss_fn = ndl.nn.SoftmaxLoss()
    optimizer = ndl.optim.Adam(model.parameters(), lr=0.02)
    initial = run_epoch(model, loader, loss_fn)
    weights = [p.numpy().copy() for p in model.parameters()]
    for _ in range(20):
        run_epoch(model, loader, loss_fn, optimizer)
    final = run_epoch(model, loader, loss_fn)
    assert final < initial * 0.2
    assert all(p.grad is not None and np.isfinite(p.grad.numpy()).all() for p in model.parameters())
    assert all(not np.array_equal(w, p.numpy()) for w, p in zip(weights, model.parameters()))
    sample = generate(model, "abc", list("abc"), device, count=8)
    assert sample.startswith("abc") and len(sample) == 11 and set(sample) <= set("abc")
