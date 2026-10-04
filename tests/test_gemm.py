"""Project tests, independent of the course grader and PyTorch installation."""
import numpy as np
import pytest
import needle as ndl
from needle.backend_ndarray import NDArray

MODES = ("baseline", "simt", "wmma", "tensorcore")
SHAPES = [(1, 1, 1), (17, 33, 19), (63, 31, 65), (64, 32, 64),
          (65, 65, 127), (128, 72, 256), (256, 128, 128)]


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    dev = ndl.cpu() if request.param == "cpu" else ndl.cuda()
    if request.param == "cuda":
        def unavailable(reason):
            if request.config.getoption("--require-cuda"):
                pytest.fail(reason)
            pytest.skip(reason)
        if not dev.enabled():
            unavailable("CUDA backend not built")
        try:
            info = dev.gemm_device_info()
        except RuntimeError as exc:
            unavailable(f"CUDA device unavailable: {exc}")
        if info["sm_major"] < 8:
            unavailable("Project CUDA tests require SM80+")
    return dev


def reference(a, b, mode, device):
    if device.name == "cuda" and mode in ("wmma", "tensorcore"):
        a, b = a.astype(np.float16), b.astype(np.float16)
    return a.astype(np.float64) @ b.astype(np.float64)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("shape", SHAPES)
def test_matmul(device, mode, shape):
    m, k, n = shape
    rng = np.random.default_rng(7)
    a = rng.normal(size=(m, k)).astype(np.float32)
    b = rng.normal(size=(k, n)).astype(np.float32)
    with ndl.gemm_mode(mode):
        out = NDArray(a, device=device) @ NDArray(b, device=device)
    assert out.shape == (m, n)
    np.testing.assert_allclose(out.numpy(), reference(a, b, mode, device), rtol=1e-3, atol=3e-4)


@pytest.mark.parametrize("mode", MODES)
def test_transposed_and_sliced_inputs(device, mode):
    rng = np.random.default_rng(9)
    a = rng.normal(size=(35, 21)).astype(np.float32)
    b = rng.normal(size=(35, 40)).astype(np.float32)
    lhs = NDArray(a, device=device).permute((1, 0))
    rhs = NDArray(b, device=device)[:, 1:39:2]
    with ndl.gemm_mode(mode):
        result = lhs @ rhs
    np.testing.assert_allclose(result.numpy(), reference(a.T, b[:, 1:39:2], mode, device),
                               rtol=1e-3, atol=3e-4)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", ["zero", "identity"])
def test_structured_inputs(device, mode, kind):
    a = np.eye(65, dtype=np.float32) if kind == "identity" else np.zeros((65, 65), np.float32)
    b = np.arange(65 * 17, dtype=np.float32).reshape(65, 17) / 1024
    with ndl.gemm_mode(mode):
        result = NDArray(a, device=device) @ NDArray(b, device=device)
    np.testing.assert_allclose(result.numpy(), reference(a, b, mode, device), rtol=1e-3, atol=3e-4)


@pytest.mark.parametrize("mode", ["baseline", "simt"])
def test_autograd(device, mode):
    rng = np.random.default_rng(11)
    a = rng.normal(size=(17, 33)).astype(np.float32)
    b = rng.normal(size=(33, 19)).astype(np.float32)
    x, w = ndl.Tensor(a, device=device), ndl.Tensor(b, device=device)
    with ndl.gemm_mode(mode):
        (x @ w).sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), np.ones((17, 19)) @ b.T, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(w.grad.numpy(), a.T @ np.ones((17, 19)), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("mode", MODES)
def test_linear_uses_matmul_policy(device, mode):
    layer = ndl.nn.Linear(33, 19, bias=False, device=device)
    data = np.random.default_rng(4).normal(size=(17, 33)).astype(np.float32)
    with ndl.gemm_mode(mode):
        output = layer(ndl.Tensor(data, device=device)).numpy()
    np.testing.assert_allclose(output, reference(data, layer.weight.numpy(), mode, device),
                               rtol=1e-3, atol=3e-4)


def test_context_restores_on_error():
    assert ndl.get_gemm_mode() == "baseline"
    with ndl.gemm_mode("simt"):
        with pytest.raises(RuntimeError):
            with ndl.gemm_mode("tensorcore"):
                assert ndl.get_gemm_mode() == "tensorcore"
                raise RuntimeError("test")
        assert ndl.get_gemm_mode() == "simt"
    assert ndl.get_gemm_mode() == "baseline"
    with pytest.raises(ValueError):
        with ndl.gemm_mode("typo"):
            pass


def test_invalid_shapes(device):
    a = NDArray(np.ones((3, 5), np.float32), device=device)
    b = NDArray(np.ones((4, 2), np.float32), device=device)
    with ndl.gemm_mode("tensorcore"), pytest.raises(AssertionError):
        a @ b


def test_prepared_gemm(device):
    if device.name != "cuda":
        pytest.skip("PreparedGemm is a CUDA API")
    a = np.random.default_rng(1).normal(size=(65, 33)).astype(np.float32)
    b = np.random.default_rng(2).normal(size=(33, 71)).astype(np.float32)
    lhs, rhs = NDArray(a, device=device), NDArray(b, device=device)
    packed = device.PreparedGemm(lhs._handle, rhs._handle, 65, 33, 71)
    out = NDArray.make((65, 71), device=device)
    for pipeline in (False, True):
        packed.run(out._handle, pipeline)
        np.testing.assert_allclose(out.numpy(), reference(a, b, "tensorcore", device),
                                   rtol=1e-3, atol=3e-4)
        assert packed.benchmark(3, 1, pipeline) > 0
    with pytest.raises(ValueError):
        packed.benchmark(0, 1, True)
    with pytest.raises(ValueError):
        packed.run(NDArray.make((1, 1), device=device)._handle)
    packed.run_cublas(out._handle)
    np.testing.assert_allclose(out.numpy(), reference(a, b, "tensorcore", device), rtol=1e-3, atol=3e-4)
    assert packed.benchmark_cublas(3, 1) > 0


@pytest.mark.parametrize("shape", [(1, 17), (3, 257), (4, 1024)])
def test_fused_layernorm_forward(device, shape):
    x = np.random.default_rng(8).normal(size=shape).astype(np.float32)
    layer = ndl.nn.LayerNorm1d(shape[1], device=device, use_layernorm=True)
    out = layer(ndl.Tensor(x, device=device)).numpy()
    expected = (x - x.mean(axis=1, keepdims=True)) / np.sqrt(x.var(axis=1, keepdims=True) + layer.eps)
    np.testing.assert_allclose(out, expected, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("shape", [(1, 2, 17, 16), (2, 3, 33, 17)])
def test_attention_forward(device, causal, shape):
    rng = np.random.default_rng(15)
    q, k, v = [rng.normal(size=shape).astype(np.float32) for _ in range(3)]
    scores = q.astype(np.float64) @ k.astype(np.float64).swapaxes(-1, -2) / np.sqrt(shape[-1])
    if causal:
        scores = np.where(np.triu(np.ones((shape[-2], shape[-2]), bool), 1), -np.inf, scores)
    probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    reference_output = probs @ v
    attention = ndl.nn.MultiHeadAttention(dropout=0, causal=causal, device=device, use_flash_attn=True)
    attention.eval()
    actual, _ = attention(*(ndl.Tensor(t, device=device) for t in (q, k, v)))
    np.testing.assert_allclose(actual.numpy(), reference_output, rtol=2e-4, atol=2e-5)
