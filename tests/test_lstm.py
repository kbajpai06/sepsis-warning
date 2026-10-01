import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sepsis.data.synthetic import make_synthetic  # noqa: E402
from sepsis.features import preprocess  # noqa: E402
from sepsis.models.lstm import SepsisRNN, compute_stats, predict_rows, to_sequences  # noqa: E402


def test_sequences_shapes_and_alignment():
    pre = preprocess(make_synthetic(30, seed=2))
    stats = compute_stats(pre)
    X, y, m, groups = to_sequences(pre, stats)
    assert X.shape[0] == pre["patient_id"].nunique() and m.sum() == len(pre)
    probs = predict_rows(SepsisRNN(), pre, stats)
    assert probs.shape == (len(pre),) and not np.isnan(probs).any()


def test_rnn_is_causal():
    """Prediction at hour t must not depend on later hours."""
    pre = preprocess(make_synthetic(10, seed=4))
    stats = compute_stats(pre)
    X, _, _, _ = to_sequences(pre, stats)
    model = SepsisRNN().eval()
    x1 = torch.from_numpy(X[:1].copy())
    x2 = x1.clone()
    x2[:, 30:] += 5.0
    with torch.no_grad():
        a, b = model(x1), model(x2)
    assert torch.allclose(a[:, :30], b[:, :30], atol=1e-5)
