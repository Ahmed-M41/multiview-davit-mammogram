"""Verify that INbreast ensemble uses exactly 2 models and never calls DaViT-ALL."""

from unittest.mock import MagicMock, patch, call
import numpy as np
import torch
import torch.nn.functional as F


def _make_mock_model(prob_malignant: float = 0.7):
    """Return a mock that behaves like a DaViTClassifier."""
    m = MagicMock()
    logits = torch.tensor([[1 - prob_malignant, prob_malignant]])
    m.return_value = logits
    m.to = MagicMock(return_value=m)
    m.eval = MagicMock(return_value=m)
    m.training = False
    return m


def test_inbreast_uses_two_models_only():
    from multiview_davit.models.ensemble import MultiViewEnsemble

    cc_spec = _make_mock_model(0.8)
    mlo_spec = _make_mock_model(0.6)
    cc_all = _make_mock_model(0.9)
    mlo_all = _make_mock_model(0.85)

    ensemble = MultiViewEnsemble(
        cc_spec=cc_spec,
        mlo_spec=mlo_spec,
        cc_all=cc_all,
        mlo_all=mlo_all,
        device="cpu",
    )

    cc_img = torch.zeros(1, 3, 224, 224)
    mlo_img = torch.zeros(1, 3, 224, 224)

    # Reset call counts
    cc_spec.reset_mock()
    mlo_spec.reset_mock()
    cc_all.reset_mock()
    mlo_all.reset_mock()

    probs = ensemble.predict_inbreast(cc_img, mlo_img)

    assert cc_spec.called, "cc_spec should be called for INbreast"
    assert mlo_spec.called, "mlo_spec should be called for INbreast"
    assert not cc_all.called, "cc_all must NOT be called for INbreast (data leakage)"
    assert not mlo_all.called, "mlo_all must NOT be called for INbreast (data leakage)"
    assert probs.shape == (1, 2)


def test_cbis_uses_four_models():
    from multiview_davit.models.ensemble import MultiViewEnsemble

    cc_spec = _make_mock_model(0.7)
    mlo_spec = _make_mock_model(0.6)
    cc_all = _make_mock_model(0.8)
    mlo_all = _make_mock_model(0.75)

    ensemble = MultiViewEnsemble(
        cc_spec=cc_spec, mlo_spec=mlo_spec,
        cc_all=cc_all, mlo_all=mlo_all, device="cpu",
    )

    cc_img = torch.zeros(1, 3, 224, 224)
    mlo_img = torch.zeros(1, 3, 224, 224)

    cc_spec.reset_mock(); mlo_spec.reset_mock()
    cc_all.reset_mock(); mlo_all.reset_mock()

    probs = ensemble.predict_cbis(cc_img, mlo_img)

    assert cc_spec.called and mlo_spec.called and cc_all.called and mlo_all.called
    assert probs.shape == (1, 2)
