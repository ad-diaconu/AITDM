"""fl_app/aitdm_fl/task.py (models, metrics, FedBN state helpers, local training) and the
naming / config helpers of experiments/run_seeds.py and summarize_fl.py."""

import numpy as np
import pytest
import torch
from aitdm_fl.task import (CLASSES, balanced_class_weights, bn_state, build_model, confusion,
                           load_state, metrics, scores, shared_state, train, uses_local_bn)
from run_seeds import method_name, parse_overrides
from summarize_fl import fedavg_reference


@pytest.mark.parametrize("name", ["fcn", "lstm"])
def test_models_output_one_logit_per_class(name):
    out = build_model(name)(torch.zeros(4, 200, 3))
    assert out.shape == (4, len(CLASSES))


def test_scores_from_confusion_matrix():
    y = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 1, 1, 1, 2, 0])
    s = metrics(y, pred)
    assert s["accuracy"] == pytest.approx(4 / 6)
    assert s["balanced_accuracy"] == pytest.approx((0.5 + 1 + 0.5) / 3)  # only the 3 classes present
    assert s == scores(confusion(y, pred))
    assert scores(np.zeros((6, 6)))["f1_macro"] == 0.0  # empty client: no crash


def test_fedbn_state_helpers():
    fcn = build_model("fcn")
    shared = shared_state(fcn)
    assert not any(k.endswith("num_batches_tracked") for k in shared)
    bn = bn_state(fcn)
    assert bn and all(".features." in f".{k}" for k in bn)  # only the BatchNorm layers
    assert bn_state(build_model("lstm")) == {}
    other = build_model("fcn")
    load_state(other, bn, partial=True)
    assert all(torch.equal(other.state_dict()[k], v) for k, v in bn.items())
    with pytest.raises(AssertionError):
        load_state(other, bn)  # a partial state needs partial=True


def test_local_bn_switch():
    assert uses_local_bn({"strategy": "fedbn", "local-bn": False})
    assert uses_local_bn({"strategy": "krum", "local-bn": True})
    assert not uses_local_bn({"strategy": "fedavg", "local-bn": "false"})


def test_training_is_reproducible_and_prox_is_a_no_op_at_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 200, 3)).astype(np.float32)
    y = rng.integers(0, 6, size=64)

    def run(**kw):
        torch.manual_seed(0)
        model = build_model("fcn")
        loss = train(model, x, y, epochs=1, lr=1e-3, batch_size=16, clip=1.0, seed=1,
                     device=torch.device("cpu"), **kw)
        return loss, shared_state(model)

    (l1, s1), (l2, s2) = run(), run(proximal_mu=0.0, class_weights=None)
    assert l1 == l2 and all(torch.equal(s1[k], s2[k]) for k in s1)
    l3, _ = run(proximal_mu=1.0)
    assert l3 != l1


def test_balanced_class_weights():
    w = balanced_class_weights(np.array([0, 0, 0, 1]))
    assert w[0] < w[1] and w[2] == 1.0  # absent classes keep weight 1


def test_method_names_and_fedavg_reference():
    assert method_name("fcn", "fedbn", False, "") == "fcn-fedbn"
    assert method_name("fcn", "krum", True, "signflip") == "fcn-krum-localbn-signflip"
    assert fedavg_reference("fcn-fedbn-e3", "fcn", "fedbn") == "fcn-fedavg-e3"
    assert fedavg_reference("fcn-krum-localbn-signflip", "fcn", "krum") == "fcn-fedavg-signflip"
    assert fedavg_reference("lstm-fedavg-r80", "lstm", "fedavg") is None


def test_parse_overrides():
    cfg = parse_overrides("dp-noise-multiplier=1.0 malicious-clients='1,3' attack='noise'")
    assert cfg == {"dp-noise-multiplier": 1.0, "malicious-clients": "1,3", "attack": "noise"}
    with pytest.raises(SystemExit):
        parse_overrides("seed=3")  # set with --seeds
    with pytest.raises(SystemExit):
        parse_overrides("dp-noise-multipler=1.0")  # typo: would be ignored silently
