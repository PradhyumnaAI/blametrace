"""Milestone 0 verification tests: hooks, causal math, tasks, registry, contract spike."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from blametrace import (
    SCHEMA_VERSION,
    AuditReport,
    BaseAdapter,
    ClassificationTask,
    CulpritUnit,
    CustomAdapter,
    FailurePattern,
    HookManager,
    RegressionTask,
    SegmentationTask,
    auto_detect_adapter,
    clean_mean_ablate,
    get_adapter,
    list_adapters,
    register_adapter,
    resolve_task,
)
from blametrace.core.registry import unregister_adapter


class _ExplodingNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = torch.relu(self.fc1(x))
        raise RuntimeError("Simulated mid-forward crash")


class _PlantedLinearShortcutNet(nn.Module):
    """Deterministic 2-hidden-unit network where unit 1 is a planted shortcut."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(2, 2, bias=False)
        self.out = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.hidden.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
            self.out.weight.copy_(torch.tensor([[1.0, 5.0]]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.hidden(x))


def test_hook_manager_zero_leaks_and_eval_restoration_on_crash(
    request: pytest.FixtureRequest,
) -> None:
    model = _ExplodingNet()
    model.train(True)
    request.node.tracked_models.append(model)
    manager = HookManager(model)

    with (
        pytest.raises(RuntimeError, match="Simulated mid-forward crash"),
        manager.session(),
    ):
        assert not model.training
        manager.attach_capture(["fc1", "fc2"], capture_grads=True, capture_pre=True)
        assert manager.active_hook_count == 6
        _ = model(torch.randn(2, 4))
    assert manager.active_hook_count == 0
    assert model.training is True


def test_clean_mean_ablation_exact_causal_math(
    request: pytest.FixtureRequest,
) -> None:
    model = _PlantedLinearShortcutNet().eval()
    request.node.tracked_models.append(model)
    adapter = CustomAdapter(target_layers=["hidden"])
    task = ClassificationTask()

    # True label = 0 (negative logit). Clean samples have shortcut=0.
    fail_x = torch.tensor([[-1.0, 2.0], [-1.2, 2.0]])
    targets = torch.tensor([0, 0])
    clean_unit_means = torch.tensor([-1.1, 0.0])

    baseline_logits = model(fail_x)
    assert torch.allclose(baseline_logits, torch.tensor([[9.0], [8.8]]))
    assert bool(task.is_failure(baseline_logits, targets).all().item())

    # Ablate unit 1 (shortcut) -> logits drop to [-1.0, -1.2], 100% flip rate
    ablated_logits = clean_mean_ablate(
        model=model,
        adapter=adapter,
        layer_name="hidden",
        unit_indices=[1],
        clean_unit_means=clean_unit_means,
        failure_inputs=fail_x,
    )
    assert torch.allclose(ablated_logits, torch.tensor([[-1.0], [-1.2]]))
    assert task.recovery_metric(baseline_logits, ablated_logits, targets) == 1.0


def test_contract_spike_lstm_attention_and_mamba_round_trip() -> None:
    # 1. LSTM nested tuple: (out (B, T, H), (h_n, c_n)) with unit_dim=2 (H)
    lstm_adapter = CustomAdapter(unit_dim=2, primary_key=0)
    lstm_out = torch.randn(2, 5, 8)
    h_n, c_n = torch.randn(1, 2, 8), torch.randn(1, 2, 8)
    raw_lstm = (lstm_out, (h_n, c_n))
    canon_lstm = lstm_adapter.to_canonical(raw_lstm)
    assert canon_lstm.shape == (2, 8, 5)
    restored_lstm = lstm_adapter.from_canonical(canon_lstm, raw_lstm)
    assert isinstance(restored_lstm, tuple)
    assert torch.equal(restored_lstm[0], lstm_out)
    assert torch.equal(restored_lstm[1][0], h_n)
    assert torch.equal(restored_lstm[1][1], c_n)

    # 2. Attention head tensor: (B, Heads, L, d_head) -> (B, Heads, L*d_head)
    attn_adapter = CustomAdapter(unit_dim=1)
    raw_attn = torch.randn(2, 4, 6, 16)
    canon_attn = attn_adapter.to_canonical(raw_attn)
    assert canon_attn.shape == (2, 4, 96)
    restored_attn = attn_adapter.from_canonical(canon_attn, raw_attn)
    assert torch.equal(restored_attn, raw_attn)

    # 3. Mamba-style dict boundary output: {"mamba": (B, L, D), "gate": (B, L, 1)}
    mamba_adapter = CustomAdapter(unit_dim=2, primary_key="mamba")
    mamba_stream = torch.randn(2, 16, 32)
    gate_tensor = torch.sigmoid(torch.randn(2, 16, 1))
    raw_mamba = {"mamba": mamba_stream, "gate": gate_tensor}
    canon_mamba = mamba_adapter.to_canonical(raw_mamba)
    assert canon_mamba.shape == (2, 32, 16)
    restored_mamba = mamba_adapter.from_canonical(canon_mamba, raw_mamba)
    assert isinstance(restored_mamba, dict)
    assert torch.equal(restored_mamba["mamba"], mamba_stream)
    assert torch.equal(restored_mamba["gate"], gate_tensor)


def test_registry_priority_matching() -> None:
    class _SpecialNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(2, 2)

    @register_adapter("low_prio_test", priority=10)
    class _LowPrioAdapter(CustomAdapter):
        @classmethod
        def matches(cls, model: nn.Module) -> bool:
            return isinstance(model, _SpecialNet)

    @register_adapter("high_prio_test", priority=50)
    class _HighPrioAdapter(CustomAdapter):
        @classmethod
        def matches(cls, model: nn.Module) -> bool:
            return isinstance(model, _SpecialNet)

    try:
        detected = auto_detect_adapter(_SpecialNet())
        assert isinstance(detected, _HighPrioAdapter)
        assert isinstance(get_adapter("low_prio_test"), _LowPrioAdapter)
        names = list_adapters()
        assert names.index("high_prio_test") < names.index("low_prio_test")
        assert isinstance(auto_detect_adapter(nn.Linear(2, 2)), CustomAdapter)
    finally:
        unregister_adapter("low_prio_test")
        unregister_adapter("high_prio_test")


def test_tasks_and_report_schema() -> None:
    reg = RegressionTask(error_threshold=0.5)
    base = torch.tensor([2.0, 0.1])
    abl = torch.tensor([0.2, 0.1])
    tgt = torch.tensor([0.0, 0.0])
    assert reg.is_failure(base, tgt).tolist() == [True, False]
    assert reg.recovery_metric(base, abl, tgt) > 0.5

    seg = SegmentationTask(dice_threshold=0.7)
    seg_base = torch.tensor([[[[-1.0, -1.0], [-1.0, -1.0]]]])
    seg_abl = torch.tensor([[[[2.0, 2.0], [-1.0, -1.0]]]])
    seg_gt = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    assert seg.is_failure(seg_base, seg_gt).tolist() == [True]
    assert seg.recovery_metric(seg_base, seg_abl, seg_gt) > 0.5
    assert isinstance(resolve_task("classification"), ClassificationTask)

    culprit = CulpritUnit(
        layer_name="hidden",
        unit_index=1,
        triage_score=5.0,
        confidence_drop=10.0,
        flip_rate=1.0,
    )
    report = AuditReport(
        model_name="PlantedNet",
        adapter_name="custom",
        num_clean_samples=10,
        num_failure_samples=2,
        patterns=[
            FailurePattern(
                pattern_id=1,
                layer_name="hidden",
                culprits=[culprit],
                combined_flip_rate=1.0,
            )
        ],
    )
    assert report.schema_version == SCHEMA_VERSION
    assert isinstance(get_adapter("custom"), BaseAdapter)
