from pathlib import Path
from types import SimpleNamespace

import importlib.util
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "train_fdt_v4_curriculum_speed_observable.py"
SPEC = importlib.util.spec_from_file_location("fdt_v4_observable", SOURCE)
observable = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(observable)


def test_generated_prefix_schedule_and_training_math_are_unchanged():
    source = SOURCE.read_text(encoding="utf-8")
    due = source.index("generated_prefix_due = (")
    backward = source.index("prefix_loss.backward()", due)
    capture = source.index("pending_generated_prefix_observation = {", backward)

    assert 'step % int(train_cfg.get("generated_prefix_every_steps", 1)) == 0' in source
    assert backward < capture
    assert '"optimizer_step": step + 1' in source[capture:]


def test_generated_prefix_metrics_are_deferred_to_the_next_train_log():
    source = SOURCE.read_text(encoding="utf-8")
    capture = source.index("pending_generated_prefix_observation = {")
    log = source.index('"recent_generated_prefix": recent_generated_prefix')
    clear = source.index("pending_generated_prefix_observation = None", capture)

    assert capture < clear < log
    assert '"loss": prefix_loss.detach()' in source[capture:clear]
    assert "value.detach()" in source[capture:clear]


def test_interval_throughput_is_reported_without_replacing_session_contract():
    source = SOURCE.read_text(encoding="utf-8")

    assert "last_log_time = started" in source
    assert "last_log_tokens = tokens_seen" in source
    assert "interval_tps = interval_tokens / interval_elapsed" in source
    assert '"interval_tokens_per_sec": interval_tps' in source
    assert '"interval_throughput_contract": "wall_clock_between_train_logs_v1"' in source
    assert '"throughput_contract": "session_local_v2"' in source


class _ExactObjectiveModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Embedding(16, 4)
        self.pointer = torch.nn.Linear(4, 1, bias=False)
        self.forward_grad_enabled = None

    def forward(self, input_ids, attention_mask, return_logits=False):
        self.forward_grad_enabled = torch.is_grad_enabled()
        return {"hidden": self.base(input_ids)}

    def exact_memory_loss(self, hidden, *args, **kwargs):
        return SimpleNamespace(loss=self.pointer(hidden).square().mean())


def test_detached_exact_objective_skips_only_the_unused_base_graph():
    torch.manual_seed(5)
    model = _ExactObjectiveModel()
    ids = torch.tensor([[1, 2, 3, 4]])
    batch = {
        "input_ids": ids,
        "labels": torch.tensor([[-100, 2, 3, 4]]),
        "attention_mask": torch.ones_like(ids),
        "prompt_mask": torch.tensor([[1, 0, 0, 0]]),
        "source_boundary": torch.tensor([2]),
        "copy_source_positions": torch.tensor([[-1, -1, 0, 1]]),
        "copy_target_mask": torch.tensor([[0, 0, 1, 1]]),
    }

    loss, _ = observable.exact_copy_objective(model, batch, 0.1, detach_hidden=True)
    loss.backward()

    assert model.forward_grad_enabled is False
    assert model.base.weight.grad is None
    assert model.pointer.weight.grad is not None
    assert torch.isfinite(model.pointer.weight.grad).all()


class _OrderedPool:
    def __init__(self, source_id: int, sequence_length: int = 512):
        self.source_id = source_id
        self.sequence_length = sequence_length
        self.cursor = 0

    def next_batch(self, _: int):
        value = self.source_id * 100 + self.cursor
        self.cursor += 1
        return {
            "input_ids": torch.full((1, self.sequence_length), value),
            "attention_mask": torch.ones((1, self.sequence_length), dtype=torch.long),
        }


def test_batch4_groups_short_rows_without_changing_source_order():
    pools = {"natural": _OrderedPool(1), "factual": _OrderedPool(2)}
    cycle = ["natural", "factual"]

    groups = observable.planned_base_forward_groups(
        pools, cycle, optimizer_step=0, effective_samples=8,
        short_batch_size=4, short_sequence_max_length=512,
    )

    assert [group["input_ids"].size(0) for group in groups] == [4, 4]
    assert [int(row[0]) for group in groups for row in group["input_ids"]] == [
        100, 200, 101, 201, 102, 202, 103, 203,
    ]
