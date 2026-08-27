from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol

import torch

from .actions import Action, ActionName
from .anchor_state import AnchorState
from .budget import BudgetConfig, BudgetExceeded, BudgetTracker
from .environment import ActionResult, Environment
from .loop_detection import LoopDetector
from .parser import parse_action
from .text_environment import TextEnvironment
from .trace import RLMTrace, TraceEvent
from .uncertainty_policy import UncertaintyFeatures, UncertaintyPolicy


@dataclass
class ModelStep:
    text: str
    generated_tokens: int = 0
    signals: Dict[str, float] = field(default_factory=dict)
    anchor_summary: Optional[torch.Tensor] = None
    anchor_usage: Optional[torch.Tensor] = None


class ModelBackend(Protocol):
    def generate_action(
        self,
        prompt: str,
        anchor_state: Optional[AnchorState],
    ) -> ModelStep: ...


@dataclass
class RLMResult:
    answer: str
    status: str
    trace: RLMTrace
    steps: int
    depth: int
    budget_remaining: Dict[str, int]


def build_action_prompt(
    question: str,
    history: list[dict[str, str]],
    budget: BudgetTracker,
    depth: int,
    environment: Environment,
    controller_hint: str = "",
) -> str:
    allowed = ", ".join(action.value for action in environment.allowed_actions)
    return (
        f"Return exactly one JSON action. Allowed: {allowed}.\n"
        f"Question: {question}\nDepth: {depth}\nRemaining: {json.dumps(budget.remaining())}\n"
        f"History: {json.dumps(history[-4:], ensure_ascii=False)}\n{controller_hint}"
    )


class RecursiveController:
    def __init__(
        self,
        backend: ModelBackend,
        budget_config: BudgetConfig | None = None,
        uncertainty_policy: UncertaintyPolicy | None = None,
        anchor_state: AnchorState | None = None,
    ):
        self.backend = backend
        self.config = budget_config or BudgetConfig()
        self.uncertainty_policy = uncertainty_policy
        self.anchor_state = anchor_state

    def _prompt(
        self,
        question: str,
        history: list[dict[str, str]],
        budget: BudgetTracker,
        depth: int,
        environment: Environment,
        last_signals: Dict[str, float],
    ) -> str:
        controller_hint = ""
        if self.uncertainty_policy and last_signals:
            features = UncertaintyFeatures(
                anchor_entropy=last_signals.get("anchor_entropy", 0.0),
                normalized_anchor_entropy=last_signals.get("normalized_anchor_entropy", 0.0),
                effective_anchor_count=last_signals.get("effective_anchor_count", 0.0),
                top1_membership=last_signals.get("top1_membership", 0.0),
                top1_top2_margin=last_signals.get("top1_top2_margin", 0.0),
                token_entropy=last_signals.get("token_entropy", 0.0),
                mean_logprob=last_signals.get("mean_logprob", 0.0),
                state_conflict=last_signals.get("state_conflict", 0.0),
                confidence=last_signals.get("confidence", 0.0),
                remaining_budget_ratio=budget.remaining()["steps"] / max(self.config.max_steps, 1),
                depth=float(depth),
            )
            probability = self.uncertainty_policy.need_more_context_probability(features)
            hint = self.uncertainty_policy.action_hint(features)
            controller_hint = (
                "Advisory uncertainty controller: "
                f"need_more_context={probability:.3f}, suggested_action={hint}. "
                "Verify against evidence."
            )
        return build_action_prompt(
            question,
            history,
            budget,
            depth,
            environment,
            controller_hint=controller_hint,
        )

    def run(
        self,
        question: str,
        environment: Environment,
        depth: int = 0,
        budget: BudgetTracker | None = None,
    ) -> RLMResult:
        if depth > self.config.max_depth:
            raise BudgetExceeded("maximum recursion depth exceeded")
        budget = budget or BudgetTracker(self.config)
        trace = RLMTrace()
        loop = LoopDetector(self.config.max_repeated_action)
        history: list[dict[str, str]] = []
        last_signals: Dict[str, float] = {}
        last_observation = ""
        while True:
            prompt = self._prompt(question, history, budget, depth, environment, last_signals)
            step = self.backend.generate_action(prompt, self.anchor_state)
            try:
                budget.consume_step(step.generated_tokens)
            except BudgetExceeded as exc:
                return RLMResult(last_observation, f"budget_exceeded:{exc}", trace, budget.steps, depth, budget.remaining())

            if self.anchor_state is not None and step.anchor_summary is not None and step.anchor_usage is not None:
                metrics = self.anchor_state.update(step.anchor_summary, step.anchor_usage)
                step.signals["state_conflict"] = metrics.state_conflict
                step.signals["confidence"] = float(self.anchor_state.confidence.mean().item())
            last_signals = dict(step.signals)
            parsed = parse_action(step.text)
            if parsed.action is None:
                try:
                    budget.consume_repair()
                except BudgetExceeded as exc:
                    return RLMResult(last_observation, f"parse_failed:{exc}", trace, budget.steps, depth, budget.remaining())
                observation = f"Parse error: {parsed.error}. Return one valid JSON object."
                history.append({"output": step.text, "observation": observation})
                trace.add(TraceEvent(budget.steps, depth, step.text, None, observation, False, step.signals, budget.remaining()))
                continue

            action = parsed.action
            if loop.observe(action):
                return RLMResult(last_observation, "loop_detected", trace, budget.steps, depth, budget.remaining())
            if action.name == ActionName.STOP:
                answer = action.arguments["answer"]
                trace.add(TraceEvent(budget.steps, depth, step.text, action.to_dict(), answer, True, step.signals, budget.remaining()))
                return RLMResult(answer, "stopped", trace, budget.steps, depth, budget.remaining())

            if action.name == ActionName.CALL:
                try:
                    budget.consume_subcall(depth + 1)
                    sub_context = environment.resolve_ref(action.arguments["context_ref"])
                except (BudgetExceeded, KeyError) as exc:
                    result = ActionResult(False, str(exc))
                else:
                    sub_environment = TextEnvironment(sub_context)
                    sub_result = self.run(
                        action.arguments["question"],
                        sub_environment,
                        depth=depth + 1,
                        budget=budget,
                    )
                    result = ActionResult(True, sub_result.answer, tokens_read=0)
            else:
                result = environment.execute(action)
            try:
                budget.consume_read(result.tokens_read)
            except BudgetExceeded as exc:
                return RLMResult(last_observation, f"budget_exceeded:{exc}", trace, budget.steps, depth, budget.remaining())
            last_observation = result.observation
            history.append({"action": json.dumps(action.to_dict()), "observation": result.observation})
            trace.add(
                TraceEvent(
                    budget.steps,
                    depth,
                    step.text,
                    action.to_dict(),
                    result.observation,
                    result.ok,
                    step.signals,
                    budget.remaining(),
                )
            )
