"""What to do with 0.94.

A probability is not a decision. The decision depends on what each mistake costs and on what it costs
to hand the case to a person or to a bigger model. Give those costs and the rule is arithmetic: take the
action with the smallest expected loss, and abstain when abstaining costs less than any of them.

    policy = Policy(options=["approve", "reject"],
                    costs={("fraud", "approve"): 10_000, ("clean", "reject"): 100},
                    escalate=30)
    policy.act(decision.probabilities)
"""

from __future__ import annotations

from dataclasses import dataclass, field

ESCALATE = "escalate"


@dataclass
class Action:
    kind: str                      # "answer" or "escalate"
    answer: str | None
    expected_loss: float
    margin: float                  # how much better than the next best course of action

    def __repr__(self) -> str:
        what = self.answer if self.kind == "answer" else "escalate"
        return f"Action({what!r}, expected_loss={self.expected_loss:.2f})"


@dataclass
class Policy:
    """Costs are given per (true option, chosen option) pair; anything not named costs `wrong` when the
    answer is wrong and nothing when it is right."""

    options: list[str]
    costs: dict[tuple[str, str], float] = field(default_factory=dict)
    wrong: float = 1.0
    escalate: float | None = None

    def loss(self, true_option: str, chosen: str) -> float:
        if true_option == chosen:
            return float(self.costs.get((true_option, chosen), 0.0))
        return float(self.costs.get((true_option, chosen), self.wrong))

    def expected_losses(self, probabilities: dict[str, float]) -> dict[str, float]:
        """What each course of action is expected to cost, under the model's own probabilities."""
        out = {chosen: sum(p * self.loss(true_option, chosen) for true_option, p in probabilities.items())
               for chosen in self.options}
        if self.escalate is not None:
            out[ESCALATE] = float(self.escalate)
        return out

    def act(self, probabilities: dict[str, float]) -> Action:
        losses = self.expected_losses(probabilities)
        ranked = sorted(losses.items(), key=lambda kv: kv[1])
        (best, loss), (_, second) = ranked[0], ranked[1]
        if best == ESCALATE:
            return Action(kind=ESCALATE, answer=None, expected_loss=loss, margin=second - loss)
        return Action(kind="answer", answer=best, expected_loss=loss, margin=second - loss)

    # ── choosing how much to take on ──────────────────────────────────────────────────────────
    def coverage(self, probabilities: list[dict[str, float]]) -> float:
        """The share of these requests the model would answer itself."""
        answered = sum(1 for p in probabilities if self.act(p).kind == "answer")
        return answered / len(probabilities)

    def realised_loss(self, probabilities: list[dict[str, float]], truths: list[str]) -> dict:
        """What this policy would actually have cost on labelled data — the number to compare policies
        on, and the one that goes wrong when the probabilities are not honest."""
        total, answered, mistakes = 0.0, 0, 0
        for p, truth in zip(probabilities, truths):
            action = self.act(p)
            if action.kind == ESCALATE:
                total += float(self.escalate or 0.0)
                continue
            answered += 1
            total += self.loss(truth, action.answer)
            mistakes += action.answer != truth
        n = len(truths)
        return {"loss_per_request": total / n, "coverage": answered / n,
                "accuracy_when_answering": (answered - mistakes) / answered if answered else float("nan")}


def escalation_budget(policy: Policy, probabilities: list[dict[str, float]], budget: float) -> Policy:
    """Escalating everything the model is unsure about only works while there is someone to escalate to.
    Given a budget — the share of requests that may be handed over — this returns the policy that keeps
    to it: the cost of escalating is raised until few enough cases are worth escalating."""
    if policy.escalate is None:
        raise ValueError("this policy never escalates; give it an `escalate` cost first")
    low, high = 0.0, max(policy.wrong, policy.escalate) * 100
    for _ in range(40):
        middle = (low + high) / 2
        trial = Policy(policy.options, policy.costs, policy.wrong, middle)
        if 1 - trial.coverage(probabilities) > budget:
            low = middle            # too many escalations: make escalating dearer
        else:
            high = middle
    return Policy(policy.options, policy.costs, policy.wrong, high)
