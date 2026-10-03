from sharada import Policy, escalation_budget


def test_the_cheaper_mistake_wins():
    # letting fraud through costs a hundred times more than a false alarm, so at 10% fraud the model
    # should still reject
    policy = Policy(options=["approve", "reject"],
                    costs={("fraud", "approve"): 10_000, ("clean", "reject"): 100},
                    wrong=0.0)
    action = policy.act({"fraud": 0.1, "clean": 0.9})
    assert action.kind == "answer" and action.answer == "reject"
    assert action.expected_loss == 90.0                      # 0.9 × 100


def test_escalation_is_taken_when_it_is_cheaper():
    policy = Policy(options=["approve", "reject"],
                    costs={("fraud", "approve"): 10_000, ("clean", "reject"): 100},
                    wrong=0.0, escalate=30)
    assert policy.act({"fraud": 0.5, "clean": 0.5}).kind == "escalate"
    assert policy.act({"fraud": 0.001, "clean": 0.999}).kind == "answer"


def test_budget_limits_how_much_is_handed_over():
    policy = Policy(options=["a", "b"], wrong=100, escalate=1)
    hard = [{"a": 0.5 + 0.004 * i, "b": 0.5 - 0.004 * i} for i in range(100)]
    assert 1 - policy.coverage(hard) > 0.5                   # as given, it escalates most of them
    limited = escalation_budget(policy, hard, budget=0.1)
    assert 1 - limited.coverage(hard) <= 0.1 + 1e-9
    assert limited.escalate > policy.escalate


def test_realised_loss_counts_what_actually_happened():
    policy = Policy(options=["a", "b"], wrong=10, escalate=1)
    probabilities = [{"a": 0.99, "b": 0.01}, {"a": 0.5, "b": 0.5}]
    out = policy.realised_loss(probabilities, truths=["b", "a"])
    assert out["coverage"] == 0.5                            # the uncertain one went to a person
    assert out["loss_per_request"] == (10 + 1) / 2           # the confident one was wrong
    assert out["accuracy_when_answering"] == 0.0
