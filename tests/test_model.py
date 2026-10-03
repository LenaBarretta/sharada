"""The properties the design is built on, checked on an untrained model.

They hold by construction, not by training, so an untrained model is the honest place to check them.
These tests need the encoder's weights, so they download them once.
"""

import pytest
import torch

from sharada import DecisionModel, Request
from sharada.model import encode_one

TEXT = "The central bank raised interest rates again, and markets fell sharply."
QUESTION = "Which section does this news item belong to?"
OPTIONS = ["world news", "sports", "business", "science and technology"]


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return DecisionModel().eval()


def scores(model, options):
    _, batch = encode_one(model, Request(text=TEXT, question=QUESTION, options=options))
    with torch.no_grad():
        return model(batch)[0]


def test_reordering_the_options_reorders_nothing_else(model):
    order = [2, 0, 3, 1]
    straight = scores(model, OPTIONS)
    shuffled = scores(model, [OPTIONS[i] for i in order])
    for i in range(len(OPTIONS)):
        assert abs(float(straight[i]) - float(shuffled[order.index(i)])) < 1e-5


def test_an_options_score_ignores_the_other_options(model):
    straight = scores(model, OPTIONS)
    fewer = scores(model, OPTIONS[:2])
    assert torch.allclose(straight[:2], fewer[:2], atol=1e-6)


def test_the_text_does_not_depend_on_the_options(model):
    from sharada.masking import encoder_masks

    def states(options):
        _, batch = encode_one(model, Request(text=TEXT, question=QUESTION, options=options))
        masks = encoder_masks(batch.segments, batch.positions, model.config.local_reach,
                              model.attn, torch.float32)
        with torch.no_grad():
            return model.encoder(input_ids=batch.ids, position_ids=batch.positions,
                                 attention_mask=masks).last_hidden_state[0]

    n_text = len(model.layout._ids(TEXT)) + 2
    assert torch.allclose(states(OPTIONS)[:n_text], states(["cars", "cooking"])[:n_text], atol=1e-6)


def test_a_question_answers_the_same_alone_as_in_a_batch(model):
    other = Request(text="I loved every minute of it.", question="How positive is this review?",
                    options=["negative", "neutral", "positive"], kind="scale")
    alone = model.probabilities([other])[0]
    together = model.probabilities([Request(TEXT, QUESTION, OPTIONS), other])[1]
    assert max(abs(a - b) for a, b in zip(alone, together)) < 1e-5


def test_decide_returns_a_probability_per_option(model):
    decision = model.decide(TEXT, QUESTION, OPTIONS)
    assert decision.answer in OPTIONS
    assert abs(sum(decision.probabilities.values()) - 1.0) < 1e-5
    assert decision.confidence == max(decision.probabilities.values())


def test_two_options_are_the_minimum(model):
    with pytest.raises(ValueError):
        model.decide(TEXT, QUESTION, ["business"])
