"""Sharada — typed decisions from text in one forward pass.

    from sharada import DecisionModel

    model = DecisionModel.from_pretrained("LenaBarretta/sharada-base")
    d = model.decide("My card hasn't arrived yet",
                     "Which team should handle this?",
                     ["billing", "technical", "sales"])
    d.answer, d.confidence

The model reads the text, the question and every option in one pass and returns a probability for each
option. Every option is a branch of the sequence starting at the same position, so their order cannot
change the answer; an option reads the text, the question and itself, so its score does not depend on
which other options are offered.
"""

from .calibrate import calibrate, check_passport, fit_temperature, save_passport
from .data import Example, Request
from .evaluate import calibration_error, evaluate, latency, risk_coverage
from .layout import Limits
from .model import Config, Decision, DecisionModel
from .policy import Action, Policy, escalation_budget
from .train import fit

__all__ = [
    "DecisionModel", "Config", "Decision", "Request", "Example", "Limits",
    "fit", "calibrate", "fit_temperature", "save_passport", "check_passport",
    "evaluate", "calibration_error", "risk_coverage", "latency",
    "Policy", "Action", "escalation_budget",
]
__version__ = "0.1.0.dev1"
