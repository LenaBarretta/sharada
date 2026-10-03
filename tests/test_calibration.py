import datetime as dt
import math

from sharada import calibration_error, fit_temperature
from sharada.calibrate import check_passport
from sharada.evaluate import risk_coverage

# Honest probabilities, with the answers arriving in the proportions the probabilities claim: of ten
# requests the model calls 0.7/0.2/0.1, seven really are the first option, two the second, one the third.
# Anything less than that is not calibrated, and then sharpening it can genuinely lower the log loss.
HONEST = [([0.7, 0.2, 0.1], [7, 2, 1]),
          ([0.4, 0.4, 0.2], [4, 4, 2]),
          ([0.9, 0.05, 0.05], [18, 1, 1]),
          ([0.4, 0.3, 0.3], [4, 3, 3])]


def calibrated_sample(repeats=8):
    rows, labels = [], []
    for _ in range(repeats):
        for row, counts in HONEST:
            for label, n in enumerate(counts):
                rows += [row] * n
                labels += [label] * n
    return rows, labels


def sharpened(probabilities, temperature):
    out = []
    for row in probabilities:
        scaled = [math.log(p) / temperature for p in row]
        top = max(scaled)
        exp = [math.exp(v - top) for v in scaled]
        out.append([v / sum(exp) for v in exp])
    return out


def test_temperature_undoes_a_known_sharpening():
    honest, labels = calibrated_sample()
    assert abs(fit_temperature(honest, labels) - 1.0) < 0.1          # nothing to fix

    twice_as_sharp = sharpened(honest, 0.5)
    assert abs(fit_temperature(twice_as_sharp, labels) - 2.0) < 0.2

    too_timid = sharpened(honest, 2.0)                               # half as sharp as it should be
    assert abs(fit_temperature(too_timid, labels) - 0.5) < 0.1


def test_calibration_error_is_zero_when_the_numbers_are_true():
    confidence = [0.9] * 100
    correct = [1] * 90 + [0] * 10
    assert calibration_error(confidence, correct) < 1e-9

    overconfident = [1.0] * 100
    assert abs(calibration_error(overconfident, correct) - 0.1) < 1e-9


def test_risk_coverage_is_ordered_by_confidence():
    confidence = [0.99, 0.9, 0.6, 0.51]
    correct = [1, 1, 0, 0]
    curve = risk_coverage(confidence, correct, steps=4)
    assert [round(c["coverage"], 2) for c in curve] == [0.25, 0.5, 0.75, 1.0]
    assert curve[0]["accuracy"] == 1.0 and curve[-1]["accuracy"] == 0.5


def test_passport_expires_and_says_so():
    passport = {"valid_until": "2020-01-01", "examples": 1000, "fingerprint": {}}
    problems = check_passport(passport, today=dt.date(2026, 1, 1))
    assert any("expired" in p for p in problems)

    fresh = {"valid_until": "2099-01-01", "examples": 1000, "fingerprint": {}}
    assert check_passport(fresh, today=dt.date(2026, 1, 1)) == []

    thin = {"valid_until": "2099-01-01", "examples": 50, "fingerprint": {}}
    assert any("wide" in p for p in check_passport(thin, today=dt.date(2026, 1, 1)))
