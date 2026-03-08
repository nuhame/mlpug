"""Tests for pass@k metric utilities.

Run from the gsm8k task directory:
    cd examples/agentic_llm_pretraining/evaluation/lm_eval/tasks/gsm8k
    python -m pytest test_utils.py -v
"""

import pytest

from utils import (
    _normalize_answer,
    _answers_match,
    _pass_at_k_estimator,
    pass_at_k,
)


# ── _normalize_answer ──────────────────────────────────────────────────

class TestNormalizeAnswer:

    def test_plain_number(self):
        assert _normalize_answer("42") == "42"

    def test_strip_commas(self):
        assert _normalize_answer("6,000") == "6000"

    def test_strip_dollar(self):
        assert _normalize_answer("$100") == "100"

    def test_strip_percent(self):
        assert _normalize_answer("85%") == "85"

    def test_strip_trailing_period(self):
        assert _normalize_answer("42.") == "42"

    def test_decimal_not_stripped(self):
        assert _normalize_answer("3.14") == "3.14"

    def test_negative(self):
        assert _normalize_answer("-3") == "-3"

    def test_whitespace(self):
        assert _normalize_answer("  42  ") == "42"

    def test_invalid_marker(self):
        assert _normalize_answer("[invalid]") == ""

    def test_empty(self):
        assert _normalize_answer("") == ""

    def test_none(self):
        assert _normalize_answer(None) == ""

    def test_combined_formatting(self):
        assert _normalize_answer("$6,000.") == "6000"


# ── _answers_match ─────────────────────────────────────────────────────

class TestAnswersMatch:

    def test_exact(self):
        assert _answers_match("42", "42") is True

    def test_float_comparison(self):
        assert _answers_match("6.0", "6") is True

    def test_comma_normalization(self):
        assert _answers_match("6,000", "6000") is True

    def test_invalid_vs_number(self):
        assert _answers_match("[invalid]", "42") is False

    def test_empty_vs_number(self):
        assert _answers_match("", "42") is False

    def test_negative(self):
        assert _answers_match("-3", "-3") is True

    def test_wrong_answer(self):
        assert _answers_match("39", "42") is False

    def test_dollar_in_predicted(self):
        assert _answers_match("$100", "100") is True

    def test_trailing_period(self):
        assert _answers_match("42.", "42") is True


# ── _pass_at_k_estimator ──────────────────────────────────────────────

class TestPassAtKEstimator:

    def test_half_correct_k1(self):
        # n=4, c=2, k=1: 1 - C(2,1)/C(4,1) = 1 - 2/4 = 0.5
        assert _pass_at_k_estimator(4, 2, 1) == pytest.approx(0.5)

    def test_half_correct_k2(self):
        # n=4, c=2, k=2: 1 - C(2,2)/C(4,2) = 1 - 1/6 = 5/6
        assert _pass_at_k_estimator(4, 2, 2) == pytest.approx(5 / 6)

    def test_half_correct_k4(self):
        # n=4, c=2, k=4: n-c=2 < k=4, so 1.0
        assert _pass_at_k_estimator(4, 2, 4) == 1.0

    def test_none_correct(self):
        assert _pass_at_k_estimator(4, 0, 1) == 0.0
        assert _pass_at_k_estimator(4, 0, 4) == 0.0

    def test_all_correct(self):
        # n-c=0 < k for any k>=1
        assert _pass_at_k_estimator(4, 4, 1) == 1.0
        assert _pass_at_k_estimator(4, 4, 4) == 1.0

    def test_one_correct_k1(self):
        # n=4, c=1, k=1: 1 - C(3,1)/C(4,1) = 1 - 3/4 = 0.25
        assert _pass_at_k_estimator(4, 1, 1) == pytest.approx(0.25)

    def test_large_n(self):
        # n=256, c=10, k=64: verify formula produces valid probability
        result = _pass_at_k_estimator(256, 10, 64)
        assert 0.0 < result < 1.0

    def test_monotonic_in_k(self):
        # pass@k should increase with k for fixed n, c
        results = [_pass_at_k_estimator(256, 10, k) for k in [1, 8, 32, 64, 128, 256]]
        for i in range(len(results) - 1):
            assert results[i] <= results[i + 1]


# ── pass_at_k (integration) ───────────────────────────────────────────

class TestPassAtK:

    def test_basic_call(self):
        """Single problem, 4 predictions, 2 correct."""
        result = pass_at_k(
            references=["42"],
            predictions=[["42", "39", "42", "[invalid]"]],
            k=[1, 2, 4],
        )
        assert result["pass@1"] == pytest.approx(0.5)
        assert result["pass@2"] == pytest.approx(5 / 6)
        assert result["pass@4"] == 1.0

    def test_no_correct(self):
        """No predictions match gold."""
        result = pass_at_k(
            references=["7"],
            predictions=[["8", "9", "[invalid]", "10"]],
            k=[1, 2, 4],
        )
        assert result["pass@1"] == 0.0
        assert result["pass@2"] == 0.0
        assert result["pass@4"] == 0.0

    def test_all_correct(self):
        """All predictions match gold."""
        result = pass_at_k(
            references=["42"],
            predictions=[["42", "42", "42", "42"]],
            k=[1, 4],
        )
        assert result["pass@1"] == 1.0
        assert result["pass@4"] == 1.0

    def test_k_larger_than_n_skipped(self):
        """k values larger than n should be omitted from result."""
        result = pass_at_k(
            references=["42"],
            predictions=[["42", "39"]],
            k=[1, 2, 4, 8],
        )
        assert "pass@1" in result
        assert "pass@2" in result
        assert "pass@4" not in result
        assert "pass@8" not in result

    def test_single_k_as_int(self):
        """k can be passed as a single int."""
        result = pass_at_k(
            references=["42"],
            predictions=[["42", "39", "42", "[invalid]"]],
            k=1,
        )
        assert result["pass@1"] == pytest.approx(0.5)

    def test_normalization_in_matching(self):
        """Verify normalization is applied during matching."""
        result = pass_at_k(
            references=["6000"],
            predictions=[["6,000", "$6000", "6000.", "wrong"]],
            k=[1],
        )
        # 3 out of 4 match after normalization
        # pass@1 = 1 - C(1,1)/C(4,1) = 1 - 1/4 = 0.75
        assert result["pass@1"] == pytest.approx(0.75)

    def test_float_matching(self):
        """Float comparison: '6.0' matches '6'."""
        result = pass_at_k(
            references=["6"],
            predictions=[["6.0", "wrong"]],
            k=[1],
        )
        # c=1, n=2: pass@1 = 1 - C(1,1)/C(2,1) = 1 - 1/2 = 0.5
        assert result["pass@1"] == pytest.approx(0.5)

    def test_missing_k_raises(self):
        """k=None should raise an assertion error."""
        with pytest.raises(AssertionError):
            pass_at_k(
                references=["42"],
                predictions=[["42"]],
                k=None,
            )

    def test_simulated_lm_eval_pipeline(self):
        """Simulate how lm-eval calls pass_at_k across multiple problems.

        lm-eval calls pass_at_k once per problem, then aggregates with mean.
        This test simulates 2 problems and verifies the per-problem results
        match expected values, and the aggregate mean is correct.
        """
        k_values = [1, 2, 4]

        # Problem 1: gold="42", 2/4 correct
        result_1 = pass_at_k(
            references=["42"],
            predictions=[["42", "39", "42", "[invalid]"]],
            k=k_values,
        )

        # Problem 2: gold="7", 0/4 correct
        result_2 = pass_at_k(
            references=["7"],
            predictions=[["8", "9", "[invalid]", "10"]],
            k=k_values,
        )

        # Verify per-problem results
        assert result_1["pass@1"] == pytest.approx(0.5)
        assert result_1["pass@2"] == pytest.approx(5 / 6)
        assert result_1["pass@4"] == 1.0

        assert result_2["pass@1"] == 0.0
        assert result_2["pass@2"] == 0.0
        assert result_2["pass@4"] == 0.0

        # Simulate lm-eval's mean aggregation across problems
        assert (result_1["pass@1"] + result_2["pass@1"]) / 2 == pytest.approx(0.25)
        assert (result_1["pass@2"] + result_2["pass@2"]) / 2 == pytest.approx(5 / 12)
        assert (result_1["pass@4"] + result_2["pass@4"]) / 2 == pytest.approx(0.5)
