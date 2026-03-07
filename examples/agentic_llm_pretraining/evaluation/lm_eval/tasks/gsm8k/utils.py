"""
Pass@k metric for GSM8K (and other answer-matching benchmarks).

This module provides a pass@k implementation for benchmarks where correctness
is determined by matching extracted answers against ground truth, rather than
by code execution (as in HumanEval).

The pass@k metric measures: "given k randomly-selected samples, what is the
probability that at least one is correct?" This is the right metric for
assessing RLVR readiness, where GRPO needs any correct answer in a group
to produce useful gradient signal.

Usage with lm-eval:
    Place this file in lm-eval's tasks/gsm8k/ directory alongside the
    gsm8k_cot_pass_at_k.yaml config. The YAML references this module via
    `metric: !function utils.pass_at_k`.

Interface:
    lm-eval calls pass_at_k(references, predictions, k) once per problem:
    - references: [gold_answer_str]  (list with 1 element)
    - predictions: [[extracted_0, extracted_1, ...]]  (list with 1 list of N extractions)
    - k: [1, 8, 32, 64, 128, 256]  (from YAML config)

    Returns dict like {"pass@1": 0.5, "pass@8": 0.833, ...}
"""

import re
from math import comb


def _normalize_answer(answer: str) -> str:
    """Normalize a numeric answer string for comparison.

    Strips formatting characters (commas, dollar signs, percent signs,
    trailing periods) that don't affect the numeric value.

    :param answer: Raw answer string from regex extraction or ground truth.

    :return: Normalized string, or empty string if invalid.
    """
    if not answer or answer == "[invalid]":
        return ""

    answer = answer.strip()
    answer = answer.replace(",", "")
    answer = answer.replace("$", "")
    answer = answer.replace("%", "")

    if answer.endswith("."):
        answer = answer[:-1]

    return answer


def _answers_match(predicted: str, gold: str) -> bool:
    """Check if a predicted answer matches the gold answer.

    Compares normalized strings first, then attempts numeric comparison
    to handle cases like "6.0" vs "6".

    :param predicted: Extracted answer from model output.
    :param gold: Ground truth answer.

    :return: True if answers match.
    """
    pred = _normalize_answer(predicted)
    gold_norm = _normalize_answer(gold)

    if not pred or not gold_norm:
        return False

    if pred == gold_norm:
        return True

    try:
        return float(pred) == float(gold_norm)
    except ValueError:
        return False


def _pass_at_k_estimator(n: int, c: int, k: int) -> float:
    """Compute the unbiased pass@k estimator.

    Formula: pass@k = 1 - C(n-c, k) / C(n, k)

    This is the unbiased estimator from Chen et al. (2021) "Evaluating Large
    Language Models Trained on Code" (Codex paper). It computes the probability
    that at least 1 of k randomly-selected samples (from n total) is correct,
    given that c of the n samples are correct.

    :param n: Total number of samples generated.
    :param c: Number of correct samples.
    :param k: Number of samples to select.

    :return: Probability that at least one selected sample is correct.
    """
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_at_k(
    references: list[str],
    predictions: list[list[str]],
    k: list[int] = None,
) -> dict:
    """Compute pass@k for answer-matching benchmarks.

    Called by lm-eval once per problem. Counts how many of the N predictions
    match the gold answer, then computes the unbiased pass@k estimator for
    each requested k value.

    :param references: List with one element: the gold answer string.
    :param predictions: List with one element: a list of N extracted
        answer strings (one per repeat).
    :param k: List of k values to compute pass@k for.

    :return: Dict mapping "pass@{ki}" to float score for each ki in k.
    """
    assert k is not None, "k values must be provided in YAML config"

    if isinstance(k, int):
        k = [k]

    gold = references[0]
    preds = predictions[0]

    n = len(preds)
    c = sum(1 for p in preds if _answers_match(p, gold))

    result = {}
    for ki in k:
        if ki > n:
            continue
        result[f"pass@{ki}"] = _pass_at_k_estimator(n, c, ki)

    return result
