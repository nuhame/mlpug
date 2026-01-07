"""
Filter functions for dataset acquisition.

Filter functions are applied during download to exclude samples before saving.
Each function takes a sample dict and optional keyword arguments for configuration.
Returns True to keep the sample, False to discard.

Configuration is passed via filter_config in metadata, which becomes **kwargs.

See transform_analysis.md for filtering specifications.
"""


def filter_openmath_instruct_1(
    sample: dict,
    is_correct: bool = True,
    **kwargs,
) -> bool:
    """
    Filter openmath-instruct-1 samples.

    Only keep samples where is_correct matches the expected value.

    :param sample: Sample dict from dataset.
    :param is_correct: Expected value for is_correct field (default: True).

    :return: True to keep, False to discard.
    """
    return sample.get("is_correct", False) == is_correct


def filter_generics_kb(
    sample: dict,
    min_score: float = 0.5,
    **kwargs,
) -> bool:
    """
    Filter generics-kb samples by confidence score.

    :param sample: Sample dict from dataset.
    :param min_score: Minimum score threshold (default: 0.5).

    :return: True to keep, False to discard.
    """
    score = sample.get("score", 0.0)
    return score > min_score


def filter_fineweb_edu(
    sample: dict,
    min_int_score: int = 3,
    **kwargs,
) -> bool:
    """
    Filter fineweb-edu samples by quality score.

    :param sample: Sample dict from dataset.
    :param min_int_score: Minimum int_score threshold (default: 3).

    :return: True to keep, False to discard.
    """
    score = sample.get("int_score", 0)
    return score >= min_int_score


def filter_stackexchange(
    sample: dict,
    min_pm_score: int = 0,
    **kwargs,
) -> bool:
    """
    Filter stackexchange samples by answer score.

    Only keep samples where at least one answer has pm_score above threshold.

    :param sample: Sample dict from dataset.
    :param min_pm_score: Minimum pm_score threshold (default: 0).

    :return: True to keep, False to discard.
    """
    answers = sample.get("answers", [])
    if not answers:
        return False

    return any(ans.get("pm_score", 0) > min_pm_score for ans in answers)


# Language IDs for code-contests
CODE_CONTESTS_TARGET_LANGUAGES = {2, 3, 4}  # C++, Python3, Java


def filter_code_contests(
    sample: dict,
    require_languages: set = None,
    **kwargs,
) -> bool:
    """
    Filter code-contests samples by available solution languages.

    Only keep samples that have at least one solution in the target languages.

    :param sample: Sample dict from dataset.
    :param require_languages: Set of language IDs to require (default: {2, 3, 4} for C++/Python3/Java).
        Language IDs: 0=Unknown, 1=Python2, 2=C++, 3=Python3, 4=Java

    :return: True to keep, False to discard.
    """
    if require_languages is None:
        require_languages = CODE_CONTESTS_TARGET_LANGUAGES

    solutions = sample.get("solutions", {})
    available_languages = set(solutions.get("language", []))

    # Keep if any target language is available
    return bool(available_languages & require_languages)
