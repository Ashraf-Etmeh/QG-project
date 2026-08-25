import re

from sentence_transformers import util

from nlp_models import semantic_model


def fix_residual_phrases(
    question_text,
    answer_term=""
):
    """
    Clean question-generation artifacts.
    """

    cleaned = re.sub(
        r"\s+",
        " ",
        question_text
    ).strip()

    patterns_to_remove = [
        r"\band the will\?",
        r"\byou must substitute any\.\.\.",
        r"\bwill n ot be paid\b",
        r"\bevery attempt will be made to have a reviewed within\b",
        r"\bfor any hours not worked\b",
        r"\bin the event of any conflict between this policy\b",
    ]

    for pattern in patterns_to_remove:

        cleaned = re.sub(
            pattern,
            "",
            cleaned,
            flags=re.IGNORECASE
        )

    cleaned = re.sub(
        r"\s+",
        " ",
        cleaned
    ).strip(" ,.-")

    if answer_term:

        endings = (
            "regarding",
            "concerning",
            "related to",
            "about",
            "for",
            "on",
        )

        if cleaned.lower().endswith(endings):

            cleaned = (
                f"{cleaned} "
                f"{answer_term.strip()}"
            )

    cleaned = re.sub(
        r"\?+",
        "?",
        cleaned
    )

    if not cleaned.endswith("?"):
        cleaned += "?"

    return cleaned


def clean_options(
    options,
    correct_answer,
    max_options=4
):
    """
    Clean MCQ options.
    """

    correct_answer = (
        correct_answer.strip()
    )

    final_options = []
    seen_options = set()

    fallback_distractors = [
        "General company policy",
        "Standard administrative procedure",
        "Internal company guideline",
        "Management procedure",
    ]

    for option in options:

        if option is None:
            continue

        option = str(option).strip()

        if not option:
            continue

        option_key = option.casefold()
        correct_key = correct_answer.casefold()

        if option_key in seen_options:
            continue

        if (
            option_key == correct_key
            and any(
                existing.casefold()
                == correct_key
                for existing in final_options
            )
        ):
            continue

        final_options.append(option)
        seen_options.add(option_key)

    # Remove duplicate correct answer
    unique_options = []
    correct_added = False

    for option in final_options:

        if (
            option.casefold()
            == correct_answer.casefold()
        ):

            if correct_added:
                continue

            correct_added = True

        unique_options.append(option)

    final_options = unique_options

    # Emergency fallback
    for fallback in fallback_distractors:

        if len(final_options) >= max_options:
            break

        existing = {
            x.casefold()
            for x in final_options
        }

        if (
            fallback.casefold()
            != correct_answer.casefold()
            and fallback.casefold()
            not in existing
        ):

            final_options.append(
                fallback
            )

    return final_options[:max_options]


def evaluate_free_response(
    employee_answer,
    reference_answer,
    threshold=0.60
):
    """
    Evaluate a free-response answer
    using semantic similarity.
    """

    if (
        not employee_answer
        or not employee_answer.strip()
    ):

        return {
            "score": 0.0,
            "status": "Incorrect"
        }

    if (
        not reference_answer
        or not reference_answer.strip()
    ):

        return {
            "score": 0.0,
            "status": "Cannot evaluate"
        }

    embeddings = semantic_model.encode(
        [
            employee_answer,
            reference_answer
        ],
        convert_to_tensor=True
    )

    similarity = util.cos_sim(
        embeddings[0],
        embeddings[1]
    ).item()

    status = (
        "Correct"
        if similarity >= threshold
        else "Incorrect"
    )

    return {
        "score": similarity,
        "status": status
    }