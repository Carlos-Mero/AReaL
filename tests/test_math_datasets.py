from areal.dataset.math import (
    _boxed_answer,
    _extract_ground_truth,
    _normalize_prompt_messages,
    _prompt_to_text,
)


def test_normalize_prompt_messages_keeps_chat_prompt():
    prompt = [{"role": "user", "content": "Solve 1+1."}]

    messages = _normalize_prompt_messages(prompt)

    assert messages == [{"role": "user", "content": "Solve 1+1."}]


def test_prompt_to_text_joins_chat_contents():
    prompt = [
        {"role": "system", "content": "You are a math assistant."},
        {"role": "user", "content": "Solve 1+1."},
    ]

    assert _prompt_to_text(prompt) == "You are a math assistant.\nSolve 1+1."


def test_extract_ground_truth_reads_reward_model_dict():
    sample = {"reward_model": {"ground_truth": r"\frac{1}{2}"}}

    assert _extract_ground_truth(sample) == r"\frac{1}{2}"


def test_boxed_answer_wraps_unboxed_answer():
    assert _boxed_answer("42") == r"\boxed{42}"


def test_boxed_answer_preserves_boxed_answer():
    assert _boxed_answer(r"\boxed{42}") == r"\boxed{42}"
