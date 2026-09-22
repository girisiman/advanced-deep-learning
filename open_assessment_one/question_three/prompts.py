from __future__ import annotations

PREFIX_NEXT = "[next]"
PREFIX_FILL = "[fill]"
PREFIX_EN2NE = "[en2ne]"
PREFIX_NE2EN = "[ne2en]"
MASK_TOKEN = "[mask]"
SEP_TOKEN = "[sep]"

SPECIAL_PROMPT_TOKENS = (
    PREFIX_NEXT,
    PREFIX_FILL,
    PREFIX_EN2NE,
    PREFIX_NE2EN,
    MASK_TOKEN,
    SEP_TOKEN,
)

def format_next(tokens: list[str]) -> list[str]:
    """Put [next] in front of a word window.

    Args:
        tokens: e.g. ["the", "king", "shall", "come"].

    Returns:
        ["[next]", "the", "king", "shall", "come"]
    """
    if len(tokens) < 2:
        raise ValueError("next-word needs at least 2 tokens")
    return [PREFIX_NEXT] + list(tokens)

def format_fill(tokens: list[str], mask_index: int) -> list[str]:
    """Hide one word. Append the answer after [sep].

    Args:
        tokens: A word window.
        mask_index: Which word to hide (0-based).

    Returns:
        ["[fill]", "the", "king", "[mask]", "come", "[sep]", "shall"]
    """
    if not (0 <= mask_index < len(tokens)):
        raise ValueError("mask_index out of range")
    if len(tokens) < 3:
        raise ValueError("fill-in needs at least 3 tokens")

    answer = tokens[mask_index]
    masked = list(tokens)
    masked[mask_index] = MASK_TOKEN
    return [PREFIX_FILL] + masked + [SEP_TOKEN, answer]

def format_translate(
    source_tokens: list[str],
    target_tokens: list[str],
    direction: str,
) -> list[str]:
    """Prefix chooses the direction.

    Args:
        source_tokens: Source sentence tokens.
        target_tokens: Target sentence tokens.
        direction: "en2ne" or "ne2en".

    Returns:
        ["[en2ne]", "good", "morning", "[sep]", "शुभ", "प्रभात"]
    """
    if direction == "en2ne":
        prefix = PREFIX_EN2NE
    elif direction == "ne2en":
        prefix = PREFIX_NE2EN
    else:
        raise ValueError("direction must be 'en2ne' or 'ne2en'")
    if len(source_tokens) == 0 or len(target_tokens) == 0:
        raise ValueError("empty source or target")
    return [prefix] + list(source_tokens) + [SEP_TOKEN] + list(target_tokens)

def preview_prompts() -> None:
    print("next :", format_next(["the", "king", "shall", "come"]))
    print("fill :", format_fill(["the", "king", "shall", "come"], mask_index=2))
    print(
        "en2ne:",
        format_translate(["good", "morning"], ["शुभ", "प्रभात"], "en2ne"),
    )
    print(
        "ne2en:",
        format_translate(["शुभ", "प्रभात"], ["good", "morning"], "ne2en"),
    )


if __name__ == "__main__":
    preview_prompts()

