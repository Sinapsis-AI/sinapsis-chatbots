# -*- coding: utf-8 -*-
"""This module provides utilities for cleaning and post-processing text."""

import re


def postprocess_text(input_text: str, pattern: str | None = None, keep_before: bool = True) -> str:
    """
    Cleans text by removing or splitting based on structured delimiters using a regex pattern.

    The default pattern detects:
    - `<| ... |>`: Matches anything inside `<|` and `|>`, often used in AI outputs.
    - `</ ... >`: Matches anything inside `</` and `>`, common in XML-like markup.

    Parameters:
    - response (str): The input string to be cleaned.
    - pattern (str, optional): A regex pattern to match delimiters. Defaults to handling `<|...|>` and `</...>`.
    - keep_before (bool, optional): If True, returns the portion before the first match;
      if False, returns the portion after the first match.

    Returns:
    - str: The cleaned response string.
    """
    if pattern is None:
        pattern = r"(<\|.*?\|>|</.*?>)"

    match = re.search(pattern, input_text)

    if match:
        split_index = match.start() if keep_before else match.end()
        return input_text[:split_index].strip() if keep_before else input_text[split_index:].strip()

    return input_text
