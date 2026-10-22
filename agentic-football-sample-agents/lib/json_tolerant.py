"""Recover JSON from model output that used Python literals, trailing commas or code fences."""

import json
import re

_PY_LITERALS = (("True", "true"), ("False", "false"), ("None", "null"))

_WORD_CHAR = re.compile(r"[A-Za-z0-9_]")

_FENCED_BLOCK = re.compile(r"```[A-Za-z0-9_+-]*[ \t]*\r?\n?([\s\S]*?)```")


def _is_word_char(ch: str) -> bool:
    return bool(ch) and bool(_WORD_CHAR.match(ch))


def _strip_fences(text: str) -> str:
    """Return the contents of the first markdown code fence, or the text unchanged."""
    if "```" not in text:
        return text
    match = _FENCED_BLOCK.search(text)
    if match and match.group(1).strip():
        return match.group(1)
    return text.replace("```", " ")


def normalise_json_text(text: str) -> str:
    """Rewrite Python-flavoured JSON into JSON, leaving string contents untouched."""
    src = _strip_fences(text)
    out = []
    i = 0
    n = len(src)
    in_string = False
    while i < n:
        ch = src[i]
        if in_string:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(src[i + 1])
                i += 2
                continue
            if ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue

        replaced = False
        for py, js in _PY_LITERALS:
            if not src.startswith(py, i):
                continue
            before = src[i - 1] if i > 0 else ""
            after = src[i + len(py)] if i + len(py) < n else ""
            if _is_word_char(before) or _is_word_char(after):
                continue
            out.append(js)
            i += len(py)
            replaced = True
            break
        if replaced:
            continue

        if ch == ",":
            j = i + 1
            while j < n and src[j].isspace():
                j += 1
            if j < n and src[j] in "}]":
                i += 1
                continue

        out.append(ch)
        i += 1
    return "".join(out)


def parse_json_tolerant(text: str):
    """Parse ``text`` as JSON, retrying once on a normalised copy."""
    try:
        return json.loads(text), False
    except json.JSONDecodeError:
        pass

    normalised = normalise_json_text(text)
    if normalised == text:
        return None
    try:
        return json.loads(normalised), True
    except json.JSONDecodeError:
        return None
