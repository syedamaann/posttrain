"""Outcome reward for S2: closed think, a real body, boxed answer, then anti-rumination."""

DAPO_HI = 0.8
MIN_THINK = 128
OVERLONG_CACHE = 256


def think_answer(text):
    """Span after the last </think>. Missing the tag means there is no answer to score."""
    if "</think>" not in text:
        return None
    return text.split("</think>")[-1]


def think_body(text):
    """Text before the last </think>. The whole completion when the tag is missing."""
    if "</think>" not in text:
        return text
    return text.rsplit("</think>", 1)[0]


def closed_verify(text, gold, verify, think_tokens):
    """1 only for a closed think of at least MIN_THINK tokens whose answer span boxes and verifies."""
    span = think_answer(text)
    if span is None or think_tokens < MIN_THINK or "\\boxed{" not in span:
        return 0.0
    try:
        return float(verify(gold, span))
    except Exception:
        return 0.0


def dapo_keep(mean_r, hi=DAPO_HI):
    return 0.0 < mean_r <= hi


def overlong_penalize(rewards, lengths, max_len, cache=OVERLONG_CACHE):
    """Correct and finished by max_len-cache stays 1. Linear down to 0 at max_len. Wrong stays 0."""
    start = max(max_len - cache, 0)
    span = max(max_len - start, 1)
    out = []
    for r, L in zip(rewards, lengths):
        if r <= 0 or L >= max_len:
            out.append(0.0)
        elif L <= start:
            out.append(1.0)
        else:
            out.append((max_len - L) / span)
    return out


def shape_group(binary, lengths, max_len):
    """DAPO on the 0/1 scores. A kept group then decays only in the last cache tokens."""
    mean_r = sum(binary) / max(len(binary), 1)
    if not dapo_keep(mean_r):
        return [0.0] * len(binary)
    return overlong_penalize(binary, lengths, max_len)
