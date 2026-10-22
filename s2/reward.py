"""Binary Math-Verify, DAPO group filter, length penalty vs shortest correct."""

DAPO_HI = 0.8


def dapo_keep(mean_r, hi=DAPO_HI):
    return 0.0 < mean_r <= hi


def length_penalize(rewards, lengths, max_len):
    """Wrong stays 0. Shortest correct stays 1. Longer correct linear down to 0 at max_len."""
    shortest = min((L for r, L in zip(rewards, lengths) if r > 0), default=None)
    if shortest is None:
        return [0.0] * len(rewards)
    span = max(max_len - shortest, 1)
    out = []
    for r, L in zip(rewards, lengths):
        if r <= 0:
            out.append(0.0)
        elif L <= shortest:
            out.append(1.0)
        else:
            out.append(max(0.0, 1.0 - (L - shortest) / span))
    return out


def shape_group(binary, lengths, max_len):
    """DAPO: if mean r not in (0, 0.8], zero the group. Else length-penalize corrects."""
    mean_r = sum(binary) / max(len(binary), 1)
    if not dapo_keep(mean_r):
        return [0.0] * len(binary)
    return length_penalize(binary, lengths, max_len)
