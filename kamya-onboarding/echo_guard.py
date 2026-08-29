"""Recognise Mira's own voice arriving back through the microphone.

Pure logic, no pipecat, so the matching rules can be tested directly.

WHY THIS EXISTS. On the 2026-08-28 call Mira's greeting began "नमस्ते Dhruv!"
and one second later a USER transcript of "नमस्ते." arrived -- while she was
still speaking. It counted as a barge-in, cut her off mid-sentence, and was
fed back to her as the user's turn, so she answered her own greeting. The user
had not said a word. The browser has echo cancellation enabled; it did not
hold.

THE RISK RUNS THE OTHER WAY TOO. A false positive here silently deletes
something the user really said, which is worse than the echo: they repeat
themselves and Mira appears deaf. So every rule below is deliberately narrow,
and when in doubt this returns False and lets the transcript through.
"""
import re

# Above this, an overlapping transcript is a genuine barge-in. Real echo is
# short -- the first word or two before the canceller catches up.
MAX_ECHO_WORDS = 6

# Shorter than this and a match means nothing: "हाँ" appears inside half of
# what Mira ever says, and it is also the single most common real answer.
MIN_ECHO_CHARS = 3


def normalise(s):
    """Letters and digits only, lowercased.

    What TTS was handed and what STT hears back never agree on punctuation or
    spacing, so comparing raw strings finds nothing.
    """
    return re.sub(r"[^\wऀ-ॿ]+", "", (s or "").lower())


def is_echo(heard, said, max_words=MAX_ECHO_WORDS):
    """True if `heard` (a user transcript) is really `said` (Mira's speech).

    Both conditions must hold: short enough to be echo rather than speech, and
    actually contained in what she is saying.
    """
    if not heard or not said:
        return False
    if len(heard.split()) > max_words:
        return False
    h, s = normalise(heard), normalise(said)
    if len(h) < MIN_ECHO_CHARS:
        return False
    return h in s
