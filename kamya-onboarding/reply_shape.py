"""Bound what Mira says, without damaging what she says.

Pure logic, no pipecat import, so it can be tested directly. `bot.py`'s
ReplyShapeFilter is a thin frame-processor wrapper around ReplyShaper.

THE POINT. Length rules were asked for in prose three separate times -- in
GLOBAL_RULES, in the P-3 stage budgets, and in the node prompts -- and a live
call still produced eleven questions in one breath. So it is enforced here.

But enforcement that truncates is not free. Mira's question is almost always
the LAST sentence of her reply, because every rule tells her to end with one.
So a naive word cap cuts off exactly the part that matters and leaves a reply
that only makes statements -- and an onboarding call where Mira stops asking
is a call that stalls, with the user left to fill the silence. That is a
worse failure than a reply running a few words long.

So the two cuts are deliberately NOT symmetrical:

  FIRST QUESTION MARK -- hard cut, always safe. Everything up to and including
      the "?" is kept, so the question always survives; only a SECOND question
      and anything trailing it is dropped. This is what actually bounds a
      normal reply, and it costs nothing.

  WORD CAP -- only applies ONCE THE QUESTION HAS ALREADY BEEN ASKED, i.e. to
      trailing chatter. Before the question, going over the cap is NOT cut,
      because cutting there is precisely what would discard the question.
      Length before the question is the prompt's job; a generous runaway
      ceiling is the only backstop, for a model that never punctuates at all.
"""


class ReplyShaper:
    """Feed streamed text in, get the text that should actually be spoken out.

    Usage:
        sh = ReplyShaper(word_cap=32)
        for chunk in stream:
            keep, stop = sh.feed(chunk)
            if keep: emit(keep)
            if stop: break
    """

    SENT_END = ("।", ".", "?", "!")
    QUESTION = ("?", "？")

    # A model that emits no sentence-ending punctuation at all would otherwise
    # run to max_tokens. Multiple of the cap, generous on purpose: it must
    # never fire on a reply that is merely a bit long, only on a broken one.
    RUNAWAY_MULTIPLE = 4

    def __init__(self, word_cap=32):
        self._cap = word_cap
        self.reset()

    def reset(self):
        self.saw_punctuation = False
        self.cut = False
        self.reason = ""
        self.dropped = 0
        self.words = 0
        self.asked = False          # has the question mark been emitted yet?
        self.sent_any = False
        self.ends_punctuated = False

    def feed(self, text):
        """Returns (text_to_emit, stop_now). `text` is one streamed chunk."""
        text = text or ""
        if self.cut:
            self.dropped += len(text)
            return "", True

        # 1. The question mark ends the reply. Keep it -- never drop it.
        qi = min((text.index(q) for q in self.QUESTION if q in text), default=-1)
        if qi >= 0:
            keep = text[: qi + 1]
            self.dropped += len(text) - len(keep)
            self.cut, self.reason = True, "question asked"
            self.words += len(keep.split())
            self.asked = True
            if keep:
                self.sent_any, self.ends_punctuated = True, True
            return keep, True

        self.words += len(text.split())
        if text.strip():
            self.sent_any = True
            self.ends_punctuated = text.rstrip()[-1] in self.SENT_END
        if any(c in text for c in self.SENT_END):
            self.saw_punctuation = True

        # 2. Over the cap BEFORE the question has been asked. Do NOT cut --
        #    cutting here is exactly what would throw the question away and
        #    leave the call stalled. Let it reach its "?"; only a genuinely
        #    broken generation is stopped.
        # The guard is for a model emitting one endless unpunctuated string.
        # It must NOT fire on a reply that is merely long but well formed:
        # cutting there discards the question, which is the exact failure this
        # class was restructured to stop. So word count alone is not enough --
        # the reply must also have produced no sentence end at all.
        if (self.words > self._cap * self.RUNAWAY_MULTIPLE
                and not self.saw_punctuation):
            self.dropped += len(text)
            self.cut, self.reason = True, "unpunctuated runaway"
            return "", True

        return text, False

    def overlong(self):
        """True if the finished reply exceeded the cap. For logging only --
        the reply was still delivered whole, on purpose."""
        return self.words > self._cap
