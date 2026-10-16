from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class TranscriptSegment:
    start: float
    end: float
    speaker: str
    text: str


class TranscriptStreamParser:
    _SEEK_START = 0
    _READ_START = 1
    _EXPECT_SPEAKER_OPEN = 2
    _READ_SPEAKER = 3
    _READ_TEXT = 4
    _READ_END = 5
    _AFTER_END = 6

    def __init__(self, strip_text=True, skip_empty=True):
        self.strip_text = strip_text
        self.skip_empty = skip_empty
        self.reset()

    def reset(self):
        self._state = self._SEEK_START
        self._token = []
        self._text = []
        self._pending_after_end = []
        self._start = None
        self._end = None
        self._end_token = ""
        self._speaker = None

    def feed(self, chunk):
        segments = []
        self.feed_into(chunk, segments.append)
        return segments

    def feed_into(self, chunk, emit):
        for ch in chunk:
            state = self._state
            if state == self._SEEK_START:
                self._seek_start(ch)
            elif state == self._READ_START:
                self._read_start(ch)
            elif state == self._EXPECT_SPEAKER_OPEN:
                self._expect_speaker_open(ch)
            elif state == self._READ_SPEAKER:
                self._read_speaker(ch)
            elif state == self._READ_TEXT:
                self._read_text(ch)
            elif state == self._READ_END:
                self._read_end(ch, emit)
            elif state == self._AFTER_END:
                self._after_end(ch, emit)

    def close(self):
        segments = []
        self.close_into(segments.append)
        return segments

    def close_into(self, emit):
        if self._state == self._AFTER_END:
            self._emit_segment(emit)
        self.reset()

    def _seek_start(self, ch):
        if ch == "[":
            self._token.clear()
            self._state = self._READ_START

    def _read_start(self, ch):
        if ch == "]":
            start = _parse_timestamp(self._token)
            if start is None:
                self.reset()
                return
            self._start = start
            self._state = self._EXPECT_SPEAKER_OPEN
            self._token.clear()
            return
        if _is_timestamp_char(ch):
            self._token.append(ch)
            if len(self._token) <= 32:
                return
        self.reset()
        if ch == "[":
            self._state = self._READ_START

    def _expect_speaker_open(self, ch):
        if ch == "[":
            self._token.clear()
            self._state = self._READ_SPEAKER
        elif not ch.isspace():
            self._speaker = "S01"
            self._text.clear()
            self._text.append(ch)
            self._state = self._READ_TEXT

    def _read_speaker(self, ch):
        if ch == "]":
            speaker = _parse_speaker(self._token)
            if speaker is None:
                self.reset()
                return
            self._speaker = speaker
            self._text.clear()
            self._state = self._READ_TEXT
            self._token.clear()
            return
        if _is_speaker_char(ch):
            self._token.append(ch)
            if len(self._token) <= 16:
                return
        self.reset()
        if ch == "[":
            self._state = self._READ_START

    def _read_text(self, ch):
        if ch == "[":
            self._token.clear()
            self._state = self._READ_END
        else:
            self._text.append(ch)

    def _read_end(self, ch, emit):
        if ch == "]":
            end = _parse_timestamp(self._token)
            if end is not None and self._start is not None and end >= self._start:
                self._end = end
                self._end_token = "".join(self._token)
                self._pending_after_end.clear()
                self._state = self._AFTER_END
            else:
                self._text.append("[")
                self._text.extend(self._token)
                self._text.append("]")
                self._state = self._READ_TEXT
            self._token.clear()
            return
        if _is_timestamp_char(ch):
            self._token.append(ch)
            if len(self._token) <= 32:
                return
        self._text.append("[")
        self._text.extend(self._token)
        self._text.append(ch)
        self._token.clear()
        self._state = self._READ_TEXT

    def _after_end(self, ch, emit):
        if ch == "[":
            self._emit_segment(emit)
            self._token.clear()
            self._state = self._READ_START
            return
        if ch.isspace():
            self._pending_after_end.append(ch)
            return
        self._text.append("[")
        self._text.append(self._end_token)
        self._text.append("]")
        self._text.extend(self._pending_after_end)
        self._text.append(ch)
        self._pending_after_end.clear()
        self._end = None
        self._end_token = ""
        self._state = self._READ_TEXT

    def _emit_segment(self, emit):
        if self._start is None or self._end is None or self._speaker is None:
            self.reset()
            return
        text = "".join(self._text)
        if self.strip_text:
            text = text.strip()
        if text or not self.skip_empty:
            emit(TranscriptSegment(start=self._start, end=self._end, speaker=self._speaker, text=text))
        self._token.clear()
        self._text.clear()
        self._pending_after_end.clear()
        self._start = None
        self._end = None
        self._end_token = ""
        self._speaker = None
        self._state = self._SEEK_START


def parse_transcript(text):
    parser = TranscriptStreamParser()
    segments = parser.feed(text)
    segments.extend(parser.close())
    return segments


def _parse_timestamp(chars):
    if not chars:
        return None
    dot_count = 0
    digit_count = 0
    for ch in chars:
        if "0" <= ch <= "9":
            digit_count += 1
        elif ch == ".":
            dot_count += 1
            if dot_count > 1:
                return None
        else:
            return None
    if digit_count == 0:
        return None
    return float("".join(chars))


def _parse_speaker(chars):
    if len(chars) < 2 or chars[0] != "S":
        return None
    for ch in chars[1:]:
        if not ("0" <= ch <= "9"):
            return None
    return "".join(chars)


def _is_timestamp_char(ch):
    return ("0" <= ch <= "9") or ch == "."


def _is_speaker_char(ch):
    return ch == "S" or ("0" <= ch <= "9")
