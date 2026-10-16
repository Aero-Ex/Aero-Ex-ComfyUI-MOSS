import json


def format_timestamp(sec):
    sec = max(0.0, float(sec))
    ms = int(round((sec - int(sec)) * 1000.0))
    total = int(sec)
    if ms == 1000:
        total += 1
        ms = 0
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return "{:02d}:{:02d}:{:02d},{:03d}".format(h, m, s, ms)


def segments_to_srt(segments, show_speaker=True):
    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append("{} --> {}".format(format_timestamp(seg.start), format_timestamp(seg.end)))
        lines.append("{}: {}".format(seg.speaker, seg.text) if show_speaker and seg.speaker else seg.text)
        lines.append("")
    return "\n".join(lines).strip() + ("\n" if lines else "")


def segments_to_json(segments):
    return json.dumps([{"start": s.start, "end": s.end, "speaker": s.speaker, "text": s.text} for s in segments], ensure_ascii=False, indent=2) + "\n"


def segments_to_raw(segments):
    return "".join("[{:.2f}][{}]{}[{:.2f}]".format(s.start, s.speaker, s.text, s.end) for s in segments)
