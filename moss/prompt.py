DEFAULT_PROMPT = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
    "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
    "并在段末标注结束时间戳，以清晰标明该段语音范围。"
)

AUDIO_TOKENS_PER_SECOND = 12.5
TIME_MARKER_EVERY_SECONDS = 5


def prompt_ids(tokenizer, user_text, num_audio_tokens, audio_token_id):
    """Chat prompt with the audio span expanded; the span carries a digit time marker every 5 s, as upstream."""
    before = tokenizer.encode("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|audio_start|>").ids
    after = tokenizer.encode("<|audio_end|>\n" + user_text + "<|im_end|>\n<|im_start|>assistant\n").ids
    tokens_per_marker = int(AUDIO_TOKENS_PER_SECOND * TIME_MARKER_EVERY_SECONDS)
    span, consumed = [], 0
    for sec in range(TIME_MARKER_EVERY_SECONDS, int(num_audio_tokens / AUDIO_TOKENS_PER_SECOND) + 1, TIME_MARKER_EVERY_SECONDS):
        pos = sec // TIME_MARKER_EVERY_SECONDS * tokens_per_marker
        span += [audio_token_id] * (pos - consumed) + [tokenizer.token_to_id(d) for d in str(sec)]
        consumed = pos
    return before + span + [audio_token_id] * (num_audio_tokens - consumed) + after
