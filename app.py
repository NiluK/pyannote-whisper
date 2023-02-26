import whisper
from pyannote.audio import Pipeline
from pyannote_whisper.utils import diarize_text
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization", use_auth_token="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")
print(pipeline)
model = whisper.load_model("base.en")
print(model)
asr_result = model.transcribe("data/ep1.mp3")
print(asr_result['text'])
diarization_result = pipeline("data/ep1.mp3")
print(diarization_result)
final_result = diarize_text(asr_result, diarization_result)


for seg, spk, sent in final_result:
    line = f'{seg.start:.2f} {seg.end:.2f} {spk} {sent}'
    print(line)
