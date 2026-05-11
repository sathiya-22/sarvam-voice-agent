"""Sarvam Voice Agent — record → STT → Sarvam-M → TTS → play."""
import os, io, tempfile
import numpy as np, sounddevice as sd, soundfile as sf
from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv()
client = SarvamAI(api_subscription_key=os.environ["SARVAM_API_KEY"])
SR = 16000
SPEAKERS = {"ta-IN":"anushka","hi-IN":"meera","te-IN":"arvind","kn-IN":"amol","en-IN":"anushka"}

def record(sec=6):
    print(f"Recording {sec}s..."); a = sd.rec(int(sec*SR), samplerate=SR, channels=1, dtype="float32"); sd.wait(); return a.flatten()

def stt(audio, lang):
    buf = io.BytesIO(); sf.write(buf, audio, SR, format="WAV"); buf.seek(0)
    return client.speech_to_text.transcribe(file=buf, model="saaras:v3", language_code=lang).transcript

def chat(msg, history, lang):
    msgs = [{"role":"system","content":f"You are a helpful support assistant. Respond in {lang}."}]
    for u, b in history: msgs += [{"role":"user","content":u},{"role":"assistant","content":b}]
    msgs.append({"role":"user","content":msg})
    return client.chat.completions(messages=msgs, model="sarvam-m").choices[0].message.content

def speak(text, lang):
    r = client.text_to_speech.convert(text=text[:500], target_language_code=lang, speaker=SPEAKERS.get(lang,"anushka"))
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(r.audios[0]); tmp=f.name
    d, sr = sf.read(tmp); sd.play(d, sr); sd.wait()

def run(lang="hi-IN"):
    print(f"\n🎙️ Sarvam Voice Agent — {lang} | Ctrl+C to exit\n")
    history = []
    while True:
        try:
            audio=record(); user=stt(audio,lang); print(f"You: {user}")
            if not user.strip(): continue
            reply=chat(user,history,lang); print(f"Agent: {reply}\n")
            history.append((user,reply)); speak(reply,lang)
        except KeyboardInterrupt: print("\nGoodbye!"); break

if __name__=="__main__":
    import sys; run(sys.argv[1] if len(sys.argv)>1 else "hi-IN")
