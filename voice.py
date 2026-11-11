"""
Kenya Maize Advisory - voice input and output
=============================================

The speech front-end for the advisory chat in ``app.py``. Two independent
directions, both local, both free, neither needing an API key:

* **Speech to text** uses ``faster-whisper`` (CTranslate2, int8, CPU). Whisper
  is multilingual, so one model covers English and Kiswahili; the detected
  language comes back with the transcript so the UI can tell the farmer what it
  heard before the question is sent to the RAG pipeline. Whisper has no Dholuo
  language token, so Dholuo is supported for typed questions and spoken answers
  but not for microphone input.
* **Text to speech** uses Meta's MMS VITS voices - one checkpoint per language
  (``facebook/mms-tts-eng``, ``facebook/mms-tts-swh`` and a community Dholuo
  MMS checkpoint). Each is about 145 MB, runs on CPU, and is a native voice
  rather than an English model mangling the text.

Everything is lazy. Importing this module loads nothing, and a missing
``faster-whisper`` install raises :class:`VoiceUnavailable`, which the app shows
as a message instead of a stack trace. That matches the rest of the project: no
heavy import at module scope, no crash when an optional piece is absent.

Why the cleanup before synthesis
--------------------------------
The RAG answers are written to be read, not spoken. They carry ``[1]`` citation
markers, markdown emphasis, units like ``kg/acre`` and bare digits. Fed straight
to a TTS model that is a stream of "bracket one bracket", "k g slash acre" and
numbers the voice does not know. :func:`speakable` rewrites the answer into
something a voice can actually say, and :func:`synthesize_answer` reads the
English disclaimer with the English voice even when the body is Swahili or
Dholuo, because the pipeline appends that sentence verbatim in all languages.

Latency and memory
------------------
``small`` Whisper is roughly 500 MB of weights in int8; the three MMS voices are
about 450 MB together. On a laptop that is fine. On Streamlit Community Cloud
the advisory tab is already over the memory cap before voice is added - see the
deployment note in ``README.md``.
"""

from __future__ import annotations

import io
import os
import re
import threading
import wave
from pathlib import Path
from typing import Any

import numpy as np

from rag_pipeline import DISCLAIMER, detect_language

#: Whisper checkpoint size. ``small`` is the smallest model that handles
#: Kiswahili well; ``base`` and ``tiny`` are available for low-memory hosts but
#: mis-hear domain words noticeably more often. Override with VOICE_ASR_MODEL.
ASR_MODEL = os.environ.get("VOICE_ASR_MODEL", "small")

#: MMS voice per language. English and Kiswahili are Meta checkpoints. Meta did
#: not publish a Dholuo MMS voice, so the default is a community checkpoint
#: trained on the same architecture; override it with VOICE_TTS_LUO.
TTS_VOICES = {
    "en": os.environ.get("VOICE_TTS_EN", "facebook/mms-tts-eng"),
    "sw": os.environ.get("VOICE_TTS_SW", "facebook/mms-tts-swh"),
    "luo": os.environ.get("VOICE_TTS_LUO", "rnjema-unima/mms-tts-luo-baseline"),
}

#: Bias the decoder towards the vocabulary the corpus actually uses. Without
#: this, "mbolea" comes back as "mboleha" and "DAP" as "yadab", which then fails
#: the advisory pipeline's topic guardrail. Mixed English/Kiswahili/Dholuo on
#: purpose: the recording's language is not known until after detection.
ASR_PROMPT = os.environ.get(
    "VOICE_ASR_PROMPT",
    "Kenyan maize farming. Kilimo cha mahindi Kenya: mbolea, DAP, CAN, ekari, "
    "hekta, ugonjwa, mavuno, mbegu, wadudu, aflatoxin, mahindi. "
    "Temo oduma: lowo, puodho, koth, kodhi, manyasi, kudni, tuoche, keyo.",
)

SAMPLE_RATE = 16_000

_asr = None
_asr_lock = threading.Lock()
_tts: dict[str, tuple[Any, Any, int]] = {}
_tts_lock = threading.Lock()


class VoiceUnavailable(RuntimeError):
    """Raised when a speech backend cannot be imported or loaded."""


# --------------------------------------------------------------------------- #
#  Speech to text
# --------------------------------------------------------------------------- #

def asr_available() -> bool:
    """True when ``faster-whisper`` is importable (the model may still download)."""
    try:
        import faster_whisper  # noqa: F401
    except Exception:
        return False
    return True


def _load_asr():
    global _asr
    if _asr is not None:
        return _asr
    with _asr_lock:
        if _asr is None:
            try:
                from faster_whisper import WhisperModel
            except Exception as exc:  # pragma: no cover - depends on the host
                raise VoiceUnavailable(
                    "Speech input needs the 'faster-whisper' package. Install it "
                    "with: pip install faster-whisper"
                ) from exc
            try:
                _asr = WhisperModel(ASR_MODEL, device="cpu", compute_type="int8")
            except Exception as exc:
                raise VoiceUnavailable(f"Could not load the speech model: {exc}") from exc
    return _asr


def _as_audio_stream(source: Any):
    """Normalise an ``UploadedFile``, ``bytes``, path or buffer to something
    ``faster-whisper`` can open. PyAV accepts file-like objects, so the bytes
    never touch the disk."""
    if isinstance(source, (bytes, bytearray, memoryview)):
        return io.BytesIO(bytes(source))
    if isinstance(source, (str, Path)):
        return str(source)
    getvalue = getattr(source, "getvalue", None)
    if callable(getvalue):
        return io.BytesIO(getvalue())
    return source


def _wav_to_array(data: bytes) -> np.ndarray | None:
    """Decode a mono/stereo 16-bit WAV to float32 at :data:`SAMPLE_RATE`.

    Streamlit's recorder produces WAV, so this covers the normal path without
    touching PyAV. That matters on macOS, where ``opencv`` (pulled in by
    ``ultralytics``) and ``av`` both ship a copy of libavdevice and the loader
    prints a duplicate-class warning when both are in the same process.
    """
    try:
        with wave.open(io.BytesIO(data)) as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())
    except Exception:
        return None
    if width != 2 or not frames:
        return None
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return _resample(samples, rate, SAMPLE_RATE)


def _decode(source: Any):
    """Return a 16 kHz float32 array when we can, otherwise a stream for PyAV."""
    stream = _as_audio_stream(source)
    if isinstance(stream, io.BytesIO):
        array = _wav_to_array(stream.getvalue())
        if array is not None:
            return array
    return stream


def transcribe(source: Any, language: str | None = None) -> dict:
    """Transcribe one recording.

    ``language`` may be ``"en"``, ``"sw"`` or ``None`` to auto-detect. Returns
    ``{"text", "language", "probability", "duration"}``; ``language`` is
    Whisper's own two-letter code, which the caller can use to preselect the
    reply language and the TTS voice. Dholuo is not one of Whisper's languages,
    so forcing ``"luo"`` raises :class:`VoiceUnavailable` instead of letting
    CTranslate2 fail with an obscure error.
    """
    if language == "luo":
        raise VoiceUnavailable(
            "Speech input does not support Dholuo yet - type the question instead."
        )
    model = _load_asr()
    try:
        segments, info = model.transcribe(
            _decode(source),
            language=language,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
            temperature=0.0,
            initial_prompt=ASR_PROMPT or None,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
    except Exception as exc:
        raise VoiceUnavailable(f"Could not transcribe that recording: {exc}") from exc

    return {
        "text": re.sub(r"\s+", " ", text),
        "language": getattr(info, "language", language or "en"),
        "probability": float(getattr(info, "language_probability", 0.0) or 0.0),
        "duration": float(getattr(info, "duration", 0.0) or 0.0),
    }


# --------------------------------------------------------------------------- #
#  Text to speech
# --------------------------------------------------------------------------- #

def tts_available() -> bool:
    try:
        import torch  # noqa: F401
        from transformers import VitsModel  # noqa: F401
    except Exception:
        return False
    return True


def _load_tts(lang: str):
    lang = lang if lang in TTS_VOICES else "en"
    if lang in _tts:
        return _tts[lang]
    with _tts_lock:
        if lang not in _tts:
            try:
                import torch  # noqa: F401
                from transformers import AutoTokenizer, VitsModel
            except Exception as exc:  # pragma: no cover - depends on the host
                raise VoiceUnavailable(
                    "Speech output needs 'transformers' with PyTorch installed."
                ) from exc
            name = TTS_VOICES[lang]
            try:
                tokenizer = AutoTokenizer.from_pretrained(name)
                model = VitsModel.from_pretrained(name)
            except Exception as exc:
                raise VoiceUnavailable(f"Could not load the {lang} voice: {exc}") from exc
            model.eval()
            _tts[lang] = (tokenizer, model, int(model.config.sampling_rate))
    return _tts[lang]


# -- number spelling -------------------------------------------------------- #
# VITS is character/phoneme based and reads bare digits unreliably, especially
# in Kiswahili. Spelling them out is the difference between "DAP 50 kg" being
# heard as "DAP five zero" and "DAP hamsini kilogramu".

_EN_ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen"
).split()
_EN_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_SW_ONES = (
    "sifuri moja mbili tatu nne tano sita saba nane tisa kumi "
    "kumi na moja kumi na mbili kumi na tatu kumi na nne kumi na tano "
    "kumi na sita kumi na saba kumi na nane kumi na tisa"
).split()
_SW_TENS = {
    2: "ishirini", 3: "thelathini", 4: "arobaini", 5: "hamsini",
    6: "sitini", 7: "sabini", 8: "themanini", 9: "tisini",
}
_LUO_ONES = (
    "non", "achiel", "ariyo", "adek", "ang'wen", "abich", "auchiel",
    "abiriyo", "aboro", "ochiko", "apar", "apar gi achiel", "apar gi ariyo",
    "apar gi adek", "apar gi ang'wen", "apar gi abich", "apar gi auchiel",
    "apar gi abiriyo", "apar gi aboro", "apar gi ochiko",
)
_LUO_TENS = {
    2: "pier ariyo", 3: "pier adek", 4: "pier ang'wen", 5: "pier abich",
    6: "pier auchiel", 7: "pier abiriyo", 8: "pier aboro", 9: "pier ochiko",
}


def _en_int(n: int) -> str:
    if n < 20:
        return _EN_ONES[n]
    if n < 100:
        tens, rest = divmod(n, 10)
        return _EN_TENS[tens] + (f" {_EN_ONES[rest]}" if rest else "")
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return f"{_EN_ONES[hundreds]} hundred" + (f" and {_en_int(rest)}" if rest else "")
    thousands, rest = divmod(n, 1000)
    head = "one thousand" if thousands == 1 else f"{_en_int(thousands)} thousand"
    return head + (f" {_en_int(rest)}" if rest else "")


def _sw_int(n: int) -> str:
    if n < 20:
        return _SW_ONES[n]
    if n < 100:
        tens, rest = divmod(n, 10)
        return _SW_TENS[tens] + (f" na {_sw_int(rest)}" if rest else "")
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        head = "mia moja" if hundreds == 1 else f"mia {_sw_int(hundreds)}"
        return head + (f" na {_sw_int(rest)}" if rest else "")
    thousands, rest = divmod(n, 1000)
    head = "elfu moja" if thousands == 1 else f"elfu {_sw_int(thousands)}"
    return head + (f" na {_sw_int(rest)}" if rest else "")


def _luo_int(n: int) -> str:
    if n < 20:
        return _LUO_ONES[n]
    if n < 100:
        tens, rest = divmod(n, 10)
        return _LUO_TENS[tens] + (f" gi {_luo_int(rest)}" if rest else "")
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        head = "mia achiel" if hundreds == 1 else f"mia {_luo_int(hundreds)}"
        return head + (f" gi {_luo_int(rest)}" if rest else "")
    thousands, rest = divmod(n, 1000)
    head = "alufu achiel" if thousands == 1 else f"alufu {_luo_int(thousands)}"
    return head + (f" gi {_luo_int(rest)}" if rest else "")


def _spell_numbers(text: str, lang: str) -> str:
    spell = {"sw": _sw_int, "luo": _luo_int}.get(lang, _en_int)
    point = {"sw": "nukta", "luo": "nukta"}.get(lang, "point")

    def decimals(match: re.Match) -> str:
        return f"{match.group(1)} {point} {match.group(2)}"

    def integers(match: re.Match) -> str:
        value = int(match.group(0))
        return spell(value) if value < 100_000 else match.group(0)

    text = re.sub(r"\b(\d+)\.(\d+)\b", decimals, text)
    return re.sub(r"\b\d+\b", integers, text)


_UNITS = {
    "en": [
        (r"\bkg\s*/\s*(?:acre|ekari)\b", "kilograms per acre"),
        (r"\bkg\s*/\s*(?:ha|hectare)\b", "kilograms per hectare"),
        (r"\bkg\b", "kilograms"),
        (r"\bml\b", "millilitres"),
        (r"\bha\b", "hectares"),
        (r"\bpH\b", "p H"),
        (r"%", " percent"),
        (r"\b(\d+)\s*C\b", r"\1 degrees Celsius"),
        (r"\be\.g\.\b", "for example"),
        (r"\bi\.e\.\b", "that is"),
        (r"&", " and "),
    ],
    "sw": [
        (r"\bkg\s*/\s*(?:acre|ekari)\b", "kilogramu kwa ekari"),
        (r"\bkg\s*/\s*(?:ha|hectare)\b", "kilogramu kwa hekta"),
        (r"\bkg\b", "kilogramu"),
        (r"\bml\b", "mililita"),
        (r"\bha\b", "hekta"),
        (r"\bpH\b", "p H"),
        (r"%", " asilimia"),
        (r"\b(\d+)\s*C\b", r"\1 digrii Selsiasi"),
        (r"\bkm\b", "kilomita"),
        (r"&", " na "),
    ],
    "luo": [
        (r"\bkg\s*/\s*(?:acre|ekari)\b", "kilogramu kuom ekari"),
        (r"\bkg\s*/\s*(?:ha|hectare)\b", "kilogramu kuom hekta"),
        (r"\bkg\b", "kilogramu"),
        (r"\bml\b", "mililita"),
        (r"\bha\b", "hekta"),
        (r"\bpH\b", "p H"),
        (r"%", " asilimia"),
        (r"\b(\d+)\s*C\b", r"\1 digrii Selsiasi"),
        (r"\bkm\b", "kilomita"),
        (r"\be\.g\.\b", "ka lantie"),
        (r"\bi\.e\.\b", "ma en"),
        (r"&", " gi "),
    ],
}


def speakable(text: str, lang: str | None = None) -> str:
    """Rewrite a written RAG answer into something a voice can read.

    Removes citation markers and markdown, expands units, and spells numbers.
    The disclaimer is kept - for a farmer who cannot read the screen it is the
    most important sentence in the reply.
    """
    if not text:
        return ""
    lang = lang or detect_language(text)

    text = re.sub(r"\[\d+\]", " ", text)                  # [1], [2]
    text = re.sub(r"\((?:see\s+)?source[^)]*\)", " ", text, flags=re.I)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[*_`#>|~]+", " ", text)               # markdown furniture
    text = re.sub(r"^\s*[-•]\s*", " ", text, flags=re.M)  # bullets

    # "18-27 C" and "2-3 seeds" must be "to"/"hadi"/"nyaka", not a compound word.
    joiner = {"sw": " hadi ", "luo": " nyaka "}.get(lang, " to ")
    text = re.sub(r"(?<=\d)\s*[-\u2013]\s*(?=\d)", joiner, text)
    text = re.sub(r"\s+-\s+", ", ", text)                 # "a - b" reads badly

    for pattern, replacement in _UNITS[lang]:
        text = re.sub(pattern, replacement, text, flags=re.I)

    text = _spell_numbers(text, lang)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def _resample(samples: np.ndarray, rate: int, target: int) -> np.ndarray:
    if rate == target or samples.size == 0:
        return samples
    duration = samples.size / rate
    target_size = max(1, int(round(duration * target)))
    source_x = np.linspace(0.0, 1.0, samples.size, endpoint=False)
    target_x = np.linspace(0.0, 1.0, target_size, endpoint=False)
    return np.interp(target_x, source_x, samples).astype(np.float32)


def _wav_bytes(samples: np.ndarray, rate: int) -> bytes:
    pcm = (np.clip(np.asarray(samples, dtype="float32"), -1.0, 1.0) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def _sentences(text: str, limit: int = 220) -> list[str]:
    """Split into speakable chunks. VITS degrades on long inputs and has no
    notion of a paragraph break, so sentence-sized pieces also sound better."""
    parts = re.split(r"(?<=[.!?;:])\s+", text.strip())
    chunks: list[str] = []
    buffer = ""
    for part in parts:
        candidate = f"{buffer} {part}".strip()
        if buffer and len(candidate) > limit:
            chunks.append(buffer)
            buffer = part
        else:
            buffer = candidate
    if buffer:
        chunks.append(buffer)
    return [chunk for chunk in chunks if chunk.strip()]


def _render(text: str, lang: str) -> np.ndarray:
    import torch

    tokenizer, model, rate = _load_tts(lang)
    pieces: list[np.ndarray] = []
    for chunk in _sentences(text):
        inputs = tokenizer(chunk, return_tensors="pt")
        with torch.no_grad():
            waveform = model(**inputs).waveform[0].detach().cpu().numpy()
        pieces.append(_resample(np.asarray(waveform, dtype=np.float32), rate, SAMPLE_RATE))
        pieces.append(np.zeros(int(0.12 * SAMPLE_RATE), dtype=np.float32))
    if not pieces:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(pieces)


def synthesize(text: str, lang: str | None = None) -> bytes:
    """Render ``text`` as a mono 16-bit WAV and return the bytes."""
    lang = lang or detect_language(text)
    spoken = speakable(text, lang)
    if not spoken:
        return b""
    return _wav_bytes(_render(spoken, lang), SAMPLE_RATE)


def synthesize_answer(answer: str, lang: str | None = None) -> bytes:
    """Render a full RAG answer, giving the trailing disclaimer an English voice.

    ``AdvisoryPipeline`` appends :data:`DISCLAIMER` verbatim in every language,
    so a Swahili or Dholuo reply ends with one English sentence. Reading that
    with the local voice is unintelligible; reading it separately is not.
    """
    lang = lang or detect_language(answer)
    if not answer:
        return b""

    body, _, _tail = answer.partition(DISCLAIMER)
    parts: list[np.ndarray] = []

    body_spoken = speakable(body, lang)
    if body_spoken:
        parts.append(_render(body_spoken, lang))

    if DISCLAIMER in answer:
        parts.append(np.zeros(int(0.35 * SAMPLE_RATE), dtype=np.float32))
        parts.append(_render(speakable(DISCLAIMER, "en"), "en"))

    if not parts:
        return b""
    return _wav_bytes(np.concatenate(parts), SAMPLE_RATE)
