"""
Kenya Maize Advisory - reusable RAG pipeline
=============================================

Loads the artifacts exported by ``RAG_Advisory_Chatbot.ipynb`` section 11
(``rag_artifacts/export/``) and answers farmer questions by retrieving evidence
from the corpus and generating a grounded, cited reply.

The Streamlit app imports this module and calls :meth:`AdvisoryPipeline.ask`.
``multilingual-e5-small`` (about 0.5 GB) embeds queries locally, and answers
are written by a hosted language model through :mod:`llm_client` - see that
module for the provider settings. A local 1.5B generator plus NLLB needs about
6 GB, far above Streamlit Community Cloud's ~2.7 GB ceiling.

Swahili and Dholuo questions are pivoted through English. Translation uses
NLLB-200 distilled (600M) quantised to int8 with CTranslate2 (about 0.6 GB),
loaded only when a non-English question arrives; set
``ADVISORY_TRANSLATION_BACKEND=google`` to use the hosted Google Translate API
instead, or ``=llm`` to translate with the chat model. The translation model is
never loaded for English questions, so the common path stays small.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from pathlib import Path

import numpy as np

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent
EXPORT_DIR = ROOT / "rag_artifacts" / "export"
TOP_K = 5

DISCLAIMER = "For decisions that affect your harvest, consult your county extension officer."

#: Languages the chat accepts. Dholuo is a low-resource language in NLLB, but
#: the translation pivot means retrieval always happens in English.
LANGUAGES = ("en", "sw", "luo")
LANGUAGE_LABELS = {"en": "English", "sw": "Kiswahili", "luo": "Dholuo"}
NLLB_CODES = {"en": "eng_Latn", "sw": "swh_Latn", "luo": "luo_Latn"}

#: int8 CTranslate2 conversion of ``facebook/nllb-200-distilled-600M``. It is
#: already installed via faster-whisper and keeps translation off the RAM
#: budget that the hosted answer model freed up.
DEFAULT_CT2_TRANSLATION_MODEL = "JustFrederik/nllb-200-distilled-600M-ct2-int8"

SW_STOP = {
    "ya", "na", "wa", "kwa", "cha", "hii", "hiyo", "hizo", "wao", "yake", "wake",
    "katika", "kuhusu", "ambayo", "ambao", "hivyo", "lakini", "sana", "aidha", "bali",
    "kabla", "baada", "hadi", "pia", "kila", "wakulima", "shamba", "kilimo", "mahindi",
    "mazao", "udongo", "mbolea", "wadudu", "ugonjwa", "magonjwa", "mbegu", "nchini",
    "kutumia", "kupitia", "zaidi", "chini", "juu", "wakati", "hali", "hewa",
    "ni", "nini", "gani", "yapi", "ipi", "nani", "wapi", "lini", "vipi", "je",
    "jinsi", "naweza", "nifanye", "ninawezaje", "ninapaswa",
    "kupanda", "kuvuna", "dalili", "tiba", "kutibu", "kuzuia",
}

#: Distinctive Dholuo words used for routing and follow-up detection. Short,
#: ambiguous forms ("to", "ma", "ne", "en", "in") are deliberately absent:
#: they collide with English or Swahili and would misroute a question.
LUO_STOP = {
    "oduma", "lowo", "puodho", "koth", "chiemo", "cham", "keyo", "kano",
    "tuo", "tuoche", "kudni", "kodhi", "manyasi", "yath", "kue", "piny",
    "dala", "ji", "dhano", "ng'ato", "gima", "gik", "kendo", "bende",
    "kuom", "mondo", "nyaka", "magi", "mago", "mae", "mano", "mar",
    "nang'o", "ang'o", "nade", "ere", "ng'ano", "nyalo", "dwaro", "adwaro",
    "ng'eyo", "ang'eyo", "kony", "timo", "wacho", "winjo", "neno", "kelo",
    "miyo", "kawo", "chon", "koro", "ndalo", "higa", "dwe", "ot", "pi",
    "mach", "ber", "rach", "matin", "maduong'", "onego", "thiedho", "pidho",
    "jiwo", "onge", "nitie", "nigi", "ng'adi", "kawuono", "kiny", "nyoro",
    "odiechieng'", "otieno", "okinyi", "pang", "chamo", "tiyo", "rito",
    "dongo", "medo", "ochiek",
}

TOPIC_WORDS = [
    "maize", "corn", "crop", "farm", "field", "soil", "rain", "seed", "fertilis",
    "fertiliz", "manure", "pest", "disease", "insect", "weed", "harvest", "yield",
    "plant", "variety", "varieties", "irrigation", "storage", "grain", "cob",
    "leaf", "blight", "rust", "hectare", "acre", "agronom", "extension",
    "mahindi", "mbolea", "shamba", "kilimo", "wakulima", "mazao", "ugonjwa",
    "magonjwa", "wadudu", "mbegu", "udongo", "mvua", "kuvuna", "mavuno",
    "kupanda", "dalili", "tiba", "kuzuia", "ipil",
    "oduma", "lowo", "puodho", "koth", "kodhi", "manyasi", "kudni", "tuo",
    "tuoche", "keyo", "kano", "cham", "chiemo", "yath", "kue", "pidho", "pang",
]

REFERENTIAL = re.compile(
    r"\b(it|its|they|them|their|that|this|these|those|the same|the above"
    r"|hii|hiyo|hizi|hizo|huo|hayo|hao|wao|yake|wake|zao"
    r"|mae|mano|magi|mago|ene)\b", re.I
)

LIST_TRIGGER = (r"(list|enumerate|name|what are|which are|all|aina|gani|yapi|ipi|zipi"
                r"|nang'o|ang'o|gin mage|gik mage|magi)")
LIST_NOUN = (r"(diseases?|pests?|insects?|weeds?|varieties?|varieti|symptoms?"
             r"|problems?|magonjwa|wadudu|dalili|tuoche|kudni|kodhi)")
LIST_NOUN_EN = {"magonjwa": "diseases", "wadudu": "pests", "dalili": "symptoms",
                "varieti": "varieties", "tuoche": "diseases", "kudni": "pests",
                "kodhi": "varieties"}

LIST_INTENT = re.compile(
    r"\b" + LIST_TRIGGER + r"\b.{0,60}?\b" + LIST_NOUN + r"\b"
    r"|\b" + LIST_NOUN + r"\b.{0,60}?\b" + LIST_TRIGGER + r"\b",
    re.I | re.S
)

REFUSAL_RULES = [
    ("live_prices", [r"\bprices?\b", r"how much is", r"\bbei\b", r"ruzuku",
                     r"subsid", r"market rate", r"\bsoko\b", r"\bpesa\b",
                     r"\bng'adi\b"]),
    ("human_medicine", [r"malaria", r"\bhuman\b", r"binadamu", r"\bdawa\b",
                        r"medicine", r"\bdaktari\b", r"ngima dhano"]),
]

REFUSALS = {
    "live_prices": {
        "en": "I do not have live market prices or subsidy information - my sources are "
              "static documents. Check today's prices with your local produce buyer, the "
              "county market office, or the e-Mkulima platform.",
        "sw": "Sina bei za soko za sasa wala taarifa za ruzuku - vyanzo vyangu ni vitabu "
              "vya zamani. Uliza mnunuzi wako wa mazao, ofisi ya soko ya kaunti, au tumia "
              "jukwaa la e-Mkulima.",
        "luo": "Aonge gi bei mag soko manyaka koro kata weche mag ruzuku - gik ma "
               "ang'eyo gin gik ma nondiki. Penj jachiew e soko, ofisi mar soko mar "
               "kaunti, kata tiy gi e-Mkulima.",
    },
    "human_medicine": {
        "en": "I only advise on growing maize. Questions about human health should go to "
              "a clinic or hospital.",
        "sw": "Nashauri tu kuhusu ukuzaji wa mahindi. Maswali kuhusu afya ya binadamu "
              "yawaoneshe kwa daktari.",
        "luo": "Ajiwo mana kuom temo oduma. Penjo kuom ngima dhano onego dhi ne daktari.",
    },
    "off_topic": {
        "en": "I only answer questions about growing maize in Kenya - diseases, planting, "
              "fertiliser, pests, harvest and storage. Ask me about maize!",
        "sw": "Najibu maswali tu kuhusu ukuzaji wa mahindi nchini Kenya - magonjwa, "
              "kupanda, mbolea, wadudu, mavuno na uhifadhi. Niulize kuhusu mahindi!",
        "luo": "Ajiwo mana kuom gik ma ochan e temo oduma e Kenya - tuoche, pidho, "
               "manyasi, kudni, keyo gi kano. Penja kuom oduma!",
    },
}

REFUSAL_TEXTS = {text for category in REFUSALS.values() for text in category.values()}

PROMPT_TEMPLATE = """You are the Kenya Maize Advisory Chatbot, an assistant for smallholder maize farmers.

Answer ONLY from the numbered sources below.
- Cite sources like [1] or [2][3] after every claim.
- If the sources do not contain the answer, say so and advise consulting a county extension officer.
- Reply in the SAME language the question was asked in (English, Kiswahili or Dholuo).
- For pesticides, only state what the source says (product, rate, waiting period) and add a wear-gloves/wash warning.
- Answer the LATEST question. If it is a follow-up, use the conversation to work out what it refers to, keep the same subject, and do NOT repeat or reword an earlier answer.
- Use the conversation so far only to understand the latest question; the facts must still come from the sources.
- If the question asks for a list, include every relevant item named in the sources, not only the first one.
- Do not copy apologies, refusals or "I cannot" phrasing out of the conversation; if the sources answer the latest question, answer it.
- Never describe yourself as an AI model and never name your developer.
- End every answer with: "For decisions that affect your harvest, consult your county extension officer."
{history}
Sources:
{context}

Question: {question}
Answer:"""


def _word_set(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z']+", text.lower()))


def detect_language(text: str) -> str:
    """Best guess between ``en``, ``sw`` and ``luo``.

    Dholuo must score strictly higher than Swahili, because Dholuo questions
    often borrow Swahili farming words (mbolea, mahindi) and a tie would route
    them the wrong way. Two matching stopwords are required, so a single
    loanword does not change the answer language.
    """
    words = _word_set(text)
    sw = len(words & SW_STOP)
    luo = len(words & LUO_STOP)
    if luo >= 2 and luo > sw:
        return "luo"
    if sw >= 2:
        return "sw"
    return "en"


def looks_swahili(text: str) -> bool:
    return detect_language(text) == "sw"


def looks_dholuo(text: str) -> bool:
    return detect_language(text) == "luo"


def mentions_topic(text: str) -> bool:
    return bool(re.search(r"\b(" + "|".join(TOPIC_WORDS) + ")", text.lower()))


def is_follow_up(question: str) -> bool:
    text = question.lower()
    words = re.findall(r"[a-z0-9']+", text)
    if mentions_topic(text) and len(words) > 4:
        return False
    if REFERENTIAL.search(text):
        return True
    return len(words) <= 4


def refuse(question: str, has_context: bool = False):
    text = question.lower()
    lang = detect_language(question)
    for category, patterns in REFUSAL_RULES:
        if any(re.search(pattern, text) for pattern in patterns):
            return REFUSALS[category].get(lang) or REFUSALS[category]["en"]
    if not mentions_topic(text) and not (has_context and is_follow_up(question)):
        return REFUSALS["off_topic"].get(lang) or REFUSALS["off_topic"]["en"]
    return None


def normalize_answer(answer: str) -> str:
    """Tidy model formatting that would otherwise fail the guardrails.

    Chat models occasionally emit full-width CJK brackets around citations
    (``【2】``) or wrap the disclaimer in markdown emphasis. Both mean the same
    thing, so normalise them before validation instead of spending a retry.
    """
    answer = re.sub(r"[【［]\s*(\d+)\s*[】］]", r"[\1]", answer)
    index = answer.rfind(DISCLAIMER)
    if index != -1 and re.fullmatch(r"[\s*_>`\"']*", answer[index + len(DISCLAIMER):]):
        answer = answer[:index].rstrip(" \t\n*_>`\"'") + " " + DISCLAIMER
    return answer.strip()


def validate_answer(answer: str, question: str, n_sources: int):
    problems = []
    citations = [int(n) for n in re.findall(r"\[(\d+)\]", answer)]
    if not citations:
        problems.append("no citation")
    if any(n > n_sources for n in citations):
        problems.append("citation points outside the given sources")
    if not answer.strip().endswith(DISCLAIMER):
        problems.append("disclaimer missing at the end")
    if detect_language(question) != detect_language(answer):
        problems.append("answer language does not match the question")
    if re.search(r"\b(guaranteed|guarantee|guarantees|hakikisha)\b|\b100%", answer, re.I):
        problems.append("promises certainty")
    if re.search(r"\b(pesticide|insecticide|fungicide)s?\b", answer, re.I) \
            and not re.search(r"\d+\s*(ml|g|kg|%|litre|liter)\b", answer, re.I) \
            and not re.search(r"\b(product label|manufacturer|label)\b", answer, re.I):
        problems.append("pesticide advice without an application rate")
    return problems


def retry_note(problems) -> str:
    """Turn validation problems into the extra instruction for one retry."""
    notes = []
    if any("citation" in problem for problem in problems):
        notes.append("Cite every claim with [n], using only the source numbers given.")
    if "disclaimer missing at the end" in problems:
        notes.append(f"End with exactly this sentence: {DISCLAIMER}")
    if "pesticide advice without an application rate" in problems:
        notes.append("If you recommend a pesticide, give the application rate from "
                     "the sources; if they give none, tell the farmer to follow the "
                     "product label.")
    if "promises certainty" in problems:
        notes.append("Do not promise certainty and do not use words such as guaranteed.")
    return " ".join(notes) or "Answer more carefully, following every instruction above."


def _sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _login_huggingface() -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        return
    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
    except Exception:
        pass


class _NumpyIndex:
    def __init__(self, vectors: np.ndarray):
        self.vectors = np.asarray(vectors, dtype="float32")

    def search(self, queries: np.ndarray, k: int):
        scores = np.asarray(queries, dtype="float32") @ self.vectors.T
        order = np.argsort(-scores, axis=1)[:, :k]
        return np.take_along_axis(scores, order, 1), order


class AdvisoryPipeline:
    """Retrieval + generation for the maize advisory chat, loaded on demand."""

    def __init__(self, export_dir: Path | str = EXPORT_DIR):
        self.export_dir = Path(export_dir)
        self.device = None
        self.config: dict = {}
        self.chunks: list[dict] = []
        self.bm25 = None
        self._loaded = False
        self._lock = threading.Lock()
        self._trans_tokenizers = {}
        self._ct2 = None
        self._ct2_repo: str | None = None

    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> "AdvisoryPipeline":
        with self._lock:
            if self._loaded:
                return self
            from rank_bm25 import BM25Okapi
            from sentence_transformers import SentenceTransformer

            self.device = _resolve_device()
            self.config = self._load_verified()

            chunks_path = self.export_dir / "chunks.jsonl"
            self.chunks = [json.loads(line) for line in
                           chunks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            vectors = np.load(self.export_dir / "vectors.npy")
            self.index = _NumpyIndex(vectors)
            self._by_id = {chunk["chunk_id"]: chunk for chunk in self.chunks}

            _login_huggingface()

            # Only the retrieval encoder lives on the app server. The answer
            # model is hosted (llm_client) and NLLB is started lazily on the
            # first Swahili/Dholuo question, so an English question never pays
            # for models it does not use.
            self.embedder = SentenceTransformer(self.config["embedding_model"], device=self.device)

            tokens = [re.findall(r"[a-z0-9']+", chunk["text"].lower()) for chunk in self.chunks]
            self.bm25 = BM25Okapi(tokens)

            self.trans_model_name = self.config["translation_model"]

            self._loaded = True
        return self

    def _load_verified(self) -> dict:
        manifest = json.loads((self.export_dir / "manifest.json").read_text(encoding="utf-8"))
        for name, info in manifest.items():
            if _sha256_of(self.export_dir / name) != info["sha256"]:
                raise ValueError(f"export file changed or corrupt: {name}")
        return json.loads((self.export_dir / "config.json").read_text(encoding="utf-8"))

    def embed_query(self, text: str) -> np.ndarray:
        vectors = self.embedder.encode(
            ["query: " + text], normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype="float32")

    def search(self, query: str, k: int = TOP_K):
        scores, row_ids = self.index.search(self.embed_query(query), k)
        return [(float(score), self.chunks[row]) for score, row in zip(scores[0], row_ids[0])]

    def bm25_search(self, query: str, k: int = TOP_K):
        scores = np.asarray(self.bm25.get_scores(re.findall(r"[a-z0-9']+", query.lower())))
        order = scores.argsort()[::-1][:k]
        return [(float(scores[i]), self.chunks[i]) for i in order if scores[i] > 0]

    @staticmethod
    def fuse_ranks(dense_ranks, sparse_ranks, k=TOP_K, dense_weight=0.5, depth=10, rrf_k=10):
        votes = {}
        for rank, chunk_id in enumerate(dense_ranks[:k * depth], start=1):
            votes[chunk_id] = votes.get(chunk_id, 0) + dense_weight / (rrf_k + rank)
        for rank, chunk_id in enumerate(sparse_ranks[:k * depth], start=1):
            votes[chunk_id] = votes.get(chunk_id, 0) + (1 - dense_weight) / (rrf_k + rank)
        ranked = sorted(votes, key=votes.get, reverse=True)[:k]
        return [(chunk_id, votes[chunk_id]) for chunk_id in ranked]

    def _diversify(self, fused, k: int, cap: int):
        counts = {}
        picked = []
        overflow = []
        for chunk_id, score in fused:
            source_id = self._by_id[chunk_id]["source_id"]
            if counts.get(source_id, 0) < cap:
                counts[source_id] = counts.get(source_id, 0) + 1
                picked.append((chunk_id, score))
                if len(picked) == k:
                    return picked
            else:
                overflow.append((chunk_id, score))
        return (picked + overflow)[:k]

    def hybrid_search(self, query: str, k: int = TOP_K, per_source_cap: int | None = None):
        cfg = self.config["hybrid"]
        depth = cfg["depth"]
        pool = k * 4 if per_source_cap else k
        dense_ranks = [chunk["chunk_id"] for _, chunk in self.search(query, pool * depth)]
        sparse_ranks = [chunk["chunk_id"] for _, chunk in self.bm25_search(query, pool * depth)]
        fused = self.fuse_ranks(dense_ranks, sparse_ranks, k=pool,
                                dense_weight=cfg["dense_weight"], depth=depth, rrf_k=cfg["rrf_k"])
        if per_source_cap:
            fused = self._diversify(fused, k, per_source_cap)
        return [(score, self._by_id[chunk_id]) for chunk_id, score in fused[:k]]

    def _merge(self, queries: list[str], k: int, per_source_cap: int | None = None):
        votes = {}
        for query in queries:
            for rank, (_, chunk) in enumerate(self.hybrid_search(query, k, per_source_cap), start=1):
                chunk_id = chunk["chunk_id"]
                votes[chunk_id] = votes.get(chunk_id, 0) + 1 / (30 + rank)
        fused = sorted(votes.items(), key=lambda item: item[1], reverse=True)
        if per_source_cap:
            fused = self._diversify(fused, k, per_source_cap)
        return [(score, self._by_id[chunk_id]) for chunk_id, score in fused[:k]]

    def _trans_tokenizer(self, source_lang: str, target_lang: str):
        """Tokenizer configured for one NLLB direction (cached per pair)."""
        key = (source_lang, target_lang)
        if key not in self._trans_tokenizers:
            from transformers import AutoTokenizer

            self._trans_tokenizers[key] = AutoTokenizer.from_pretrained(
                self.trans_model_name,
                src_lang=NLLB_CODES[source_lang],
                tgt_lang=NLLB_CODES[target_lang],
            )
        return self._trans_tokenizers[key]

    def _translation_backend(self) -> str:
        """Where translation runs: ``google``, ``llm``, ``local`` or ``transformers``."""
        requested = (os.environ.get("ADVISORY_TRANSLATION_BACKEND") or "auto").strip().lower()
        if requested in {"google", "llm", "local", "transformers"}:
            return requested
        if os.environ.get("GOOGLE_TRANSLATE_API_KEY"):
            return "google"
        return "local"

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate between any two configured languages."""
        if source_lang == target_lang:
            return text
        backend = self._translation_backend()
        if backend == "google":
            from llm_client import google_translate

            return google_translate(text, source_lang, target_lang)
        if backend == "llm":
            from llm_client import translate_with_llm

            return translate_with_llm(text, source_lang, target_lang)
        if backend == "transformers":
            return self._hf_translate(text, source_lang, target_lang)
        return self._ct2_translate(text, source_lang, target_lang)

    def _ct2_translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate with NLLB int8, one sentence at a time.

        CTranslate2 keeps the model at about 0.6 GB instead of 2.4 GB.
        Translating sentence by sentence matters: NLLB's decoder sometimes
        stops after the first sentence (or repeats a phrase) when several
        sentences are passed at once, which silently dropped parts of an
        answer. Splitting keeps every sentence, and every ``[n]`` marker, in
        place.
        """
        return self._translate_chunks(
            text, lambda chunk: self._ct2_translate_once(chunk, source_lang, target_lang))

    def _ct2_translate_once(self, text: str, source_lang: str, target_lang: str) -> str:
        translator = self._ct2_model()
        tokenizer = self._trans_tokenizer(source_lang, target_lang)
        tokenizer.src_lang = NLLB_CODES[source_lang]
        target_token = tokenizer.convert_ids_to_tokens(
            tokenizer.convert_tokens_to_ids(NLLB_CODES[target_lang]))
        source_tokens = tokenizer.convert_ids_to_tokens(
            tokenizer.encode(text, truncation=True, max_length=512))
        result = translator.translate_batch(
            [source_tokens],
            target_prefix=[[target_token]],
            beam_size=int(os.environ.get("ADVISORY_TRANSLATION_BEAM", "4")),
            no_repeat_ngram_size=3,
            max_decoding_length=512,
        )
        hypothesis = result[0].hypotheses[0]
        if hypothesis and hypothesis[0] == target_token:
            hypothesis = hypothesis[1:]
        return tokenizer.decode(
            tokenizer.convert_tokens_to_ids(hypothesis),
            skip_special_tokens=True).strip()

    def _ct2_model(self):
        """Load the int8 NLLB translator once and reuse it for the process."""
        import ctranslate2
        from huggingface_hub import snapshot_download

        repo = (os.environ.get("ADVISORY_CT2_TRANSLATION_MODEL")
                or DEFAULT_CT2_TRANSLATION_MODEL)
        if self._ct2 is None or self._ct2_repo != repo:
            path = snapshot_download(repo)
            threads = max(1, min(2, os.cpu_count() or 1))
            self._ct2 = ctranslate2.Translator(
                path, device="cpu", compute_type="int8",
                inter_threads=1, intra_threads=threads)
            self._ct2_repo = repo
        return self._ct2

    _CITATION_TAIL = re.compile(r"\s*(?:\[\d+\])+(?P<punct>[.!?]?)\s*$")

    @classmethod
    def _translate_chunks(cls, text: str, translate_one) -> str:
        """Run translation per sentence/line, preserving citations and order."""
        pieces = [piece.strip() for piece in
                  re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if piece.strip()]
        chunks: list[str] = []
        for piece in pieces:
            # "1." / "a." on its own is a list marker, not a sentence: keep it
            # attached to the text that follows so NLLB never sees a fragment.
            if chunks and re.fullmatch(r"\d+[.)]|[a-zA-Z][.)]", chunks[-1]):
                chunks[-1] = chunks[-1] + " " + piece
            else:
                chunks.append(piece)
        if not chunks:
            return ""
        rendered = []
        for chunk in chunks:
            marker = re.match(r"^([-*•]|\d+[.)])\s+(.*)$", chunk, re.S)
            prefix, body = (marker.group(1) + " ", marker.group(2)) if marker else ("", chunk)
            # NLLB sometimes rewrites a bracketed number ("[5]" -> "[16]").
            # Hold the citation outside the decoder and put it back untouched.
            citation = cls._CITATION_TAIL.search(body)
            if citation and re.search(r"[A-Za-z0-9]", body[:citation.start()]):
                tail = citation.group(0).strip()
                rendered.append(prefix + translate_one(body[:citation.start()]).strip()
                                + " " + tail)
            elif re.fullmatch(r"\[\d+\][.!?]?", body):
                rendered.append(prefix + body)  # a bare citation: leave it alone
            else:
                rendered.append(prefix + translate_one(body))
        return " ".join(rendered)

    def _hf_translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Full-precision NLLB through transformers (local development only)."""
        return self._translate_chunks(
            text, lambda chunk: self._hf_translate_once(chunk, source_lang, target_lang))

    def _hf_translate_once(self, text: str, source_lang: str, target_lang: str) -> str:
        if not hasattr(self, "trans_model"):
            import torch
            from transformers import AutoModelForSeq2SeqLM

            device = self.device or _resolve_device()
            self.trans_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.trans_model_name).to(device)
        tokenizer = self._trans_tokenizer(source_lang, target_lang)
        tokens = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=512).to(self.trans_model.device)
        output = self.trans_model.generate(
            **tokens, max_length=512,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(NLLB_CODES[target_lang]))
        return tokenizer.decode(output[0], skip_special_tokens=True)

    def translate_to_english(self, text: str, source_lang: str | None = None) -> str:
        source_lang = source_lang or detect_language(text)
        if source_lang == "en":
            return text
        english = self.translate(text, source_lang, "en")
        return re.sub(r"\bcorn\b", "maize", english, flags=re.IGNORECASE)

    def translate_to_swahili(self, text: str) -> str:
        return self.translate(text, "en", "sw")

    def translate_to_dholuo(self, text: str) -> str:
        return self.translate(text, "en", "luo")

    def translate_to_language(self, text: str, target_lang: str,
                              source_lang: str | None = None) -> str:
        source_lang = source_lang or detect_language(text)
        return self.translate(text, source_lang, target_lang)

    def retrieve(self, query: str, k: int = TOP_K, context: list[str] | None = None,
                 per_source_cap: int | None = None):
        source_lang = detect_language(query)
        routed = self.translate_to_english(query, source_lang) if source_lang != "en" else query
        search_query = routed
        if context:
            recent = []
            for previous in context[-2:]:
                previous_lang = detect_language(previous)
                if previous_lang != "en":
                    previous = self.translate_to_english(previous, previous_lang)
                recent.append(previous)
            search_query = " ".join(recent + [routed])
        return routed, self.hybrid_search(search_query, k, per_source_cap)

    def _context_turns(self, question: str, history: list[dict]) -> list[str]:
        if not is_follow_up(question):
            return []
        for turn in reversed(history):
            if turn.get("role") != "user":
                continue
            text = turn.get("content", "")
            if refuse(text, has_context=True) or not mentions_topic(text):
                continue
            return [text]
        return []

    def build_prompt(self, question: str, hits, history=None) -> str:
        context = "\n\n".join(
            f"[{i}] {chunk['title']} ({chunk['source_id']}, page {chunk['page']}):\n{chunk['text']}"
            for i, (_, chunk) in enumerate(hits, start=1)
        )
        history_text = ""
        if history:
            lines = []
            for turn in history[-6:]:
                content = turn["content"].strip()
                if turn["role"] == "assistant" and content in REFUSAL_TEXTS:
                    continue
                lines.append(f"{'User' if turn['role'] == 'user' else 'Assistant'}: {content}")
            if lines:
                history_text = "\nConversation so far (for context only):\n" + "\n".join(lines) + "\n"
        return PROMPT_TEMPLATE.format(context=context, question=question, history=history_text)

    def generate(self, prompt: str, max_new_tokens: int = 400) -> str:
        """Write the answer with the configured hosted model."""
        from llm_client import chat

        return chat(prompt, max_tokens=max_new_tokens)

    @staticmethod
    def _citations(hits) -> list[dict]:
        return [
            {"n": i, "chunk_id": chunk["chunk_id"], "title": chunk["title"],
             "source_id": chunk["source_id"], "page": chunk["page"],
             "snippet": chunk["text"][:280].strip()}
            for i, (_, chunk) in enumerate(hits, start=1)
        ]

    def ask(self, question: str, k: int = TOP_K, history: list[dict] | None = None) -> dict:
        if not self._loaded:
            self.load()

        history = history or []
        language = detect_language(question)
        context = self._context_turns(question, history)

        refusal = refuse(question, has_context=bool(history))
        if refusal:
            return {"question": question, "answer": refusal, "sources": [],
                    "citations": [], "english_query": None, "refused": True, "problems": [],
                    "translated": False, "language": language}

        list_question = bool(LIST_INTENT.search(question))
        k_use = min(k * 3, 15) if list_question else k
        cap = 2 if list_question else None
        max_tokens = 700 if list_question else 400

        if list_question:
            noun_match = re.search(LIST_NOUN, question, re.I)
            noun = noun_match.group(1).lower() if noun_match else "diseases"
            noun = LIST_NOUN_EN.get(noun, noun)
            english_query = (self.translate_to_english(question, language)
                             if language != "en" else question)
            probe = f"maize {noun} list identification symptoms"
            hits = self._merge([english_query, probe], k_use, cap)
        else:
            english_query, hits = self.retrieve(question, k_use, context=context, per_source_cap=cap)

        prompt = self.build_prompt(english_query, hits, history)
        answer = normalize_answer(self.generate(prompt, max_new_tokens=max_tokens))
        problems = validate_answer(answer, question, len(hits))

        # A non-English answer that arrives in English is translated below, so
        # do not spend a second hosted call retrying it. Every other problem
        # (missing citation, missing disclaimer, ...) is worth one retry.
        retryable = [p for p in problems
                     if language == "en" or p != "answer language does not match the question"]
        if retryable:
            retry_prompt = prompt + "\n\n" + retry_note(retryable)
            answer = normalize_answer(self.generate(retry_prompt, max_new_tokens=max_tokens))
            problems = validate_answer(answer, question, len(hits))

        if language != "en" and detect_language(answer) != language:
            # The generator may answer in English even when the template asks
            # for the question's language. Translate from whatever language it
            # actually used, not only English: a Swahili answer to a Dholuo
            # question is possible.
            answer_lang = detect_language(answer)
            body = answer.split(DISCLAIMER)[0].strip()
            answer = self.translate_to_language(
                body, language, source_lang=answer_lang).strip() + " " + DISCLAIMER
            problems = validate_answer(answer, question, len(hits))

        return {
            "question": question,
            "answer": answer,
            "sources": [chunk["chunk_id"] for _, chunk in hits],
            "citations": self._citations(hits),
            "english_query": english_query,
            "refused": False,
            "problems": problems,
            "translated": english_query != question,
            "language": language,
        }
