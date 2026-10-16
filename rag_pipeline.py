"""
Kenya Maize Advisory - reusable RAG pipeline
=============================================

Loads the artifacts exported by ``RAG_Advisory_Chatbot.ipynb`` section 11
(``rag_artifacts/export/``) and answers farmer questions by retrieving evidence
from the corpus and generating a grounded, cited reply.

The Streamlit app imports this module and calls :meth:`AdvisoryPipeline.ask`.
Everything is local and free: ``multilingual-e5-small`` embeds queries,
``nllb-200-distilled-600M`` pivots Swahili to English and back, and
``Qwen2.5-1.5B-Instruct`` writes the answer. The heavy models are loaded
lazily on first use, so importing this module stays cheap.
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

TOPIC_WORDS = [
    "maize", "corn", "crop", "farm", "field", "soil", "rain", "seed", "fertilis",
    "fertiliz", "manure", "pest", "disease", "insect", "weed", "harvest", "yield",
    "plant", "variety", "varieties", "irrigation", "storage", "grain", "cob",
    "leaf", "blight", "rust", "hectare", "acre", "agronom", "extension",
    "mahindi", "mbolea", "shamba", "kilimo", "wakulima", "mazao", "ugonjwa",
    "magonjwa", "wadudu", "mbegu", "udongo", "mvua", "kuvuna", "mavuno",
    "kupanda", "dalili", "tiba", "kuzuia", "ipil",
]

REFERENTIAL = re.compile(
    r"\b(it|its|they|them|their|that|this|these|those|the same|the above"
    r"|hii|hiyo|hizi|hizo|huo|hayo|hao|wao|yake|wake|zao)\b", re.I
)

LIST_TRIGGER = r"(list|enumerate|name|what are|which are|all|aina|gani|yapi|ipi|zipi)"
LIST_NOUN = (r"(diseases?|pests?|insects?|weeds?|varieties?|varieti|symptoms?"
             r"|problems?|magonjwa|wadudu|dalili)")
LIST_NOUN_EN = {"magonjwa": "diseases", "wadudu": "pests", "dalili": "symptoms",
                "varieti": "varieties"}

LIST_INTENT = re.compile(
    r"\b" + LIST_TRIGGER + r"\b.{0,60}?\b" + LIST_NOUN + r"\b"
    r"|\b" + LIST_NOUN + r"\b.{0,60}?\b" + LIST_TRIGGER + r"\b",
    re.I | re.S
)

REFUSAL_RULES = [
    ("live_prices", [r"\bprices?\b", r"how much is", r"\bbei\b", r"ruzuku",
                     r"subsid", r"market rate"]),
    ("human_medicine", [r"malaria", r"\bhuman\b", r"binadamu", r"\bdawa\b",
                        r"medicine"]),
]

REFUSALS = {
    "live_prices": {
        "en": "I do not have live market prices or subsidy information - my sources are "
              "static documents. Check today's prices with your local produce buyer, the "
              "county market office, or the e-Mkulima platform.",
        "sw": "Sina bei za soko za sasa wala taarifa za ruzuku - vyanzo vyangu ni vitabu "
              "vya zamani. Uliza mnunuzi wako wa mazao, ofisi ya soko ya kaunti, au tumia "
              "jukwaa la e-Mkulima.",
    },
    "human_medicine": {
        "en": "I only advise on growing maize. Questions about human health should go to "
              "a clinic or hospital.",
        "sw": "Nashauri tu kuhusu ukuzaji wa mahindi. Maswali kuhusu afya ya binadamu "
              "yawaoneshe kwa daktari.",
    },
    "off_topic": {
        "en": "I only answer questions about growing maize in Kenya - diseases, planting, "
              "fertiliser, pests, harvest and storage. Ask me about maize!",
        "sw": "Najibu maswali tu kuhusu ukuzaji wa mahindi nchini Kenya - magonjwa, "
              "kupanda, mbolea, wadudu, mavuno na uhifadhi. Niulize kuhusu mahindi!",
    },
}

REFUSAL_TEXTS = {text for category in REFUSALS.values() for text in category.values()}

PROMPT_TEMPLATE = """You are the Kenya Maize Advisory Chatbot, an assistant for smallholder maize farmers.

Answer ONLY from the numbered sources below.
- Cite sources like [1] or [2][3] after every claim.
- If the sources do not contain the answer, say so and advise consulting a county extension officer.
- Reply in the SAME language the question was asked in (English or Kiswahili).
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


def looks_swahili(text: str) -> bool:
    words = set(re.findall(r"[a-zA-Z]+", text.lower()))
    return len(words & SW_STOP) >= 2


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
    lang = "sw" if looks_swahili(question) else "en"
    for category, patterns in REFUSAL_RULES:
        if any(re.search(pattern, text) for pattern in patterns):
            return REFUSALS[category][lang]
    if not mentions_topic(text) and not (has_context and is_follow_up(question)):
        return REFUSALS["off_topic"][lang]
    return None


def validate_answer(answer: str, question: str, n_sources: int):
    problems = []
    citations = [int(n) for n in re.findall(r"\[(\d+)\]", answer)]
    if not citations:
        problems.append("no citation")
    if any(n > n_sources for n in citations):
        problems.append("citation points outside the given sources")
    if not answer.strip().endswith(DISCLAIMER):
        problems.append("disclaimer missing at the end")
    if looks_swahili(question) != looks_swahili(answer):
        problems.append("answer language does not match the question")
    if re.search(r"\b(guaranteed|guarantee|guarantees|hakikisha)\b|\b100%", answer, re.I):
        problems.append("promises certainty")
    if re.search(r"\b(pesticide|insecticide|fungicide)s?\b", answer, re.I) \
            and not re.search(r"\d+\s*(ml|g|kg|%|litre|liter)\b", answer, re.I):
        problems.append("pesticide advice without an application rate")
    return problems


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

    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> "AdvisoryPipeline":
        with self._lock:
            if self._loaded:
                return self
            from rank_bm25 import BM25Okapi
            from sentence_transformers import SentenceTransformer
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForSeq2SeqLM,
                AutoTokenizer,
            )
            import torch

            self.device = _resolve_device()
            self.config = self._load_verified()

            chunks_path = self.export_dir / "chunks.jsonl"
            self.chunks = [json.loads(line) for line in
                           chunks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            vectors = np.load(self.export_dir / "vectors.npy")
            self.index = _NumpyIndex(vectors)
            self._by_id = {chunk["chunk_id"]: chunk for chunk in self.chunks}

            _login_huggingface()

            self.embedder = SentenceTransformer(self.config["embedding_model"], device=self.device)

            tokens = [re.findall(r"[a-z0-9']+", chunk["text"].lower()) for chunk in self.chunks]
            self.bm25 = BM25Okapi(tokens)

            self.trans_model_name = self.config["translation_model"]
            self.sw_tokenizer = AutoTokenizer.from_pretrained(
                self.trans_model_name, src_lang="swh_Latn", tgt_lang="eng_Latn")
            self.en_tokenizer = AutoTokenizer.from_pretrained(
                self.trans_model_name, src_lang="eng_Latn", tgt_lang="swh_Latn")
            self.trans_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.trans_model_name).to(self.device)

            self.gen_model_name = self.config["generation_model"]
            self.gen_tokenizer = AutoTokenizer.from_pretrained(self.gen_model_name)
            self.gen_model = AutoModelForCausalLM.from_pretrained(
                self.gen_model_name, dtype=torch.float16).to(self.device)
            self.gen_model.eval()

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

    def translate_to_english(self, text: str) -> str:
        tokens = self.sw_tokenizer(text, return_tensors="pt", truncation=True,
                                   max_length=128).to(self.trans_model.device)
        output = self.trans_model.generate(
            **tokens, max_length=200,
            forced_bos_token_id=self.sw_tokenizer.convert_tokens_to_ids("eng_Latn"))
        english = self.sw_tokenizer.decode(output[0], skip_special_tokens=True)
        return re.sub(r"\bcorn\b", "maize", english, flags=re.IGNORECASE)

    def translate_to_swahili(self, text: str) -> str:
        tokens = self.en_tokenizer(text, return_tensors="pt", truncation=True,
                                   max_length=512).to(self.trans_model.device)
        output = self.trans_model.generate(
            **tokens, max_length=512,
            forced_bos_token_id=self.en_tokenizer.convert_tokens_to_ids("swh_Latn"))
        return self.en_tokenizer.decode(output[0], skip_special_tokens=True)

    def retrieve(self, query: str, k: int = TOP_K, context: list[str] | None = None,
                 per_source_cap: int | None = None):
        routed = self.translate_to_english(query) if looks_swahili(query) else query
        search_query = routed
        if context:
            recent = []
            for previous in context[-2:]:
                if looks_swahili(previous):
                    previous = self.translate_to_english(previous)
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
        messages = [{"role": "user", "content": prompt}]
        text = self.gen_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.gen_tokenizer(text, return_tensors="pt").to(self.gen_model.device)
        output = self.gen_model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return self.gen_tokenizer.decode(new_tokens, skip_special_tokens=True)

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
        context = self._context_turns(question, history)

        refusal = refuse(question, has_context=bool(history))
        if refusal:
            return {"question": question, "answer": refusal, "sources": [],
                    "citations": [], "english_query": None, "refused": True, "problems": [],
                    "translated": False, "language": "sw" if looks_swahili(question) else "en"}

        list_question = bool(LIST_INTENT.search(question))
        k_use = min(k * 3, 15) if list_question else k
        cap = 2 if list_question else None
        max_tokens = 700 if list_question else 400

        if list_question:
            noun_match = re.search(LIST_NOUN, question, re.I)
            noun = noun_match.group(1).lower() if noun_match else "diseases"
            noun = LIST_NOUN_EN.get(noun, noun)
            english_query = self.translate_to_english(question) if looks_swahili(question) else question
            probe = f"maize {noun} list identification symptoms"
            hits = self._merge([english_query, probe], k_use, cap)
        else:
            english_query, hits = self.retrieve(question, k_use, context=context, per_source_cap=cap)

        prompt = self.build_prompt(english_query, hits, history)
        answer = self.generate(prompt, max_new_tokens=max_tokens)
        problems = validate_answer(answer, question, len(hits))

        if problems:
            retry_prompt = prompt + ("\n\nCite every claim with [n] and end with the exact "
                                     "disclaimer sentence.")
            answer = self.generate(retry_prompt, max_new_tokens=max_tokens)
            problems = validate_answer(answer, question, len(hits))

        if looks_swahili(question) and not looks_swahili(answer):
            body = answer.split(DISCLAIMER)[0].strip()
            answer = self.translate_to_swahili(body).strip() + " " + DISCLAIMER
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
            "language": "sw" if looks_swahili(question) else "en",
        }
