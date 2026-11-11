# Maize Leaf Disease Classification & Kenya Maize Advisory Chatbot

## Summary of the Project

Two projects in one repository:

1. **Image classification** — classify a maize leaf photo into **Blight**, **Common Rust**, **Gray Leaf Spot** or **Healthy**, comparing a pretrained YOLO26 classifier against a custom CNN built from scratch in Keras.
2. **Advisory chatbot (RAG)** — an English/Kiswahili/Dholuo retrieval-augmented chatbot that answers questions about maize farming in Kenya from a curated PDF knowledge base, integrated as a third tab in the same Streamlit app. Farmer profiles and full chat history are stored in **MySQL**.

A farmer can photograph a leaf, get an instant diagnosis, then continue the conversation for management advice — in English, Kiswahili or Dholuo. Conversations are saved so they can be revisited and analysed later.

## Problem Statement

Maize foliar diseases spread rapidly and can cut yields by 30–50%. The current diagnosis method is visual inspection by a human agronomist: slow, subjective, error-prone, and impossible to scale across large plantations. Many smallholder farmers have no access to a plant pathologist, so disease is caught only after irreversible damage. Even when a disease is identified, reliable agronomy advice — fertiliser, planting, storage, pest control — remains hard to access, especially in local languages.

## Project Objective

- Classify a maize leaf image into one of four classes so a farmer can photograph a leaf and get an instant diagnosis.
- Build and evaluate a free, grounded RAG chatbot that provides source-cited answers on Kenyan maize farming in English, Kiswahili and Dholuo, with local retrieval and a hosted answer model so it fits a free web host.

## Research Questions

**Image classification:**

1. How accurately can a pretrained YOLO26 classifier separate these four classes, and how much of that is transfer learning rather than training from scratch?
2. How close does a small custom CNN get, and is it small and fast enough to be worth deploying instead?
3. Which classes get confused with each other, and why — background clutter, lighting, or the visual overlap between Blight and Gray Leaf Spot?
4. How do model size, augmentation, and training time trade off against accuracy on a dataset this small?
5. Which pixels actually drive the final model's decision, and is it reading the lesion or the background?

**Advisory chatbot (RAG):**

6. How well does hybrid retrieval (BM25 + dense vectors) serve Kenyan maize farming queries compared with dense-only retrieval?
7. Can cross-lingual embeddings retrieve English corpus content from Swahili queries, and does the translation pivot extend that to Dholuo?
8. Does the translation-pivot architecture (Swahili/Dholuo → English → Swahili/Dholuo) produce answers comparable in quality to native English answers?
9. How do chunk size, overlap and top-k affect retrieval accuracy and answer faithfulness?
10. How reliably do guardrails keep answers on-topic, grounded in sources, and free of stale market/subsidy claims?

## Data Sources

**Image classification dataset**

Source: [Kaggle — Corn or Maize Leaf Disease Dataset](https://www.kaggle.com/datasets/smaranjitghouse/corn-or-maize-leaf-disease-dataset) — 4,186 usable images (4,188 files minus 2 cross-class duplicates), split 70/15/15 into train (3,664, classes levelled to 916 each), val (625) and test (625).

| Class | Images | Share |
|---|---|---|
| Common_Rust | 1,306 | 31.2% |
| Healthy | 1,162 | 27.8% |
| Blight | 1,146 | 27.4% |
| Gray_Leaf_Spot | 572 | 13.7% |
| **Total** | **4,186** | **100%** |

**RAG knowledge base** (`pdfs/`)

| Document | Content |
|---|---|
| `MAIZE GROWERS GUIDE.pdf` | Practical maize agronomy handbook — planting, fertiliser, field management |
| `National-Agriculture-Production-Report-2025.pdf` | National production statistics and sector context |
| `Planting_strategies_of_maize_farmers_in_Kenya_a_si.pdf` | Research article: planting strategies of Kenyan maize farmers |
| `Olwande_Smallholder_Maize_Efficiency_Kenya.pdf` | Study on smallholder maize efficiency in Kenya |
| `diversity in maize production environments and practices.pdf` | Maize production environments and on-farm practices |
| `CIMMYT_Maize_Diseases_Field_Identification_4thEd.pdf` | CIMMYT field guide to identifying maize diseases (fungal, bacterial, viral) |
| `CIMMYT_MLN_Technical_Manual_Disease_Management.pdf` | CIMMYT technical manual on maize lethal necrosis (MLN) management |
| `KALRO_MLND_Kenya_Training_Manual.pdf` | Kenya extension training manual on maize lethal necrosis disease (MLND) |
| `KEPHIS_National_Crop_Variety_List_2025.pdf` | Official maize variety register — altitude range, maturity, yield, disease tolerance |
| `MoA_Sustainable_Agricultural_Livelihood_Restoration_Training_Manual.pdf` | Conservation agriculture, land preparation, ISFM, problematic soils |
| `KALRO_Soil_Fertility_Soil_Testing_Overview.pdf` | Soil fertility constraints and fertilizer recommendations |
| `IFDC_Kenya_Central_Fertilizer_Use_Optimizer_DAP_CAN.pdf` | Per-acre DAP/CAN fertilizer rates |
| `IFDC_Kenya_Lower_Western_Fertilizer_Use_Optimizer_2023.pdf` | Per-acre DAP/Urea fertilizer rates |
| `CABI_Fall_Armyworm_Technical_Brief_Kenya_2018.pdf` | Fall armyworm identification, scouting and control |
| `CIMMYT_Fall_Armyworm_IPM_Strategic_Plan_Kenya.pdf` | Fall armyworm integrated pest management strategic plan |
| `FAO_Integrated_Management_Fall_Armyworm_Maize_FFS.pdf` | Fall armyworm management for farmer field schools |
| `IITA_Aflatoxin_Management_Maize_Groundnuts_Kenya.pdf` | Aflatoxin prevention including Aflasafe KE01 |
| `KALRO_Aflasafe_KE01_Brief.pdf` | Aflasafe KE01 biocontrol brief |
| `FAO_Maize_Post_Harvest_Compendium.pdf` | Maize drying, moisture and storage |
| `TAAT_PICS_Hermetic_Bags_Grain_Storage_Guide.pdf` | Hermetic (PICS) bag grain storage |
| `CIMMYT_Maize_Storage_Pest_Control_Metal_Silo.pdf` | Metal silo storage pest control |
| `FAO_MAFAP_Kenya_Maize_Policy_Note.pdf` | Maize pricing and input subsidy policy |
| `NCPB_Act_Cap338.pdf` | NCPB grading, pricing and buying regulations |
| `NCPB_Restructuring_Report_2014_Cost_Of_Production.pdf` | Cost of maize production breakdown |
| `DISEASE_INFO` (from `app.py`) | Seed knowledge for the four leaf-disease classes |

**MySQL database** (`maize_advisory`)

| Table | Key fields | Purpose |
|---|---|---|
| `farmers` | `id, full_name, email, password_hash, county, preferred_lang, profile_image, created_at` | Farmer accounts. `password_hash` is a salted PBKDF2-HMAC-SHA256 digest; `profile_image` is an optional path under `profile_images/` |
| `chat_sessions` | `id, farmer_id, started_at, language` | One conversation thread |
| `chat_messages` | `id, session_id, farmer_id, role, message, language, sources, feedback, created_at` | Full chat history, retrieved citations, feedback |
| `login_tokens` | `token, farmer_id, last_seen` | Opaque tokens that remember a login across reloads; expire after 30 minutes of inactivity (`LOGIN_TTL_MINUTES`) |

## Project Structure

```
maize_disease/
├── data/                              # source images, one folder per class
│   ├── Blight/
│   ├── Common_Rust/
│   ├── Gray_Leaf_Spot/
│   └── Healthy/
├── models/
│   ├── yolo26n-cls.pt                 # Model A baseline (pretrained, not committed)
│   ├── yolo26s-cls.pt                 # larger variant (not committed)
│   └── maize_disease_yolo26s.pt       # final trained model - TRACKED IN GIT, the app needs it
├── pdfs/                             # RAG knowledge base (source PDFs)
├── Maize_Disease_Detection_YOLO.ipynb # image classification pipeline (sections 0-11)
├── RAG_Advisory_Chatbot.ipynb         # RAG chatbot pipeline (13 sections)
├── rag_pipeline.py                    # chatbot inference module, imported by app.py
├── llm_client.py                      # hosted answer model (Groq / Gemini / HF), imported by rag_pipeline.py
├── voice.py                           # speech-to-text and text-to-speech, imported by app.py
├── db.py                              # MySQL chat-history functions, imported by app.py
├── rag_data/                          # raw corpus + chunks.jsonl
├── rag_artifacts/                     # vector index, config, evaluation, charts (figures/) + export bundle (export/)
├── profile_images/                    # uploaded profile pictures (git-ignored)
├── app.py                             # single-file Streamlit app (login, detection, advisory chat, metrics)
├── .env                               # local MySQL credentials (git-ignored)
├── .env.example                       # template for .env
├── .streamlit/config.toml             # headless server + dark theme for deployment
├── packages.txt                       # apt libs Streamlit Cloud needs for OpenCV
├── requirements.txt
├── README.md
└── model_comparison.md                # final classification report, written by section 11.3
```

## Running the app

Use the project virtualenv. The app needs packages (`torch`, `transformers`, `mysql-connector-python`, ...) that are **not** in the system Python, so run `streamlit` from inside `.venv` - otherwise you get `ModuleNotFoundError: No module named 'mysql'`.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

On macOS the quickest repeatable command is just `.venv/bin/streamlit run app.py`. If you have activated the venv (`source .venv/bin/activate`), plain `streamlit run app.py` uses the venv too.

The app is behind a login, so a database is **required**: farmers create an account (name, email, password, optional profile picture) and every conversation is saved against them. A login survives a page reload for 30 minutes of inactivity (an opaque token in the URL, validated against `login_tokens`), and the sidebar lists saved conversations so any chat can be reopened and continued.

1. **RAG export** — the advisory tab loads `rag_artifacts/export/` (produced by `RAG_Advisory_Chatbot.ipynb` section 12). Run that notebook once if the folder is missing.
2. **Database** — copy `.env.example` to `.env` and set the `MYSQL_*` values. Create the database and a dedicated user once:

   ```sql
   CREATE DATABASE IF NOT EXISTS maize_advisory CHARACTER SET utf8mb4;
   CREATE USER IF NOT EXISTS 'maize'@'localhost' IDENTIFIED BY 'maize_advisory';
   GRANT ALL PRIVILEGES ON maize_advisory.* TO 'maize'@'localhost';
   ```

   Or, for a local Homebrew MySQL whose `root` has no password, set `MYSQL_USER=root` and leave `MYSQL_PASSWORD` empty instead. The app creates the tables itself on first use.

Passwords are stored only as salted PBKDF2-HMAC-SHA256 hashes (Python standard library). The first advisory question loads the retrieval index and `multilingual-e5-small` (~0.5 GB, a few seconds). Swahili and Dholuo questions additionally download and start the int8 NLLB translation model (~0.6 GB) the first time they are asked. Answers are written by a hosted model, so the app server never loads a multi-gigabyte generator.

### Where the models run

| Job | Runs on | Footprint |
|---|---|---|
| Query embedding + retrieval | App server (local) | `multilingual-e5-small`, ~0.5 GB |
| Swahili/Dholuo pivot | App server (local, lazy) | NLLB-200 distilled 600M, int8 CTranslate2, ~0.6 GB |
| Answer writing | Hosted API | none on the server |

The split exists because Streamlit Community Cloud gives an app about **2.7 GB of RAM and 2 CPU cores**. A local `Qwen2.5-1.5B-Instruct` alone takes ~3 GB in fp16 and takes minutes to write one answer on two cores; adding NLLB's 2.5 GB made the app exceed the limit and get killed the moment someone asked a question. See [Hosted answer model](#hosted-answer-model) for configuration.

### Advisory languages

The advisory chat accepts **English**, **Kiswahili** and **Dholuo**. The corpus is English, so the pipeline translates Swahili (`swh_Latn`) and Dholuo (`luo_Latn`) questions into English with `nllb-200-distilled-600M`, retrieves from the English index, then translates the answer back into the question's language. The default translation model is an int8 CTranslate2 conversion (~0.6 GB); set `ADVISORY_TRANSLATION_BACKEND=google` with a `GOOGLE_TRANSLATE_API_KEY` to move translation off the server entirely, or `=llm` to let the hosted chat model translate. Refusals and guardrails are localised too. Dholuo is a low-resource language for NLLB, so its answers should be treated as a helpful draft rather than a polished translation; the 12-question evaluation set in the notebook is still English/Swahili, so Dholuo quality is not yet benchmarked.

### Hosted answer model

Answer generation is the heavy, slow part, so it is the part that runs off the app server. `llm_client.py` talks to an OpenAI-compatible chat endpoint and supports three providers:

| Provider | Free tier | Key |
|---|---|---|
| **Groq** (default) | Fast, generous, no card | `GROQ_API_KEY` from <https://console.groq.com/keys> |
| **Google Gemini** | Generous daily quota | `GEMINI_API_KEY` from <https://aistudio.google.com/apikey> |
| Hugging Face Inference Providers | Needs pre-paid credits, even on free accounts | `HF_TOKEN`, plus `ADVISORY_LLM_PROVIDER=hf` |

`ADVISORY_LLM_PROVIDER=auto` picks Groq, then Gemini. Hugging Face is opt-in because a plain `HF_TOKEN` is also used to download public models, and that path fails without inference credits. Override the model with `GROQ_MODEL` (default `openai/gpt-oss-120b`), `GEMINI_MODEL` (default `gemini-2.5-flash`) or `ADVISORY_HF_MODEL`. The advisory tab shows which provider is active, and a missing key produces a setup message instead of a stack trace.

The notebook still trains and evaluates with a local generator; the hosted path only changes where the same grounded prompt is answered in the deployed app.

## Voice input and output

The advisory tab can be used by voice. Both directions run locally and free - no API key, no per-request cost - in `voice.py`.

| Direction | Model | Size | Notes |
|---|---|---|---|
| Speech → text | `faster-whisper` `small`, int8, CPU | ~500 MB | Multilingual, so one model covers English and Kiswahili. Language is auto-detected and shown to the farmer. Whisper has no Dholuo token, so Dholuo is typed rather than spoken. |
| Text → speech | `facebook/mms-tts-eng`, `facebook/mms-tts-swh` and `rnjema-unima/mms-tts-luo-baseline` | ~145 MB each | MMS VITS voices. Kiswahili and Dholuo are native voices, not an English model reading the text. Meta did not publish an official Dholuo MMS voice, so the default is a community checkpoint; override it with `VOICE_TTS_LUO`. |

The Dholuo voice is a community checkpoint (CC BY-NC 4.0); check its model card before any commercial deployment.

**Asking by voice.** The mic in the chat bar records, Whisper transcribes, and the transcript appears in an editable box. Nothing is sent until the farmer presses *Send question*. That confirmation step is deliberate: a mis-heard word changes which documents are retrieved, so correcting the transcript is cheaper than re-asking. Whisper is also given a domain prompt (`VOICE_ASR_PROMPT`) with words like *mbolea*, *DAP*, *ekari*, *oduma* and *puodho*, which noticeably reduces domain-term errors.

**Answers read aloud.** Every assistant message has a *Listen* button, and *Read new answers aloud* speaks each new answer automatically. `speakable()` strips `[1]` citation markers and markdown, expands units (`kg/acre` → "kilograms per acre"), and spells numbers, because the VITS voices read bare digits unreliably - especially in Kiswahili and Dholuo. The closing disclaimer is always read with the English voice, since `rag_pipeline` appends it verbatim in every language.

**Secure context.** The microphone, like the camera, only works on `localhost` or HTTPS. A phone reaching the app over plain `http://<lan-ip>:8501` gets no mic button.

**Memory.** Voice adds roughly 1 GB on top of the advisory tab. With hosted generation the normal footprint is about 1.1 GB (torch + retrieval encoder), rising to ~1.7 GB for Swahili/Dholuo (int8 NLLB) and ~2.7 GB only if a voice model and NLLB are both resident. `VOICE_ASR_MODEL=base` trims the speech model if that combination is too close to the cap.

Optional overrides live in `.env.example`: `VOICE_ASR_MODEL`, `VOICE_ASR_PROMPT`, `VOICE_TTS_EN`, `VOICE_TTS_SW`, `VOICE_TTS_LUO`.

## Deploying to Streamlit Community Cloud

The repository is already wired for it (`.streamlit/config.toml`, `packages.txt`). Two things are deliberately **not** in git and have to be entered in the dashboard.

1. **Secrets.** Open *App settings -> Secrets* and paste the same keys as `.env`:

   ```toml
   MYSQL_HOST = "mysql-xxxx.aivencloud.com"
   MYSQL_PORT = "12345"
   MYSQL_DATABASE = "maize_advisory"
   MYSQL_USER = "avnadmin"
   MYSQL_PASSWORD = "..."
   HF_TOKEN = "hf_..."

   # Answer generation (free; see "Hosted answer model" above)
   GROQ_API_KEY = "gsk_..."
   # or: GEMINI_API_KEY = "..."
   ```

   Community Cloud keeps secrets in `st.secrets` and does **not** export them as environment variables, so `db.py` and `llm_client.py` copy them into `os.environ` at import time (`_hydrate_from_streamlit_secrets`). A `[mysql]` or `[advisory]` section works too. Save, then reboot the app.

   Hosted providers that require TLS (Aiven, TiDB Cloud, most managed MySQL) also need their CA certificate:

   ```toml
   MYSQL_SSL_CA = """-----BEGIN CERTIFICATE-----
   ...
   -----END CERTIFICATE-----"""
   ```

   `MYSQL_SSL_CA` accepts the PEM text or a path; `MYSQL_SSL_DISABLED = "1"` turns TLS off for servers that do not offer it.

2. **A reachable database.** A local MySQL (`MYSQL_HOST=localhost`) is not reachable from Cloud - that address is the app's own container, so the app fails with `2003: Can't connect to MySQL server on 'localhost:3306'`. Use a hosted instance (Aiven, TiDB Cloud, Railway, a VPS) that accepts remote connections, and allow Streamlit's egress (or `0.0.0.0/0` while testing). Accounts created against your local database do not exist in the cloud one, so sign up again on the deployed app.

**Still seeing "MySQL at localhost is not reachable" after deployment?** The deployed app is reading `MYSQL_HOST=localhost` from its **Secrets**, not from your local `.env` (which is git-ignored and never uploaded). Open the app dashboard -> *Settings -> Secrets*, replace `MYSQL_HOST` with the hosted database host, save, and click **Reboot app**. A plain page refresh or *Rerun* is not enough: secrets are only re-read when the app process restarts.

**Memory.** Community Cloud caps an app at roughly 2.7 GB of RAM and 2 CPU cores. The classifier (`ultralytics` + torch) needs about 0.6-0.9 GB. The advisory tab adds `multilingual-e5-small` (~0.5 GB) and, only for non-English questions, int8 NLLB (~0.6 GB). That keeps the common path around 1.1 GB and the Swahili/Dholuo path around 1.7 GB, both comfortably inside the cap. Answers are written by a hosted model (see [Hosted answer model](#hosted-answer-model)), which also removes the multi-minute CPU wait a local 1.5B generator caused. If the app is still killed, check the Cloud logs: a `Killed` line means memory, and the voice models (Whisper ~0.5 GB plus a TTS voice) are the next-largest consumers.

**Why not generate locally?** A local `Qwen2.5-1.5B-Instruct` needs ~3 GB of weights in fp16, and on two CPU cores writes roughly 2-4 tokens per second - minutes per answer. Together with fp32 NLLB (~2.5 GB) the advisory tab reached ~6 GB and was killed by the platform as soon as the weights loaded. Educational and non-profit projects can apply for increased resources, but that does not fix the CPU latency.


## Notebooks Structure

**`Maize_Disease_Detection_YOLO.ipynb`**

- **Environment Setup & Configuration**
- **1. Data Preparation & Validation**
- **2. Create Dataset Folder Structure**
- **3. Model A: YOLO26n-cls Pipeline**
  - *Transfer learning: what it actually does here*
- **4. Model B: Custom CNN Pipeline**
- **5. Model Comparison**
- **6. Error Analysis (Best Model)**
- **7. Error Reduction (Best Model)**
- **8. Model Explainability (Grad-CAM)**
- **9. Inference & Visual Inspection (Final Model)**
- **10. Limitations & Intended Use**
- **11. Model Export & Deployment (Optional)**

**`RAG_Advisory_Chatbot.ipynb`**

- **1. Environment & Setup**
- **2. Data Ingestion & Chunking**
- **3. Embeddings & Vector Store**
- **4. Retrieval & Augmentation**
- **5. Evaluation**
- **6. Guardrails & Safety**
- **7. Optimization & Tuning**
- **8. Inference Pipeline**
- **9. Error Analysis**
- **10. Error Reduction**
- **11. Generation Settings: Temperature**
  - *Temperature sweep, metrics, chart and conservative decision rule*
- **12. Export**
- **13. Database & Chat History**