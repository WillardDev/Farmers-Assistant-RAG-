"""
Maize Leaf Disease Detection - Streamlit application
====================================================

A single-file web app that classifies an uploaded maize leaf photograph into one
of four classes - Blight, Common Rust, Gray Leaf Spot or Healthy - using the
YOLO26 classifier trained in ``Maize_Disease_Detection_YOLO.ipynb``.

A third tab hosts the **Maize Advisory** chat. It answers farming questions in
English or Kiswahili from a local retrieval-augmented pipeline built in
``RAG_Advisory_Chatbot.ipynb`` and exported to ``rag_artifacts/export/``. That
pipeline (``rag_pipeline.py``) runs entirely on the machine: ``multilingual-e5-small``
for retrieval, ``nllb-200`` for the Swahili pivot and ``Qwen2.5-1.5B-Instruct``
for generation. The chat carries the conversation so far, so follow-up questions
like "so should I use phosphorus?" are understood in context.

The whole app sits behind a login. Farmer accounts (name, email, optional
profile picture, PBKDF2-hashed password) and every conversation are stored in
MySQL when the ``MYSQL_*`` environment variables are set (see ``db.py`` and the
``.env`` file). Accounts are required, so the app shows a sign-in screen until a
farmer logs in. A login is remembered across page reloads for 30 minutes of
inactivity using an opaque token in the URL (validated against ``login_tokens``),
and the sidebar lists the farmer's saved conversations so any of them can be
reopened and continued.

Run it with::

    streamlit run app.py

To open it on a phone on the same Wi-Fi network, bind to every interface::

    streamlit run app.py --server.address 0.0.0.0

then visit ``http://<your-computer-ip>:8501`` on the handset. The layout is
responsive: below 768px the columns stack, the type scale shrinks and the
controls grow to comfortable tap targets.

The interface is dark, which suits the subject and keeps projected or
night-time use comfortable. Because a single file has to stay self-contained,
the theme is applied with injected CSS rather than a ``.streamlit/config.toml``.

A user can either upload a photo or take one with the camera via
``st.camera_input``. Both paths end up as the same ``UploadedFile`` and flow
through the identical decode and analysis code, so a capture is treated exactly
like an upload. Note that browsers only grant camera access on a secure
context, so the camera works on ``localhost`` but not on a phone reaching the
app over plain HTTP - the UI says so where it matters.

Design notes
------------
* Every remote call is wrapped in try/except, so a missing weight file, a
  corrupt upload or a missing dependency degrades to a message instead of a
  stack trace.
* The model is loaded once per process via ``st.cache_resource``; a 3 MB file
  should not be re-read on every keystroke.
* There is no confidence slider. The user uploads a photo and the app reports
  the top class with its confidence, plus the full ranked distribution, so the
  strength of a result is visible without having to configure anything first.
* The metrics in the "Model Performance" tab are the real numbers produced by
  the notebook, not placeholders. Their provenance is stated inline.
"""

from __future__ import annotations

import io
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

import db
from rag_pipeline import AdvisoryPipeline

# --------------------------------------------------------------------------- #
#  Palette and static configuration
# --------------------------------------------------------------------------- #

NAVY = "#4A9EEB"       # primary: headers, branding, primary buttons (lightened for dark bg)
SLATE = "#A8B8CC"      # secondary: captions, muted labels, body copy
ORANGE = "#FF8A3D"     # accent: metrics, highlights, alerts (lightened for dark bg)

INK = "#0C1420"        # canvas: the page itself
SURFACE = "#151F2E"    # cards, sidebar, panels
SURFACE_HI = "#1D2A3C"  # hover / raised elements
LINE = "#2A3A4E"       # borders and dividers
TEXT = "#E8EDF5"       # body copy
ACCENT = "#6EC8FF"     # links, focus rings, secondary highlight

IMG_SIZE = 256         # what the classifier was trained at
PROJECT_ROOT = Path(__file__).resolve().parent
PROFILE_DIR = PROJECT_ROOT / "profile_images"

# The notebook's final model. It is a *fine-tuned* classifier, so it actually
# knows the four maize classes. ``yolo26n-cls.pt`` is the generic pretrained
# backbone and has never seen maize disease, so it is kept only as a fallback
# for a cold checkout with no trained weights.
FINAL_MODEL = "models/maize_disease_yolo26s.pt"
PRETRAINED_FALLBACK = "yolo26n-cls.pt"

# Maps the model's class labels onto the agronomically correct disease names.
# The training folder is called "Blight"; the disease it represents is
# Northern Corn Leaf Blight, so the UI says so.
CLASS_LABELS = {
    "Blight": "Northern Corn Leaf Blight",
    "Common_Rust": "Common Rust",
    "Gray_Leaf_Spot": "Gray Leaf Spot",
    "Healthy": "Healthy",
}

DISEASE_INFO: dict[str, dict[str, Any]] = {
    "Blight": {
        "pathogen": "Exserohilum turcicum (formerly Bipolaris turcica)",
        "symptoms": (
            "Long, cigar-shaped grey-green to tan lesions running parallel to the veins. "
            "Lesions start on lower leaves and move upward. Under favourable conditions they "
            "coalesce and the whole leaf dies, giving the crop a burnt appearance."
        ),
        "favourable": "Cool to moderate temperatures (18-27 C) with high humidity and prolonged leaf wetness.",
        "management": (
            "Rotate to a non-cereal crop for at least one season. Choose tolerant hybrids. "
            "Scout from tasselling and spray a protectant fungicide at first sight of lesions, "
            "repeating every 10-14 days while wet weather lasts. Manage residue to cut the "
            "inoculum that overwinters on the soil surface."
        ),
    },
    "Common_Rust": {
        "pathogen": "Puccinia sorghi",
        "symptoms": (
            "Small, oval-to-elongated cinnamon-brown pustules scattered on both leaf surfaces, "
            "raised and powdery to the touch. Leaves yellow and senesce early when pustule "
            "density is high, causing serious defoliation."
        ),
        "favourable": "Cool, moist conditions (16-25 C) with high relative humidity and frequent dew.",
        "management": (
            "Use resistant hybrids wherever available - this is by far the cheapest control. "
            "Remove volunteer maize and alternate hosts such as teosinte. Fungicides are rarely "
            "economic on susceptible varieties, and are justified only when a resistant hybrid "
            "cannot be used and rust is arriving early."
        ),
    },
    "Gray_Leaf_Spot": {
        "pathogen": "Cercospora zeae-maydis",
        "symptoms": (
            "Narrow, parallel-sided rectangular lesions restricted by the leaf veins, 3-15 mm long. "
            "They start translucent and later turn tan to dark brown with a greyish mouldy centre. "
            "Severe infection causes complete leaf death before grain fill."
        ),
        "favourable": "Warm, humid conditions (25-30 C) with heavy dew and dense canopy.",
        "management": (
            "The most yield-limiting of the three diseases here. Reduce canopy density, plant "
            "resistant hybrids and rotate crops. Because it needs long leaf wetness, spray timing "
            "matters far more than product choice: applications before or during very early "
            "infection, not once lesions are visible."
        ),
    },
    "Healthy": {
        "pathogen": "-",
        "symptoms": (
            "Uniform green tissue across the lamina, no necrotic spots, no pustules, no mould and "
            "no chlorosis. Leaf colour is even, which matters: pale but even is healthy, mottled "
            "pale is usually early nutrient or disease stress."
        ),
        "favourable": "Not applicable.",
        "management": (
            "No treatment. Record the leaf as a healthy reference for the field and keep scouting, "
            "since one healthy-looking plant in a field that is turning over usually means the "
            "problem is local."
        ),
    },
}

# Real numbers from the notebook, section 3.8 / model_comparison.md.
# Test split, 625 held-out images, yolo26n-cls baseline. Macro averages.
YOLO26N_METRICS = {
    "accuracy": 0.9456,
    "macro_precision": 0.9276,
    "macro_recall": 0.9399,
    "macro_f1": 0.9327,
    "val_macro_f1": 0.9375,
    "per_class": {
        "Blight": {"precision": 0.928, "recall": 0.901, "f1": 0.914, "support": 171},
        "Common_Rust": {"precision": 0.995, "recall": 0.959, "f1": 0.977, "support": 195},
        "Gray_Leaf_Spot": {"precision": 0.794, "recall": 0.906, "f1": 0.846, "support": 85},
        "Healthy": {"precision": 0.994, "recall": 0.994, "f1": 0.994, "support": 174},
    },
}

# --------------------------------------------------------------------------- #
#  Styling
# --------------------------------------------------------------------------- #

CSS = f"""
<style>
:root {{
  --navy: {NAVY};
  --slate: {SLATE};
  --orange: {ORANGE};
  --ink: {INK};
  --surface: {SURFACE};
  --surface-hi: {SURFACE_HI};
  --line: {LINE};
  --text: {TEXT};
  --accent: {ACCENT};
}}

/* ---------- canvas ---------- */
.stApp, .stApp [data-testid="stAppViewContainer"] {{ background: var(--ink); }}
.block-container {{ padding: 1.6rem 2rem 4rem; max-width: 1400px; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stToolbar"] button svg {{ fill: var(--slate); }}
#MainMenu, footer {{ visibility: hidden; }}

/* streamlit colours its own widgets from its theme vars; override them so the
   native controls match the injected palette instead of fighting it */
[data-testid="stAppViewContainer"] * {{ color: var(--text); }}
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4 {{ color: var(--navy); letter-spacing: -0.01em; }}
h1 {{ font-size: 1.85rem; font-weight: 700; }}
h2 {{ font-size: 1.3rem; font-weight: 650; }}
h3 {{ font-size: 1.05rem; font-weight: 600; }}
p, li, label {{ color: var(--slate); }}
a, a:link, a:visited {{ color: var(--accent); }}
code {{
  background: var(--surface-hi); color: var(--accent);
  border: 1px solid var(--line); border-radius: 5px; padding: .05rem .3rem;
}}
[data-testid="stCaptionContainer"] p {{ color: var(--slate); }}
hr {{ border-color: var(--line); }}

/* ---------- brand block ---------- */
.brand {{
  display: flex; align-items: center; gap: .8rem;
  padding: .1rem 0 .4rem 0;
}}
.brand-mark {{
  width: 42px; height: 42px; min-width: 42px; border-radius: 11px;
  background: var(--navy); color: var(--ink);
  display: flex; align-items: center; justify-content: center;
  font-weight: 700; font-size: 1.15rem;
}}
.brand-name {{ color: var(--navy); font-weight: 700; font-size: 1.05rem; line-height: 1.2; }}
.brand-sub  {{ color: var(--slate); font-size: .78rem; line-height: 1.2; }}

/* ---------- status pill ---------- */
.pill {{
  display: inline-flex; align-items: center; gap: .45rem;
  padding: .34rem .7rem; border-radius: 999px;
  font-size: .8rem; font-weight: 600; line-height: 1.2;
  border: 1px solid transparent;
}}
.pill-ok    {{ background: #14351F; color: #6EE7A0; border-color: #235B34; }}
.pill-bad   {{ background: #3A1717; color: #FCA5A5; border-color: #6B2424; }}
.pill-warn  {{ background: #382812; color: #FCD34D; border-color: #6B4A17; }}
.dot {{ width: 8px; height: 8px; border-radius: 50%; background: currentColor; }}

/* ---------- metric cards ---------- */
[data-testid="stMetric"] {{
  background: var(--surface);
  border: 1px solid var(--line);
  border-top: 3px solid var(--orange);
  border-radius: 12px;
  padding: .9rem 1rem .95rem 1rem;
  box-shadow: 0 1px 3px rgba(0,0,0,.4);
}}
[data-testid="stMetricLabel"] p {{ color: var(--slate); font-size: .78rem; }}
[data-testid="stMetricValue"] {{ color: var(--navy); font-size: 1.5rem; font-weight: 700; }}
[data-testid="stMetricDelta"] {{ color: var(--slate); }}

/* ---------- prediction hero ---------- */
.hero {{
  border-radius: 14px; padding: 1.1rem 1.3rem; margin: .2rem 0 .6rem 0;
  border: 1px solid var(--line); background: var(--surface);
  border-left: 6px solid var(--orange);
}}
.hero-label {{ font-size: .74rem; letter-spacing: .09em; text-transform: uppercase; color: var(--slate); }}
.hero-value {{ font-size: 2.1rem; font-weight: 750; color: var(--navy); line-height: 1.15; margin: .18rem 0 .1rem 0; }}
.hero-conf  {{ font-size: 1.0rem; font-weight: 600; color: var(--orange); }}
.hero-note  {{ font-size: .86rem; color: var(--slate); margin-top: .35rem; }}

.card {{
  background: var(--surface); border: 1px solid var(--line);
  border-radius: 12px; padding: 1rem 1.1rem; height: 100%;
}}
.card h4 {{ margin: 0 0 .4rem 0; font-size: 1rem; }}
.field {{ margin-top: .6rem; }}
.field-key {{ font-size: .7rem; letter-spacing: .07em; text-transform: uppercase; color: var(--orange); font-weight: 700; }}
.field-val {{ font-size: .9rem; color: var(--slate); line-height: 1.45; }}

/* ---------- tables & dataframes ---------- */
table {{ font-size: .88rem; }}
th {{ color: var(--navy) !important; font-weight: 650 !important; }}
[data-testid="stDataFrame"] {{ background: var(--surface); border: 1px solid var(--line); border-radius: 12px; }}
[data-testid="stDataFrame"] * {{ background-color: transparent; }}
[data-testid="stDataFrame"] [role="columnheader"] {{ background: var(--surface-hi); color: var(--navy) !important; }}
[data-testid="stDataFrame"] [role="gridcell"] {{ color: var(--text); border-color: var(--line) !important; }}

/* ---------- charts ---------- */
[data-testid="stVegaLiteChart"] {{ background: transparent; }}

/* ---------- tabs ---------- */
[data-baseweb="tab-list"] {{ gap: .35rem; border-bottom: 1px solid var(--line); background: transparent; }}
[data-baseweb="tab"] {{ font-weight: 600; color: var(--slate); padding: .55rem .9rem; background: transparent; }}
[data-baseweb="tab"]:hover {{ background: var(--surface-hi); color: var(--text); }}
[data-baseweb="tab"][aria-selected="true"] {{ color: var(--navy); background: transparent; }}
[data-baseweb="tab-highlight"] {{ background-color: var(--orange); }}

/* ---------- expanders ---------- */
[data-testid="stExpander"] {{
  background: var(--surface); border: 1px solid var(--line); border-radius: 12px;
}}
[data-testid="stExpander"] summary:hover {{ background: var(--surface-hi); }}

/* ---------- controls ---------- */
[data-testid="stSidebar"] {{
  background: var(--surface); border-right: 1px solid var(--line);
}}
[data-testid="stSidebar"] * {{ color: var(--text); }}
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {{ color: var(--navy); }}
[data-testid="stSidebarUserContent"] [data-testid="stCaptionContainer"] p {{ color: var(--slate); }}

/* buttons */
.stButton > button[kind="primary"] {{
  background: var(--orange); border-color: var(--orange); color: #1A1005;
  border-radius: 9px; font-weight: 650; padding: .55rem 1.1rem;
}}
.stButton > button[kind="primary"]:hover {{
  background: #FF9D5C; border-color: #FF9D5C; color: #1A1005;
}}
.stButton > button[kind="primary"]:active {{ background: #E8762C; border-color: #E8762C; }}
.stButton > button[kind="primary"]:focus {{ box-shadow: 0 0 0 3px rgba(255,138,61,.35); }}
.stButton > button[kind="secondary"],
.stButton > button:not([kind]) {{
  background: var(--surface-hi); color: var(--text); border-color: var(--line);
}}
.stButton > button:disabled {{ background: var(--surface-hi); color: var(--slate); border-color: var(--line); }}

/* uploader */
[data-testid="stFileUploaderDropzone"] {{
  border-radius: 12px; border: 1.5px dashed #3D5470; background: var(--surface);
}}
[data-testid="stFileUploaderDropzone"]:hover {{ border-color: var(--orange); background: var(--surface-hi); }}
[data-testid="stFileUploaderDropzoneInstructions"] div span {{ color: var(--slate) !important; }}
[data-testid="stFileUploaderDropzoneInstructions"] div small {{ color: var(--slate) !important; }}
[data-testid="stFileUploaderFile"] {{ color: var(--text); }}

/* alerts */
[data-testid="stAlert"] {{ border-radius: 12px; border: 1px solid var(--line); }}
[data-testid="stAlert"] * {{ color: var(--text) !important; }}
[data-testid="stAlert"][data-baseweb="notification"] {{ background: var(--surface-hi); }}

/* checkbox */
[data-testid="stCheckbox"] label p {{ color: var(--text); }}
[role="checkbox"][aria-checked="true"] {{ background-color: var(--orange); border-color: var(--orange); }}

/* scrollbar */
::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-track {{ background: var(--ink); }}
::-webkit-scrollbar-thumb {{ background: #2E4157; border-radius: 5px; }}
::-webkit-scrollbar-thumb:hover {{ background: #3D5470; }}

/* selection */
::selection {{ background: rgba(255,138,61,.3); }}

/* ---------- responsive: tablet ---------- */
@media (max-width: 1100px) {{
  .block-container {{ padding: 1.2rem 1.2rem 3rem; }}
  .hero-value {{ font-size: 1.8rem; }}
}}

/* ---------- responsive: phone ---------- */
@media (max-width: 768px) {{
  .block-container {{ padding: .9rem .85rem 3rem; }}
  h1 {{ font-size: 1.4rem; }}
  h2 {{ font-size: 1.12rem; }}
  .hero {{ padding: .85rem .9rem; border-left-width: 5px; }}
  .hero-value {{ font-size: 1.5rem; }}
  .hero-conf {{ font-size: .92rem; }}
  [data-testid="stMetric"] {{ padding: .7rem .8rem; }}
  [data-testid="stMetricValue"] {{ font-size: 1.25rem; }}
  /* give buttons and the uploader a comfortable tap target on a handset */
  .stButton > button, [data-testid="stFileUploaderDropzone"] {{ min-height: 48px; }}
  .stButton > button[kind="primary"] {{ width: 100%; }}
  [data-baseweb="tab"] {{ padding: .6rem .6rem; font-size: .9rem; }}
}}
</style>
"""


# --------------------------------------------------------------------------- #
#  Cached helpers
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner="Loading the classification model ...")
def load_model(model_path: str) -> tuple[Any, str | None]:
    """Load the YOLO classifier once per process.

    Returns ``(model, None)`` on success and ``(None, message)`` on failure, so
    that a missing file or a missing ``ultralytics`` install surfaces as a
    readable error instead of a traceback.
    """
    try:
        from ultralytics import YOLO

        return YOLO(model_path), None
    except Exception as exc:  # noqa: BLE001 - deliberately broad, we display it
        return None, f"{type(exc).__name__}: {exc}"


FINAL_WEIGHTS = Path(FINAL_MODEL)


def resolve_model_path(use_pretrained: bool) -> str:
    """Pick a weights file.

    The fine-tuned model is used unless the checkbox explicitly asks for the
    generic backbone. This deliberately does **not** fall back on its own when
    the fine-tuned file is missing: ``yolo26n-cls.pt`` has never seen maize
    disease, so silently swapping to it would produce confident nonsense about
    ImageNet classes. A missing fine-tuned checkpoint is a deployment fault and
    has to surface as one.
    """
    if use_pretrained:
        return PRETRAINED_FALLBACK
    return FINAL_MODEL


def deployment_hint(model_path: str, error: str | None) -> str | None:
    """Turn a model-load failure into an actionable fix, or ``None`` if unclear.

    Two deployment faults account for nearly every "the model could not be
    loaded" report, and in both cases the raw exception names the symptom rather
    than the cause:

    * the checkpoint was gitignored, so a deploy from git never had it;
    * ``opencv-python`` is a GUI build and the host is headless, so importing
      Ultralytics dies on ``libGL.so.1``.
    """
    text = (error or "")

    # Missing GUI/GL libraries: the classic headless-container failure.
    if "libGL" in text or "libgthread" in text or "libxcb" in text or (
        "shared object file" in text
    ):
        library = next(
            (lib for lib in ("libGL.so.1", "libxcb.so.1", "libgthread") if lib in text),
            "a GUI shared library",
        )
        return (
            f"`{library}` is missing, which means the host has no GUI libraries. "
            "Ultralytics imports OpenCV, and the default `opencv-python` wheel links "
            "against them, so the import fails on a headless server.\n\n"
            "For Streamlit Community Cloud, add the system packages and redeploy:\n\n"
            "    libgl1\n"
            "    libglib2.0-0\n"
            "    libxcb1\n"
            "    libsm6\n"
            "    libxext6\n"
            "    libxrender1\n\n"
            "each on its own line in a `packages.txt` file at the repository root. "
            "Alternatively install `opencv-python-headless` instead of "
            "`opencv-python` - but `ultralytics` depends on the latter, so the two "
            "can end up fighting over the same `cv2` directory."
        )

    if Path(model_path).exists():
        return None

    if Path(model_path).name == Path(FINAL_MODEL).name:
        return (
            f"`{model_path}` is not on disk. The weights are ignored by `.gitignore` "
            "(`*.pt`), so they are not in the repository and a deploy from git will "
            "not contain them.\n\n"
            "To fix it, commit the checkpoint:\n\n"
            f"    git add -f {model_path}\n"
            "    git commit -m 'Add fine-tuned maize disease weights'\n"
            "    git push\n\n"
            "or, to keep binaries out of git, host the file somewhere public and add a "
            "download step in `load_model` before the `YOLO(...)` call. Until one of "
            "those happens, detection is unavailable on this deployment."
        )

    return (
        f"`{model_path}` is not on disk and could not be downloaded. The generic "
        "pretrained backbone is only a fallback; it has never seen maize disease, so "
        "its predictions are not meaningful for this task."
    )


def decode_image(uploaded: Any) -> tuple[np.ndarray | None, str | None]:
    """Decode an upload into a BGR array for Ultralytics.

    EXIF orientation is applied first, because phone cameras record the image
    rotated and most viewers honour the tag while ``Image.open`` does not.

    Returns ``(image_bgr, None)`` or ``(None, message)``.
    """
    try:
        from PIL import Image, ImageOps

        raw = uploaded.getvalue()
        with Image.open(io.BytesIO(raw)) as img:
            img = ImageOps.exif_transpose(img) or img
            rgb = img.convert("RGB")
            width, height = rgb.size
            bgr = np.asarray(rgb)[:, :, ::-1].copy()  # RGB -> BGR
        return bgr, None
    except Exception as exc:  # noqa: BLE001
        return None, f"Could not read that image ({type(exc).__name__}: {exc})."


def predict_disease(model: Any, image_bgr: np.ndarray) -> dict[str, Any] | str:
    """Classify one image and return the full ranked probability distribution.

    Note: a classification head always returns the complete distribution in
    ``result.probs`` - Ultralytics does not drop low-scoring classes - so no
    confidence filtering is needed here. The whole vector is ranked and handed
    to the UI, which is what lets a weak prediction be judged from the runner-up
    scores rather than from an arbitrary cutoff.
    """
    try:
        result = model.predict(
            source=image_bgr,
            imgsz=IMG_SIZE,
            verbose=False,
        )[0]
        probs = result.probs.data.cpu().numpy()
        names = result.names

        ranked = sorted(
            (
                {"class": str(names[i]), "label": CLASS_LABELS.get(str(names[i]), str(names[i])),
                 "probability": float(probs[i])}
                for i in range(len(probs))
            ),
            key=lambda row: row["probability"],
            reverse=True,
        )
        top = ranked[0]
        runner_up = ranked[1]["probability"] if len(ranked) > 1 else 0.0
        return {
            "top": top,
            "ranked": ranked,
            # margin over the next class: a small value means two diseases look
            # alike, which is the caveat worth surfacing without a cutoff
            "margin": top["probability"] - runner_up,
        }
    except Exception as exc:  # noqa: BLE001
        return f"Inference failed ({type(exc).__name__}: {exc})."


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


# --------------------------------------------------------------------------- #
#  Page chrome
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="Maize Disease Detection & Advisory",
    page_icon="\U0001F33E",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"about": "Maize Leaf Disease Classification - YOLO26"},
)
st.markdown(CSS, unsafe_allow_html=True)


def brand() -> None:
    st.markdown(
        """
        <div class="brand">
          <div class="brand-mark">M</div>
          <div>
            <div class="brand-name">Maize Farming</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
#  Authentication
# --------------------------------------------------------------------------- #

def validate_signup(full_name: str, email: str, password: str, confirm: str,
                    county: str) -> list[str]:
    errors = []
    full_name = (full_name or "").strip()
    email = (email or "").strip()
    county = (county or "").strip()
    if len(full_name) < 2:
        errors.append("Enter your full name (at least 2 characters).")
    if len(full_name) > 120:
        errors.append("Name must be 120 characters or fewer.")
    if not db.EMAIL_RE.match(email):
        errors.append("Enter a valid email address, for example jina@example.com.")
    if len(email) > 190:
        errors.append("That email address is too long.")
    if len(password or "") < 8:
        errors.append("Choose a password with at least 8 characters.")
    if password != confirm:
        errors.append("The two passwords do not match.")
    if county and len(county) > 60:
        errors.append("County must be 60 characters or fewer.")
    return errors


def save_profile_image(farmer_id: int, uploaded) -> str | None:
    if uploaded is None:
        return None
    from PIL import Image

    PROFILE_DIR.mkdir(exist_ok=True)
    try:
        image = Image.open(uploaded).convert("RGB")
    except Exception:
        raise ValueError("That file could not be read as an image.")
    image.thumbnail((256, 256))
    path = PROFILE_DIR / f"farmer_{farmer_id}.jpg"
    image.save(path, "JPEG", quality=85)
    return str(path.relative_to(PROJECT_ROOT))


LOGIN_TIMEOUT_MINUTES = db.LOGIN_TTL_MINUTES
TOKEN_PARAM = "auth"
TOKEN_TOUCH_SECONDS = 120


def login_farmer(account: dict) -> None:
    st.session_state.auth_farmer = account
    st.session_state.adv_messages = []
    st.session_state.pop("adv_session_id", None)
    try:
        token = db.create_login_token(account["id"])
    except Exception:
        token = None
    if token:
        st.session_state.auth_token = token
        st.session_state.auth_touched = time.time()
        st.query_params[TOKEN_PARAM] = token


def logout_farmer() -> None:
    token = st.session_state.get("auth_token") or st.query_params.get(TOKEN_PARAM)
    try:
        db.delete_login_token(token)
    except Exception:
        pass
    try:
        st.query_params.pop(TOKEN_PARAM, None)
    except Exception:
        pass
    st.session_state.clear()


def restore_session() -> None:
    if st.session_state.get("auth_farmer"):
        return
    token = st.query_params.get(TOKEN_PARAM)
    if not token:
        return
    try:
        farmer_id = db.resolve_login_token(token)
    except Exception:
        farmer_id = None
    if not farmer_id:
        st.query_params.pop(TOKEN_PARAM, None)
        return
    account = db.get_farmer(farmer_id)
    if account:
        st.session_state.auth_farmer = account
        st.session_state.auth_token = token
        st.session_state.auth_touched = time.time()
        st.session_state.adv_messages = []
        st.session_state.pop("adv_session_id", None)


def keep_session_alive() -> None:
    token = st.session_state.get("auth_token")
    if not token:
        return
    now = time.time()
    if now - st.session_state.get("auth_touched", 0) < TOKEN_TOUCH_SECONDS:
        return
    st.session_state.auth_touched = now
    try:
        db.touch_login_token(token)
    except Exception:
        pass


def render_auth_page() -> None:
    brand()
    st.markdown("### Welcome")
    st.caption(
        "Sign in to use the maize disease detector and the advisory chatbot. "
        "New here? Create a free account in a few seconds. You stay signed in "
        f"for {LOGIN_TIMEOUT_MINUTES} minutes of inactivity, even after a reload."
    )

    intro_col, form_col = st.columns([1, 1], gap="large")
    with intro_col:
        st.markdown(
            """
            <div class="card">
              <h4>What you get</h4>
              <div class="field">
                <div class="field-key">Leaf diagnosis</div>
                <div class="field-val">Photograph a maize leaf and get the most likely
                disease with a confidence score.</div>
              </div>
              <div class="field">
                <div class="field-key">Advisory chat</div>
                <div class="field-val">Ask about planting, fertiliser, pests, diseases,
                harvest and storage, in English or Kiswahili.</div>
              </div>
              <div class="field">
                <div class="field-key">Saved history</div>
                <div class="field-val">Your conversations and feedback are stored
                against your account.</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with form_col:
        mode = st.radio("Account", ["Log in", "Create account"], horizontal=True,
                        label_visibility="collapsed")

        if mode == "Log in":
            with st.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Log in", type="primary", width="stretch")
            if submitted:
                errors = []
                if not db.EMAIL_RE.match((email or "").strip()):
                    errors.append("Enter a valid email address.")
                if not password:
                    errors.append("Enter your password.")
                for message in errors:
                    st.error(message)
                if not errors:
                    try:
                        account = db.authenticate(email.strip(), password)
                    except Exception as exc:
                        st.error(f"Could not reach the account database: {exc}")
                        account = None
                    if account is None:
                        st.error("Email or password is incorrect.")
                    else:
                        login_farmer(account)
                        st.rerun()
        else:
            with st.form("signup_form"):
                full_name = st.text_input("Full name")
                email = st.text_input("Email")
                password = st.text_input("Password", type="password",
                                         help="At least 8 characters.")
                confirm = st.text_input("Confirm password", type="password")
                county = st.text_input("County (optional)")
                preferred = st.selectbox("Preferred language", ["English", "Kiswahili"])
                picture = st.file_uploader("Profile picture (optional)",
                                           type=["jpg", "jpeg", "png"])
                submitted = st.form_submit_button("Create account", type="primary",
                                                  width="stretch")
            if submitted:
                errors = validate_signup(full_name, email, password, confirm, county)
                for message in errors:
                    st.error(message)
                if not errors:
                    try:
                        farmer_id = db.create_account(
                            full_name.strip(), email.strip(), password,
                            county.strip() or None,
                            "sw" if preferred == "Kiswahili" else "en")
                        if picture is not None:
                            try:
                                relative = save_profile_image(farmer_id, picture)
                            except ValueError as exc:
                                st.warning(str(exc))
                                relative = None
                            if relative:
                                db.update_profile_image(farmer_id, relative)
                        login_farmer(db.get_farmer(farmer_id))
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
                    except Exception as exc:
                        st.error(f"Could not create the account: {exc}")


def ensure_schema() -> bool:
    if not db.db_configured():
        return False
    try:
        db.create_schema()
        return True
    except Exception:
        return False


if "schema_ready" not in st.session_state:
    st.session_state.schema_ready = ensure_schema()

restore_session()

if not st.session_state.get("auth_farmer"):
    render_auth_page()
    st.stop()

farmer = st.session_state.auth_farmer
keep_session_alive()


@st.cache_resource(show_spinner=False)
def chunk_lookup() -> dict:
    lookup = {}
    path = PROJECT_ROOT / "rag_artifacts" / "export" / "chunks.jsonl"
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    chunk = json.loads(line)
                    lookup[chunk["chunk_id"]] = chunk
    except Exception:
        pass
    return lookup


def history_citations(raw) -> list:
    if not raw:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return []
    if not isinstance(raw, list):
        return []
    lookup = chunk_lookup()
    citations = []
    for item in raw:
        if isinstance(item, dict) and item.get("title"):
            citations.append(item)
        elif isinstance(item, str) and item in lookup:
            chunk = lookup[item]
            citations.append({
                "n": len(citations) + 1,
                "chunk_id": item,
                "title": chunk.get("title", item),
                "source_id": chunk.get("source_id", ""),
                "page": chunk.get("page", ""),
                "snippet": chunk.get("text", "")[:280].strip(),
            })
    return citations


def open_conversation(session_id: int) -> None:
    rows = db.read_history(farmer["id"], session_id)
    messages = []
    for row in rows:
        entry = {"role": row["role"], "content": row["message"]}
        if row["role"] == "assistant":
            entry["db_id"] = row["id"]
            citations = history_citations(row.get("sources"))
            if citations:
                entry["citations"] = citations
        messages.append(entry)
    st.session_state.adv_messages = messages
    st.session_state.adv_session_id = session_id


# --------------------------------------------------------------------------- #
#  Sidebar
# --------------------------------------------------------------------------- #

with st.sidebar:
    brand()

    profile_path = farmer.get("profile_image")
    if profile_path and (PROJECT_ROOT / profile_path).exists():
        st.image(str(PROJECT_ROOT / profile_path), width=72)
    st.markdown(f"**{farmer['full_name']}**")
    if farmer.get("email"):
        st.caption(farmer["email"])
    if farmer.get("preferred_lang"):
        st.caption("Swahili preferred" if farmer["preferred_lang"] == "sw" else "English preferred")
    if st.button("Log out", width="stretch"):
        logout_farmer()
        st.rerun()

    st.markdown("---")
    if st.button("New chat", width="stretch", type="primary"):
        st.session_state.adv_messages = []
        st.session_state.pop("adv_session_id", None)
        st.toast("Started a new chat. Open the Maize Advisory tab to ask.")

    st.markdown("#### Your conversations")
    if not db.db_configured():
        st.caption("Your conversation history is unavailable because the database "
                   "is not configured.")
    else:
        try:
            conversations = db.list_sessions(farmer["id"], limit=30)
        except Exception as exc:
            conversations = []
            st.caption(f"Could not load your conversations: {exc}")
        if conversations:
            active_id = st.session_state.get("adv_session_id")
            for convo in conversations:
                topic = (convo.get("topic") or "New conversation").strip()
                if len(topic) > 48:
                    topic = topic[:45].rstrip() + "..."
                when = (
                    convo["started_at"].strftime("%d %b %Y")
                    if convo.get("started_at") else ""
                )
                label = f"{'● ' if convo['id'] == active_id else ''}{topic}"
                if st.button(label, key=f"convo_{convo['id']}", width="stretch",
                             help=f"Opened {when}. Click to read or continue this chat."):
                    try:
                        open_conversation(convo["id"])
                    except Exception as exc:
                        st.toast(f"Could not open that conversation: {exc}")
                    else:
                        st.rerun()
        else:
            st.caption("No conversations yet. Ask a question in the Maize Advisory tab.")


# --------------------------------------------------------------------------- #
#  Tabs
# --------------------------------------------------------------------------- #

st.markdown("---")
tab_detect, tab_perf, tab_advisory = st.tabs(
    ["Disease Detection", "Model Performance", "Maize Advisory"]
)


# ---- Tab 1: the workspace -------------------------------------------------- #
with tab_detect:
    st.markdown("### Disease detection")
    st.caption(
        f"Upload one maize leaf photograph, or take one with your camera. The "
        f"classifier expects images at {IMG_SIZE}x{IMG_SIZE} and will resize what "
        f"you give it."
    )

    st.markdown("#### Model status")
    use_pretrained = st.checkbox(
        "Use pretrained yolo26n-cls instead",
        value=False,
        help=(
            "The default weights are the fine-tuned model, which knows the four "
            "maize classes. The pretrained backbone has never seen maize disease, "
            "so its predictions on this task are not meaningful."
        ),
    )
    model_path = resolve_model_path(use_pretrained)
    model, model_error = load_model(model_path)

    if model_error is None:
        size_mb = Path(model_path).stat().st_size / 1e6 if Path(model_path).exists() else None
        detail = f" &middot; {size_mb:.1f} MB" if size_mb else ""
        st.markdown(
            f'<div class="pill pill-ok"><span class="dot"></span>Loaded{detail}</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"`{model_path}`")
    else:
        st.markdown(
            '<div class="pill pill-bad"><span class="dot"></span>Load failed</div>',
            unsafe_allow_html=True,
        )
        st.error(model_error, icon="\U0001F534")
        hint = deployment_hint(model_path, model_error)
        if hint:
            st.warning(hint, icon="\U0001F4E5")

    source = st.radio(
        "Photo source",
        ["Upload a photo", "Use the camera"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if source == "Use the camera":
        uploaded = st.camera_input(
            "Take a photo of a maize leaf",
            help="Allow camera access when your browser asks, then capture.",
        )
        if uploaded is None:
            st.info(
                "Nothing captured yet. Allow camera access when the browser asks, "
                "then frame a single leaf in good light and take the photo.",
                icon="\U0001F4F7",
            )
            with st.expander("If the camera does not appear"):
                st.markdown(
                    "Browsers only expose the camera on a **secure context**. "
                    "`localhost` counts as secure, so it works on the machine "
                    "running Streamlit. Reaching the app from a phone over "
                    "`http://192.168.x.x:8501` does **not**, so the camera is "
                    "hidden on the handset and you must upload instead.\n\n"
                    "To get the camera working on a phone you need HTTPS in front "
                    "of the app - for example a tunnel such as `cloudflared tunnel "
                    "--url http://localhost:8501`, or a reverse proxy with a "
                    "certificate. Both are outside what this script can do for you, "
                    "since HTTPS has to terminate somewhere other than the Streamlit "
                    "process."
                )
    else:
        uploaded = st.file_uploader(
            "Maize leaf photograph",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
            label_visibility="collapsed",
        )

    if uploaded is None:
        # Clean empty state rather than a bare "please upload" line.
        left, right = st.columns([1, 1], gap="large")
        with left:
            st.markdown(
                """
                <div class="card">
                  <h4>No image yet</h4>
                  <div class="field">
                    <div class="field-key">What works best</div>
                    <div class="field-val">
                      A single leaf filling the frame, in focus, lit evenly.
                      The notebook measured that errors track image sharpness,
                      not brightness or colour, so a blurry photo is the most
                      likely thing to produce a wrong answer.
                    </div>
                  </div>
                  <div class="field">
                    <div class="field-key">Accepted formats</div>
                    <div class="field-val">JPG, JPEG, PNG</div>
                  </div>
                  <div class="field">
                    <div class="field-key">In the field</div>
                    <div class="field-val">
                      Switch to <strong>Use the camera</strong> above to skip the
                      file step entirely.
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with right:
            st.markdown(
                """
                <div class="card">
                  <h4>The four classes</h4>
                  <div class="field">
                    <div class="field-key">Diseases</div>
                    <div class="field-val">Northern Corn Leaf Blight &middot; Common Rust &middot; Gray Leaf Spot</div>
                  </div>
                  <div class="field">
                    <div class="field-key">Also</div>
                    <div class="field-val">Healthy</div>
                  </div>
                  <div class="field">
                    <div class="field-key">Tip</div>
                    <div class="field-val">
                      Blight and Gray Leaf Spot can look similar early on. Compare a
                      result against the symptoms listed after the model names it.
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        image_bgr, decode_error = decode_image(uploaded)

        if decode_error is not None:
            st.error(decode_error, icon="\U0001F5B1")
        else:
            height, width = image_bgr.shape[:2]

            img_col, meta_col = st.columns([1.25, 1], gap="large")
            with img_col:
                st.image(uploaded, caption="Uploaded photograph", width="stretch")
            with meta_col:
                st.markdown(
                    f"""
                    <div class="card">
                      <h4>{uploaded.name}</h4>
                      <div class="field">
                        <div class="field-key">Dimensions</div>
                        <div class="field-val">{width} x {height} px</div>
                      </div>
                      <div class="field">
                        <div class="field-key">File size</div>
                        <div class="field-val">{len(uploaded.getvalue()) / 1024:.0f} KB</div>
                      </div>
                      <div class="field">
                        <div class="field-key">Resized to</div>
                        <div class="field-val">{IMG_SIZE} x {IMG_SIZE} px</div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("")

            if model_error is not None:
                st.error(
                    "Detection is unavailable because the model could not be loaded. "
                    "The reason is shown in the sidebar under **Model status**.",
                    icon="\U0001F534",
                )

            if st.button(
                "Run Detection",
                type="primary",
                width="stretch",
                disabled=model_error is not None,
            ):
                if model_error is not None:
                    st.error("The model is not loaded, so detection is unavailable.", icon="\U0001F534")
                else:
                    with st.spinner("Classifying ..."):
                        outcome = predict_disease(model, image_bgr)

                    if isinstance(outcome, str):
                        st.error(outcome, icon="\U0001F5B1")
                    else:
                        top = outcome["top"]
                        confidence = top["probability"]
                        margin = outcome["margin"]
                        bars, table_col = st.columns([1, 1], gap="large")

                        with bars:
                            st.success(
                                f"### {top['label']}\n\n"
                                f"**{format_pct(confidence)}** confidence &middot; "
                                f"{format_pct(margin)} clear of the next class",
                                icon="\U0001F7E9",
                            )
                            if confidence < 0.60:
                                st.caption(
                                    "This model is right 94.6% of the time, but it does "
                                    "occasionalise: it averages 79.0% confidence on the "
                                    "photos it gets wrong, so most mistakes are confident "
                                    "ones. Use the ranked scores below as a second opinion "
                                    "and confirm anything consequential with an agronomist."
                                )

                            st.markdown(
                                f"""
                                <div class="hero">
                                  <div class="hero-label">Most likely class</div>
                                  <div class="hero-value">{top['label']}</div>
                                  <div class="hero-conf">{format_pct(confidence)} confidence</div>
                                  <div class="hero-note">
                                    Model class label: <code>{top['class']}</code>
                                  </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            # Disease info appears only for the class the model just
                            # predicted - not as a catalogue of all four.
                            if top["class"] in DISEASE_INFO:
                                info = DISEASE_INFO[top["class"]]
                                st.markdown("---")
                                st.markdown(f"#### About {top['label']}")
                                st.caption(
                                    f"Causal organism: {info['pathogen']}"
                                    if info["pathogen"] != "-"
                                    else "Not a disease - no causal organism."
                                )

                                sym_col, cond_col = st.columns(2, gap="large")
                                with sym_col:
                                    st.markdown("**Symptoms**")
                                    st.caption(info["symptoms"])
                                with cond_col:
                                    st.markdown("**Favourable conditions**")
                                    st.caption(info["favourable"])

                                st.markdown("**What to do**")
                                st.markdown(info["management"])

                                st.info(
                                    "Confirm with an agronomist before acting. This is a "
                                    "triage aid that says which leaves deserve a closer "
                                    "look, not a spray recommendation - reaction rates "
                                    "differ by country, variety and disease pressure.",
                                    icon="\U0001F6E0",
                                )

                        with table_col:
                            st.markdown("**Score breakdown**")
                            st.caption("Softmax probability for every class.")
                            rows = {
                                row["label"]: round(row["probability"], 4)
                                for row in outcome["ranked"]
                            }
                            st.bar_chart(
                                rows,
                                height=260,
                                color=ORANGE,
                            )
                            st.dataframe(
                                {
                                    "Class": [row["label"] for row in outcome["ranked"]],
                                    "Probability": [
                                        format_pct(row["probability"]) for row in outcome["ranked"]
                                    ],
                                },
                                hide_index=True,
                                width="stretch",
                            )
                            runner_up = outcome["ranked"][1]
                            margin = confidence - runner_up["probability"]
                            st.caption(
                                f"Margin over second place "
                                f"({runner_up['label']}): {format_pct(margin)}"
                            )
                            if margin < 0.15:
                                st.caption(
                                    "A margin this narrow means the model is close to "
                                    "indifferent between the top two classes."
                                )

    st.markdown("---")
    st.markdown("#### Model performance")
    m = YOLO26N_METRICS
    c1, c2 = st.columns(2)
    c1.metric("Test accuracy", format_pct(m["accuracy"]))
    c2.metric("Macro F1", format_pct(m["macro_f1"]))
    st.caption("yolo26n-cls baseline, 625 held-out images.")

    st.markdown("#### About")
    st.markdown(
        """
        Classifies a maize leaf photograph into **Blight**, **Common Rust**,
        **Gray Leaf Spot** or **Healthy**.

        The weights come from a YOLO26 classification model fine-tuned on the
        Corn or Maize Leaf Disease dataset, split 70/15/15 and evaluated on 625
        held-out images.

        Intended use is triage and prioritisation with agronomist confirmation
        - not standalone diagnosis.
        """
    )
    st.caption(
        "Triage aid only. A 5% error rate means roughly one leaf in twenty is "
        "misread, so a healthy leaf reported as diseased wastes money and "
        "builds resistance to the tool."
    )


# ---- Tab 2: measured performance ------------------------------------------ #
with tab_perf:
    st.markdown("### Model performance")
    st.caption(
        "Real evaluation output from the notebook, not placeholder values. "
        "Figures are for the **yolo26n-cls baseline** on the 625-image held-out "
        "test split; the deployed weights are the larger yolo26s variant, which "
        "scored 95.68% on the same split."
    )

    m = YOLO26N_METRICS
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Precision", format_pct(m["macro_precision"]), help="Macro average over 4 classes")
    k2.metric("Recall", format_pct(m["macro_recall"]), help="Macro average over 4 classes")
    k3.metric("F1-score", format_pct(m["macro_f1"]), help="Macro average over 4 classes")
    k4.metric("Accuracy", format_pct(m["accuracy"]), help="Overall, 591 of 625 correct")

    st.markdown("#### Per class")
    per_class = m["per_class"]
    frame_left, frame_right = st.columns([3, 2], gap="large")
    with frame_left:
        st.dataframe(
            {
                "Class": [CLASS_LABELS[c] for c in per_class],
                "Precision": [format_pct(v["precision"]) for v in per_class.values()],
                "Recall": [format_pct(v["recall"]) for v in per_class.values()],
                "F1": [format_pct(v["f1"]) for v in per_class.values()],
                "Support": [v["support"] for v in per_class.values()],
            },
            hide_index=True,
            width="stretch",
        )
    with frame_right:
        st.bar_chart(
            {CLASS_LABELS[c]: round(v["f1"], 4) for c, v in per_class.items()},
            height=230,
            color=ORANGE,
        )
        st.caption("F1-score by class.")

    with st.expander("Method, and what these numbers do not say"):
        st.markdown(
            f"""
- Model selection was on **validation macro F1**, never on test. The selection
  criterion was fixed in advance, and the winning model reached
  {format_pct(m['val_macro_f1'])} validation macro F1.
- Test was scored once, with the final model, to produce an estimate rather
  than a selection artefact.
- At 625 images a result near 95% carries roughly +/-1.7 points of sampling
  uncertainty. Two models within about 3 points of each other are **not**
  reliably different on this split, which is why sub-1% validation gains were
  labelled noise in the notebook instead of being banked as improvements.
- Grad-CAM on the deployed model highlights 3.4% of the frame on the images it
  gets wrong against 5.6% on the ones it gets right, so it is reading a narrow
  region rather than sweeping the background. That is reassuring, but it is
  still a coarse visualisation, not a measurement.
- The 96% project target was met on validation (96.00%) and **missed on test**
  (95.68%). Both numbers are reported because reporting only the first would be
  misleading.
"""
        )


# --------------------------------------------------------------------------- #
#  Tab 3: advisory chat
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner=False)
def advisory_pipeline() -> AdvisoryPipeline:
    return AdvisoryPipeline()


def advisory_init_db() -> None:
    if "adv_db_ok" in st.session_state:
        return
    if not db.db_configured():
        st.session_state.adv_db_ok = False
        st.session_state.adv_db_note = (
            "Conversations are not saved. Add the MYSQL_* variables to a .env file "
            "in the project root to store them."
        )
        return
    try:
        db.create_schema()
        st.session_state.adv_db_ok = True
        st.session_state.adv_db_note = ""
    except Exception as exc:
        st.session_state.adv_db_ok = False
        st.session_state.adv_db_note = (
            f"Database unavailable, using this session only: {exc}"
        )


def advisory_render_sources(citations) -> None:
    with st.expander("Sources"):
        for cite in citations:
            st.markdown(
                f"**[{cite['n']}]** {cite['title']} "
                f"(`{cite['source_id']}`, page {cite['page']})"
            )
            st.caption(cite["snippet"] + " ...")


def advisory_record_feedback(message_id: int, helpful: bool) -> None:
    try:
        db.set_feedback(message_id, helpful)
        st.toast("Thank you - feedback recorded.")
    except Exception as exc:
        st.toast(f"Could not record feedback: {exc}")


with tab_advisory:
    st.markdown("### Maize advisory")
    st.caption(
        "Ask about growing maize in Kenya - planting, varieties, fertiliser, pests, "
        "diseases, harvest and storage. English or Kiswahili."
    )

    advisory_init_db()
    if "adv_messages" not in st.session_state:
        st.session_state.adv_messages = []

    if st.session_state.adv_db_ok:
        st.caption(
            f"Signed in as {farmer['full_name']}. Your conversations are saved to "
            f"your account."
        )
    else:
        st.info(st.session_state.adv_db_note)

    for index, message in enumerate(st.session_state.adv_messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("citations"):
                advisory_render_sources(message["citations"])
            is_last = index == len(st.session_state.adv_messages) - 1
            if message["role"] == "assistant" and is_last and message.get("db_id"):
                mid = message["db_id"]
                up_col, down_col, _ = st.columns([1, 1, 6])
                if up_col.button("Helpful", key=f"adv_up_{mid}"):
                    advisory_record_feedback(mid, True)
                if down_col.button("Not helpful", key=f"adv_dn_{mid}"):
                    advisory_record_feedback(mid, False)

    question = st.chat_input("Uliza kuhusu mahindi ... / Ask about maize ...")
    if question:
        pipeline = advisory_pipeline()
        history = [{"role": m["role"], "content": m["content"]}
                   for m in st.session_state.adv_messages]
        result = None
        try:
            if not pipeline.is_loaded():
                with st.spinner(
                    "Loading the advisory models. This happens on the first question "
                    "only and can take a minute."
                ):
                    pipeline.load()
            with st.spinner("Searching the documents and writing an answer ..."):
                result = pipeline.ask(question, history=history)
        except Exception as exc:
            st.error(f"The advisory service is unavailable: {exc}")

        if result is not None:
            entry = {"role": "assistant", "content": result["answer"],
                     "citations": result["citations"]}
            st.session_state.adv_messages.append({"role": "user", "content": question})
            if st.session_state.get("adv_db_ok"):
                try:
                    if not st.session_state.get("adv_session_id"):
                        st.session_state.adv_session_id = db.start_session(
                            farmer["id"], farmer.get("preferred_lang", "en"))
                    entry["db_id"] = db.store_message(
                        farmer["id"], st.session_state.adv_session_id, question, result)
                except Exception as exc:
                    st.toast(f"This turn was not saved: {exc}")
            st.session_state.adv_messages.append(entry)
            st.rerun()