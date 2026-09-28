import html
import json

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from model import ResNet18


# ==================================================
# Page Configuration
# ==================================================

st.set_page_config(
    page_title="Intel Image Classification",
    page_icon="◉",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ==================================================
# Custom CSS — Design System
# ==================================================

st.markdown(
    """
<style>
html, body, [class*="css"] {
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI",
                 Roboto, Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
}

.stApp {
    background: #F5F7FB;
    color: #0B1B3A;
}

/* Centered single-column container */
.block-container {
    padding-top: 2.2rem;
    padding-bottom: 3rem;
    max-width: 880px;
    margin-left: auto;
    margin-right: auto;
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { background: transparent; }

/* ---------- Top bar (still centered container, but content flows) ---------- */
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.4rem 0 1.6rem 0;
    border-bottom: 1px solid #E4E9F2;
    margin-bottom: 1.8rem;
    flex-wrap: wrap;
    gap: 1rem;
}

.brand-block { display: flex; align-items: center; gap: 0.9rem; }

.brand-mark {
    width: 44px; height: 44px; border-radius: 12px;
    background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
    display: flex; align-items: center; justify-content: center;
    color: #FFFFFF; font-size: 1.3rem; font-weight: 700;
    box-shadow: 0 6px 16px rgba(30, 58, 138, 0.22);
}

.brand-text-title {
    font-size: 1.08rem; font-weight: 700; color: #0B1B3A;
    letter-spacing: -0.01em; line-height: 1.2; margin: 0;
}

.brand-text-sub {
    font-size: 0.78rem; color: #7A869A;
    margin: 0.15rem 0 0 0; font-weight: 500;
}

.header-meta { display: flex; gap: 0.55rem; align-items: center; }

.meta-pill {
    display: inline-flex; align-items: center; gap: 0.35rem;
    padding: 0.42rem 0.78rem; border-radius: 999px;
    background: #FFFFFF; border: 1px solid #E4E9F2;
    color: #3E4C66; font-size: 0.76rem; font-weight: 600;
    letter-spacing: 0.02em;
}

.meta-pill .dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #14B8A6; display: inline-block;
    box-shadow: 0 0 0 3px rgba(20, 184, 166, 0.15);
}

/* ---------- Hero (centered) ---------- */
.hero {
    padding: 0.5rem 0 1.2rem 0;
    text-align: center;
    max-width: 720px;
    margin: 0 auto;
}

.hero-badge {
    display: inline-block; padding: 0.35rem 0.75rem;
    border-radius: 999px; background: rgba(37, 99, 235, 0.08);
    color: #2563EB; font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.08em; margin-bottom: 1rem;
    border: 1px solid rgba(37, 99, 235, 0.15);
}

.hero h1 {
    color: #0B1B3A; font-size: 2.7rem; font-weight: 800;
    margin: 0; letter-spacing: -0.035em; line-height: 1.08;
}

.hero p {
    color: #5A6784; font-size: 1.05rem;
    margin-top: 0.9rem; margin-bottom: 0;
    line-height: 1.55;
}

/* ---------- Supported categories (centered) ---------- */
.cats-wrap {
    display: flex;
    justify-content: center;
    margin: 0.4rem 0 2.2rem 0;
}

.cats-row {
    display: flex; flex-wrap: wrap; gap: 0.5rem;
    justify-content: center;
}

.class-chip {
    display: inline-flex; align-items: center; gap: 0.45rem;
    padding: 0.5rem 0.95rem; border-radius: 999px;
    background: #FFFFFF; border: 1px solid #E4E9F2;
    color: #3E4C66; font-size: 0.85rem; font-weight: 600;
    text-transform: capitalize;
    box-shadow: 0 1px 2px rgba(11, 27, 58, 0.03);
}

.class-chip .chip-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #2563EB; display: inline-block;
}

/* ---------- Surfaces ---------- */
.surface {
    background: #FFFFFF;
    border: 1px solid #E4E9F2;
    border-radius: 16px;
    padding: 1.4rem 1.4rem 1.5rem 1.4rem;
    box-shadow: 0 1px 2px rgba(11, 27, 58, 0.03),
                0 8px 24px rgba(11, 27, 58, 0.04);
}

.panel-label {
    display: flex; align-items: center; justify-content: space-between;
    font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.1em; color: #7A869A;
    text-transform: uppercase; margin-bottom: 1rem;
}

.panel-label .step {
    color: #2563EB; background: rgba(37, 99, 235, 0.08);
    padding: 0.18rem 0.5rem; border-radius: 6px;
    font-size: 0.68rem; letter-spacing: 0.06em;
}

/* ---------- Uploader ---------- */
[data-testid="stFileUploader"] {
    background: #F8FAFD; border: 1.5px dashed #C9D3E5;
    border-radius: 14px; padding: 0.6rem;
    transition: all 0.2s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: #2563EB; background: #F1F6FF;
}
[data-testid="stFileUploader"] section { padding: 0.8rem; }
[data-testid="stFileUploader"] label {
    color: #3E4C66 !important; font-weight: 600 !important;
}
[data-testid="stFileUploader"] small { color: #7A869A !important; }

/* ---------- Image ---------- */
[data-testid="stImage"] {
    border-radius: 14px; overflow: hidden;
    border: 1px solid #E4E9F2; background: #FFFFFF;
}
[data-testid="stImage"] img { display: block; width: 100%; }
[data-testid="stImage"] figcaption {
    color: #7A869A !important; font-size: 0.78rem !important;
    text-align: center; padding: 0.5rem 0 0.2rem 0;
}

/* ---------- Empty state ---------- */
.prediction-empty {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    text-align: center; padding: 3rem 1.4rem; color: #7A869A;
}
.prediction-empty .icon {
    width: 56px; height: 56px; border-radius: 14px;
    background: #F1F5FB; border: 1px solid #E4E9F2;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.6rem; color: #2563EB; margin-bottom: 1rem;
}
.prediction-empty .title {
    font-size: 0.95rem; font-weight: 700;
    color: #3E4C66; margin-bottom: 0.35rem;
}
.prediction-empty .desc {
    font-size: 0.82rem; color: #7A869A;
    max-width: 280px; line-height: 1.5;
}

/* ---------- Prediction card ---------- */
.prediction-card {
    background: linear-gradient(160deg, #FFFFFF 0%, #F8FAFD 100%);
    border: 1px solid #E4E9F2; border-radius: 14px;
    padding: 1.6rem 1.6rem 1.5rem 1.6rem;
    position: relative; overflow: hidden;
}
.prediction-card::before {
    content: ""; position: absolute; top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #2563EB 0%, #14B8A6 100%);
}

.prediction-eyebrow {
    font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.12em; color: #7A869A;
    text-transform: uppercase; margin-bottom: 0.6rem;
}
.prediction-value {
    font-size: 2.6rem; font-weight: 800; color: #0B1B3A;
    letter-spacing: -0.03em; line-height: 1.05;
    margin: 0; text-transform: capitalize;
}
.confidence-row {
    display: flex; align-items: baseline; gap: 0.5rem;
    margin-top: 1rem; padding-top: 1rem;
    border-top: 1px solid #EEF2F8;
}
.confidence-label {
    font-size: 0.85rem; color: #7A869A; font-weight: 600;
}
.confidence-value {
    font-size: 1.2rem; color: #2563EB; font-weight: 800;
    letter-spacing: -0.01em;
}
.confidence-bar-track {
    margin-top: 0.9rem; height: 6px;
    border-radius: 999px; background: #EEF2F8; overflow: hidden;
}
.confidence-bar-fill {
    height: 100%; border-radius: 999px;
    background: linear-gradient(90deg, #2563EB 0%, #14B8A6 100%);
}

/* ---------- At a glance card ---------- */
.glance-title {
    font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.1em; color: #7A869A;
    text-transform: uppercase; margin-bottom: 0.6rem;
}
.glance-row {
    display: flex; justify-content: space-between;
    padding: 0.55rem 0;
    border-bottom: 1px solid #F1F4FA;
}
.glance-row:last-child { border-bottom: none; padding-bottom: 0.15rem; }
.glance-k { font-size: 0.88rem; color: #5A6784; font-weight: 600; }
.glance-v {
    font-size: 0.88rem; color: #0B1B3A; font-weight: 700;
    text-transform: capitalize;
}
.glance-v.accent { color: #2563EB; }

/* ---------- Probability rows ---------- */
.prob-row {
    display: flex; align-items: center; gap: 1rem;
    padding: 0.7rem 0; border-bottom: 1px solid #F1F4FA;
}
.prob-row:last-child {
    border-bottom: none; padding-bottom: 0.2rem;
}
.prob-name {
    flex: 0 0 120px; font-size: 0.9rem;
    font-weight: 600; color: #3E4C66;
    text-transform: capitalize;
}
.prob-name.is-top { color: #0B1B3A; font-weight: 800; }
.prob-track {
    flex: 1; height: 8px; border-radius: 999px;
    background: #EEF2F8; overflow: hidden; position: relative;
}
.prob-fill {
    height: 100%; border-radius: 999px; background: #C7D2E5;
}
.prob-fill.is-top {
    background: linear-gradient(90deg, #2563EB 0%, #14B8A6 100%);
}
.prob-value {
    flex: 0 0 70px; text-align: right;
    font-size: 0.88rem; font-weight: 700;
    color: #7A869A; font-variant-numeric: tabular-nums;
}
.prob-value.is-top { color: #0B1B3A; }

/* ---------- Footer ---------- */
.app-footer {
    text-align: center; color: #98A2B6;
    font-size: 0.78rem; margin-top: 3rem;
    padding-top: 1.5rem; border-top: 1px solid #E4E9F2;
    letter-spacing: 0.02em;
}
.app-footer strong { color: #5A6784; font-weight: 600; }

@media (max-width: 768px) {
    .hero h1 { font-size: 2rem; }
    .hero p { font-size: 0.95rem; }
    .prediction-value { font-size: 2rem; }
    .app-header { flex-direction: column; align-items: flex-start; }
    .prob-name { flex-basis: 90px; font-size: 0.82rem; }
    .prob-value { flex-basis: 56px; font-size: 0.82rem; }
}
</style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# Top bar
# ==================================================

st.markdown(
    """<div class="app-header"><div class="brand-block"><div class="brand-mark">◉</div><div><p class="brand-text-title">Vision Lab</p><p class="brand-text-sub">Intel Image Classification</p></div></div><div class="header-meta"><span class="meta-pill"><span class="dot"></span> Model Ready</span><span class="meta-pill">ResNet18</span><span class="meta-pill">PyTorch</span></div></div>""",
    unsafe_allow_html=True,
)


# ==================================================
# Device
# ==================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==================================================
# Load Config
# ==================================================

with open("model_config.json", "r") as f:
    config = json.load(f)

num_classes = config["num_classes"]
image_size = config["image_size"]


# ==================================================
# Load Classes
# ==================================================

with open("classes.json", "r") as f:
    class_names = json.load(f)


# ==================================================
# Load Model
# ==================================================

@st.cache_resource
def load_model():
    model = ResNet18(num_classes=num_classes)
    model.load_state_dict(
        torch.load("resnet18_intel_classifier.pth", map_location=device)
    )
    model.to(device)
    model.eval()
    return model


model = load_model()


# ==================================================
# Image Transform
# ==================================================

transform = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]
)


# ==================================================
# Hero (centered)
# ==================================================

st.markdown(
    """<div class="hero"><div class="hero-badge">RESNET18 · PYTORCH</div><h1>Intel Image Classification</h1><p>Upload a scene photograph and the model will identify which of the five Intel image categories it belongs to — with a full confidence breakdown across all classes.</p></div>""",
    unsafe_allow_html=True,
)


# ==================================================
# Supported Categories (centered, under hero)
# ==================================================

chips = "".join(
    f'<span class="class-chip"><span class="chip-dot"></span>{html.escape(str(name))}</span>'
    for name in class_names
)

st.markdown(
    f'<div class="cats-wrap"><div class="cats-row">{chips}</div></div>',
    unsafe_allow_html=True,
)


# ==================================================
# Input Image — Uploader
# ==================================================

st.markdown(
    """<div class="panel-label"><span>Input Image</span><span class="step">STEP 01</span></div>""",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Drop an image here or click to browse",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)


# ==================================================
# Main flow — everything below the uploader in one column
# ==================================================

if uploaded_file is None:

    st.markdown(
        """<div class="surface prediction-empty"><div class="icon">◎</div><div class="title">Awaiting input</div><div class="desc">Upload an image to run inference and see the predicted category along with the full confidence breakdown.</div></div>""",
        unsafe_allow_html=True,
    )

else:

    # ---- Load image ----
    image = Image.open(uploaded_file).convert("RGB")

    # ---- Image preview ----
    st.markdown(
        """<div class="panel-label" style="margin-top: 1.4rem;"><span>Preview</span><span class="step">INPUT</span></div>""",
        unsafe_allow_html=True,
    )

    st.image(image, width="stretch")

    # ---- Inference (unchanged ML logic) ----
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_index = predicted.item()
    predicted_class = class_names[predicted_index]
    confidence_value = confidence.item() * 100

    safe_class = html.escape(str(predicted_class))

    # ---- Predicted Class card ----
    st.markdown(
        """<div class="panel-label" style="margin-top: 1.6rem;"><span>Model Prediction</span><span class="step">STEP 02</span></div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""<div class="prediction-card"><div class="prediction-eyebrow">Predicted Category</div><div class="prediction-value">{safe_class}</div><div class="confidence-row"><span class="confidence-label">Confidence</span><span class="confidence-value">{confidence_value:.2f}%</span></div><div class="confidence-bar-track"><div class="confidence-bar-fill" style="width: {confidence_value:.2f}%;"></div></div></div>""",
        unsafe_allow_html=True,
    )

    # ---- At a glance card (directly below prediction card) ----
    st.markdown(
        f"""<div class="surface" style="margin-top: 1rem;"><div class="glance-title">At a glance</div><div class="glance-row"><span class="glance-k">Top class</span><span class="glance-v">{safe_class}</span></div><div class="glance-row"><span class="glance-k">Confidence</span><span class="glance-v accent">{confidence_value:.2f}%</span></div><div class="glance-row"><span class="glance-k">Classes evaluated</span><span class="glance-v">{num_classes}</span></div></div>""",
        unsafe_allow_html=True,
    )

    # ---- Probability Breakdown card (below At a glance) ----
    st.markdown(
        """<div class="panel-label" style="margin-top: 1.6rem;"><span>Probability Breakdown</span><span class="step">ALL CLASSES</span></div>""",
        unsafe_allow_html=True,
    )

    rows_html = ""
    for i, class_name in enumerate(class_names):
        probability = probabilities[0][i].item()
        percentage = probability * 100
        is_top = (i == predicted_index)

        fill_class = "prob-fill is-top" if is_top else "prob-fill"
        name_class = "prob-name is-top" if is_top else "prob-name"
        value_class = "prob-value is-top" if is_top else "prob-value"

        safe_name = html.escape(str(class_name))

        rows_html += (
            f'<div class="prob-row">'
            f'<div class="{name_class}">{safe_name}</div>'
            f'<div class="prob-track"><div class="{fill_class}" '
            f'style="width: {percentage:.2f}%;"></div></div>'
            f'<div class="{value_class}">{percentage:.2f}%</div>'
            f'</div>'
        )

    st.markdown(
        f'<div class="surface">{rows_html}</div>',
        unsafe_allow_html=True,
    )


# ==================================================
# Footer
# ==================================================

st.markdown(
    """<div class="app-footer">Built with <strong>PyTorch</strong> · <strong>ResNet18</strong> · <strong>Streamlit</strong></div>""",
    unsafe_allow_html=True,
)