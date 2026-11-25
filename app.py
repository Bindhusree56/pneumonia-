"""
NeuroScan AI — Pneumonia Detection from Chest X-ray
UI v4: Refined Light Medical Imaging  |  Streamlit
────────────────────────────────────────────────────
Architecture: single-pass
  1. Resolve active image (upload / sample)
  2. Run validation + inference + Grad-CAM
  3. Render: hero → steps → two columns → history → footer

Model / Grad-CAM / preprocessing / CLIP logic: untouched.
"""

import os
import datetime
import base64
import textwrap
from io import BytesIO
from pathlib import Path
from typing import Optional

import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from src.model import build_model
from src.validator import validate_chest_xray
from src.gradcam import compute_gradcam_blend
from src.report_generator import generate_pdf_report

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan AI — Pneumonia Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# STYLES — pure CSS lives in styles.css, next to this file.
# Palette: paper #F6F4EE, card #FFFFFF, raised #EFEBE2,
#          ink #14213D, muted #6B7488, teal #0E9F8E, coral #EE6B4D.
# Fonts: Space Grotesk (headings + numbers), Inter (body).
# Requires Streamlit >= 1.33 for st.html().
# ──────────────────────────────────────────────────────────────
def load_css(path: str = "styles.css") -> None:
    css = Path(__file__).parent.joinpath(path).read_text(encoding="utf-8")
    st.html(f"<style>{css}</style>")
    # Fallback for Streamlit < 1.33 (strip blank lines before markdown):
    #     clean = "\n".join(line for line in css.splitlines() if line.strip())
    #     st.markdown(f"<style>{clean}</style>", unsafe_allow_html=True)


load_css()

# ──────────────────────────────────────────────────────────────
# CONSTANTS + HELPERS
# ──────────────────────────────────────────────────────────────
SAMPLE_LABELS = {
    "none": "None (upload)",
    "normal": "Normal CXR",
    "pneumonia": "Pneumonia CXR",
}
SAMPLE_OPTIONS = {
    "none": None,
    "normal": "chest_xray/test/NORMAL/NORMAL2-IM-0052-0001.jpeg",
    "pneumonia": "chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg",
}

UPLOAD_THUMB_SVG = (
    '<svg viewBox="0 0 24 24" aria-hidden="true">'
    '<path d="M12 16V4"/><path d="M7 9l5-5 5 5"/>'
    '<path d="M5 16v3h14v-3"/>'
    '</svg>'
)

def pil_to_b64(img: Image.Image, fmt: str = "JPEG") -> str:
    """PIL image → base64 string for inline HTML <img>."""
    buf = BytesIO()
    img.convert("RGB").save(buf, format=fmt, quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def html_block(markup: str) -> str:
    """Dedent inline HTML, drop blank lines and leading indent before st.html()."""
    lines = textwrap.dedent(markup).strip().splitlines()
    return "\n".join(line.strip() for line in lines if line.strip())


def render_svg(markup: str) -> None:
    """Render SVG-bearing markup via markdown; st.html strips inline SVG."""
    st.markdown(markup.replace("\n", ""), unsafe_allow_html=True)


def sample_thumb_b64(path: str) -> str:
    """Small JPEG thumbnail for the sample cards."""
    with Image.open(path) as img:
        thumb = img.convert("RGB")
        thumb.thumbnail((120, 88))
        return pil_to_b64(thumb, fmt="JPEG")


def lung_svg_html() -> str:
    """Line-art lung mark (teal 1.5px stroke) on a soft teal circle."""
    return (
        '<svg class="lung-svg" viewBox="0 0 120 120" aria-hidden="true">'
        '<circle cx="60" cy="60" r="52" fill="#E3F5F2"/>'
        '<g fill="none" stroke="#0E9F8E" stroke-width="1.5" '
        'stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M60 18v24"/>'
        '<path d="M60 42c-5 0-8-3-12-3s-7 4-8 9c-2 9-9 14-13 23-4 8-5 18 1 23 '
        '5 4 12 1 15-5 4-8 5-18 8-26 2-6 6-9 9-9s7 3 9 9c3 8 4 18 8 26 3 6 10 9 15 5 '
        '6-5 5-15 1-23-4-9-11-14-13-23-1-5-4-9-8-9s-7 3-12 3"/>'
        '<path d="M44 56c3 2 6 5 8 9"/><path d="M76 56c-3 2-6 5-8 9"/>'
        '</g>'
        '</svg>'
    )


def hero_html() -> str:
    return html_block(f"""
        <section class="hero-card" aria-label="NeuroScan AI">
          <div class="hero-left">
            <h1 class="hero-title">NeuroScan AI</h1>
            <p class="hero-subtitle">Chest radiograph pneumonia detection · ResNet-18 + Grad-CAM</p>
            <div class="hero-status"><span class="status-dot" aria-hidden="true"></span>Model online</div>
          </div>
          <div class="hero-art">{lung_svg_html()}</div>
          <div class="ecg-line" aria-hidden="true">
            <svg viewBox="0 0 1200 40" preserveAspectRatio="none">
              <polyline points="0,22 120,22 150,10 175,34 205,4 235,22 420,22 450,22 480,12 505,30 535,22 760,22 790,22 820,8 850,34 880,22 1200,22" fill="none" stroke="#0E9F8E" stroke-width="2"/>
            </svg>
          </div>
        </section>
    """)


def steps_html(has_input: bool) -> str:
    steps = ['<div class="step step-done"><span class="step-check">✓</span>Patient</div>']
    if has_input:
        steps.append('<div class="step step-done"><span class="step-check">✓</span>Upload</div>')
        steps.append('<div class="step step-active"><span class="step-num">3</span>Results</div>')
    else:
        steps.append('<div class="step step-active"><span class="step-num">2</span>Upload</div>')
        steps.append('<div class="step step-upcoming"><span class="step-num">3</span>Results</div>')
    return '<div class="step-bar" aria-label="Workflow progress">' + "".join(steps) + '</div>'


def sample_card_html(key: str, label: str, thumb_html: str, selected: bool) -> str:
    cls = "sample-card is-selected" if selected else "sample-card"
    return (
        f'<a class="{cls}" href="?sample={key}" aria-label="Use {label}">'
        f'<span class="sample-thumb">{thumb_html}</span>'
        f'<span class="sample-label">{label}</span>'
        f'</a>'
    )


def odometer_html(value: float, suffix: str = "%") -> str:
    """Pure-CSS count-up: a vertical number strip rolled into place."""
    frames = 20
    values = [f"{value * i / frames:.1f}{suffix}" for i in range(frames)]
    values[-1] = f"{value:.1f}{suffix}"
    strip = "".join(f"<span>{v}</span>" for v in values)
    return (
        f'<span class="count-odometer">'
        f'<span class="count-strip" style="--odo-end:-{frames - 1}em">{strip}</span>'
        f'</span>'
    )


def confidence_ring_html(pct: float, is_pneu: bool) -> str:
    """SVG ring (96×96) with CSS count-up readout and side label block."""
    r    = 40
    circ = 2 * 3.14159 * r
    off  = circ * (1 - pct / 100)
    fill    = "ring-fill-pneumonia" if is_pneu else "ring-fill-normal"
    pct_cls = "ring-pct-pneumonia"  if is_pneu else "ring-pct-normal"
    return (
        f'<div class="ring-wrap fade-up" role="img" aria-label="Model confidence {pct:.1f}%">'
        f'<div class="ring-svg-box">'
        f'<svg width="96" height="96" viewBox="0 0 96 96" aria-hidden="true">'
        f'<circle class="ring-track" cx="48" cy="48" r="{r}"/>'
        f'<circle class="{fill}" cx="48" cy="48" r="{r}" '
        f'stroke-dasharray="{circ:.2f}" stroke-dashoffset="{off:.2f}"/>'
        f'</svg>'
        f'<div class="ring-center">'
        f'<span class="ring-pct {pct_cls}">{odometer_html(pct, "%")}</span>'
        f'<span class="ring-label">confidence</span>'
        f'</div>'
        f'</div>'
        f'<div class="ring-side">'
        f'<div class="ring-side-heading">Model score</div>'
        f'<div class="ring-side-value">{odometer_html(pct, "")}</div>'
        f'</div>'
        f'</div>'
    )


def prob_bars_html(norm_pct: float, pneu_pct: float) -> str:
    """Two rows: each row = label+value line, then bar 8 px below."""
    return (
        f'<div class="prob-block"><div class="prob-row">'
        f'<span class="prob-name">Normal</span>'
        f'<span class="prob-value prob-value-normal">{norm_pct:.1f}%</span>'
        f'</div><div class="prob-track">'
        f'<div class="prob-fill prob-fill-normal" style="width:{norm_pct:.1f}%"></div>'
        f'</div></div>'
        f'<div class="prob-block"><div class="prob-row">'
        f'<span class="prob-name">Pneumonia</span>'
        f'<span class="prob-value prob-value-pneumonia">{pneu_pct:.1f}%</span>'
        f'</div><div class="prob-track">'
        f'<div class="prob-fill prob-fill-pneumonia" style="width:{pneu_pct:.1f}%"></div>'
        f'</div></div>'
    )


def patient_chips_html(pid: str, age: int, gender: str, physician: str, ts: str) -> str:
    """Compact chip row — each fact shown exactly once."""
    chips = [
        ("Patient ID", pid),
        ("Age / Sex",  f"{age} / {gender[0]}"),
        ("Physician",  physician),
        ("Time",       ts),
    ]
    inner = "".join(
        f'<div class="patient-chip">'
        f'<span class="chip-key">{k}</span>'
        f'<span class="chip-val">{v}</span>'
        f'</div>'
        for k, v in chips
    )
    return f'<div class="patient-chips">{inner}</div>'


def history_card_html(entry: dict) -> str:
    """Two-line horizontal history card: filename top, meta row bottom."""
    is_pneu  = entry["pred"] == 1
    chip_cls = "hcard-chip-pneumonia" if is_pneu else "hcard-chip-normal"
    chip_lbl = "Pneumonia"            if is_pneu else "Normal"
    fname    = entry["name"].split(": ", 1)[-1]      # strip "Upload: " / "Sample: " prefix
    return (
        f'<div class="hcard">'
        f'<div class="hcard-name" title="{fname}">{fname}</div>'
        f'<div class="hcard-meta">'
        f'<span class="hcard-pid">{entry["pid"]}</span>'
        f'<span class="hcard-time">· {entry["time"]}</span>'
        f'<span class="hcard-chip {chip_cls}">{chip_lbl}</span>'
        f'</div>'
        f'</div>'
    )


def empty_state_html(title: str, desc: str, progress: int) -> str:
    """Empty/validation card: large lung icon + numbered checklist that fills in."""
    items = [
        ("Upload X-ray", 0),
        ("Review verdict", 1),
        ("Inspect heatmap", 2),
    ]
    checklist = "".join(
        f'<li class="{"done" if i < progress else "pending"}">'
        f'<span class="step-marker">{"✓" if i < progress else i + 1}</span>{label}</li>'
        for label, i in items
    )
    return html_block(f"""
        <div class="empty-state" role="status">
          <div class="empty-lung">{lung_svg_html()}</div>
          <p class="empty-title">{title}</p>
          <p class="empty-desc">{desc}</p>
          <ul class="empty-checklist">{checklist}</ul>
        </div>
    """)


def note_upload_state() -> None:
    st.session_state.upload_active = st.session_state.get("xray_uploader") is not None


# ──────────────────────────────────────────────────────────────
# MODEL LOAD  (unchanged logic)
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_diagnostic_model():
    m = build_model()
    m.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
    m.eval()
    return m


model = load_diagnostic_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ──────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────
if "scan_history" not in st.session_state:
    st.session_state.scan_history = []
if "upload_active" not in st.session_state:
    st.session_state.upload_active = False

sample_key = st.query_params.get("sample", "none")
if sample_key not in SAMPLE_OPTIONS:
    sample_key = "none"

sample_images_exist = all(
    os.path.exists(p) for p in SAMPLE_OPTIONS.values() if p is not None
)
if sample_images_exist:
    try:
        normal_thumb = (
            f'<img src="data:image/jpeg;base64,'
            f'{sample_thumb_b64(SAMPLE_OPTIONS["normal"])}" alt="">'
        )
        pneumonia_thumb = (
            f'<img src="data:image/jpeg;base64,'
            f'{sample_thumb_b64(SAMPLE_OPTIONS["pneumonia"])}" alt="">'
        )
    except Exception:
        sample_images_exist = False

# ══════════════════════════════════════════════════════════════
# PASS 1 — SIDEBAR INPUTS  (collect values before any rendering)
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    with st.container(border=True):
        st.html('<p class="sidebar-section-label">Patient</p>')
        patient_name   = st.text_input("Patient name",        value="Jane Doe")
        patient_id     = st.text_input("Patient ID / MRN",    value="PT-10492")
        # Age and Sex as full-width stacked fields → no truncation
        patient_age    = st.number_input("Age", min_value=1, max_value=120, value=45)
        patient_gender = st.selectbox("Sex", ["Female", "Male", "Other"])
        physician      = st.text_input("Referring physician",  value="Dr. Smith, M.D.")

    with st.container(border=True):
        st.html('<p class="sidebar-section-label">Samples</p>')
        if sample_images_exist:
            cards_html = "".join([
                sample_card_html("none", SAMPLE_LABELS["none"], UPLOAD_THUMB_SVG, sample_key == "none"),
                sample_card_html("normal", SAMPLE_LABELS["normal"], normal_thumb, sample_key == "normal"),
                sample_card_html("pneumonia", SAMPLE_LABELS["pneumonia"], pneumonia_thumb, sample_key == "pneumonia"),
            ])
            render_svg(f'<div class="sample-grid">{cards_html}</div>')
        else:
            labels = list(SAMPLE_LABELS.values())
            choice_label = st.selectbox(
                "Benchmark case",
                labels,
                index=labels.index(SAMPLE_LABELS[sample_key]),
                label_visibility="visible",
            )
            sample_key = next(k for k, v in SAMPLE_LABELS.items() if v == choice_label)

    with st.expander("Validation guard"):
        st.markdown(
            "CLIP zero-shot screening is **active**. "
            "Non-chest-radiograph uploads are automatically rejected "
            "before the model runs.",
        )

# ══════════════════════════════════════════════════════════════
# HERO + STEP INDICATOR
# ══════════════════════════════════════════════════════════════
has_input = sample_key != "none" or st.session_state.upload_active
render_svg(hero_html())
st.html(steps_html(has_input))

# ══════════════════════════════════════════════════════════════
# TWO-COLUMN LAYOUT  (1.4 workspace  :  1 results)
# ══════════════════════════════════════════════════════════════
col_work, col_results = st.columns([1.4, 1], gap="large")

# Shared result variables (populated inside col_work, read in col_results)
_result = None          # dict or None; holds all computed results

with col_work:

    # ── Upload card ───────────────────────────────────────────
    with st.container(border=True):
        st.html('<p class="card-heading">Radiograph input</p>')

        uploaded_file = st.file_uploader(
            "Drag and drop or click to browse — JPEG or PNG, max 10 MB",
            type=["jpg", "jpeg", "png"],
            help="Upload a frontal chest X-ray (PA or AP view).",
            label_visibility="visible",
            key="xray_uploader",
            on_change=note_upload_state,
        )

        # ── Resolve active image ──────────────────────────────
        active_image: Optional[Image.Image] = None
        image_tag = ""
        _status_html = ""

        if uploaded_file is not None:
            if uploaded_file.size > 10 * 1024 * 1024:
                _status_html = (
                    '<div class="status-chip status-chip-warn">'
                    '<span class="chip-dot chip-dot-warn"></span>'
                    'File exceeds 10 MB — please resize and retry'
                    '</div>'
                )
            else:
                try:
                    active_image = Image.open(uploaded_file).convert("RGB")
                    image_tag    = f"Upload: {uploaded_file.name}"
                    w, h         = active_image.size
                    _status_html = (
                        f'<div class="status-chip status-chip-ok">'
                        f'<span class="chip-dot chip-dot-ok"></span>'
                        f'{uploaded_file.name} &nbsp;·&nbsp;'
                        f' {uploaded_file.size / 1024:.0f} KB'
                        f' &nbsp;·&nbsp; {w}×{h} px'
                        f'</div>'
                    )
                except Exception:
                    _status_html = (
                        '<div class="status-chip status-chip-warn">'
                        '<span class="chip-dot chip-dot-warn"></span>'
                        'Could not decode this file — try a different image'
                        '</div>'
                    )

        elif sample_key != "none" and SAMPLE_OPTIONS[sample_key]:
            sp = SAMPLE_OPTIONS[sample_key]
            if os.path.exists(sp):
                active_image = Image.open(sp).convert("RGB")
                image_tag    = f"Sample: {os.path.basename(sp)}"
                w, h         = active_image.size
                _status_html = (
                    f'<div class="status-chip status-chip-ok">'
                    f'<span class="chip-dot chip-dot-ok"></span>'
                    f'{os.path.basename(sp)} &nbsp;·&nbsp; {w}×{h} px'
                    f'</div>'
                )
            else:
                _status_html = (
                    '<div class="status-chip status-chip-warn">'
                    '<span class="chip-dot chip-dot-warn"></span>'
                    'Sample file missing — upload your own image'
                    '</div>'
                )
        else:
            _status_html = (
                '<div class="status-chip status-chip-idle">'
                '<span class="chip-dot chip-dot-idle"></span>'
                'Awaiting radiograph'
                '</div>'
            )

        st.html(_status_html)

    # ── Radiograph viewer ─────────────────────────────────────
    if active_image is not None:
        with st.container(border=True):
            st.html('<p class="card-heading">Viewer</p>')

            # Validation (unchanged logic)
            with st.spinner("Verifying chest radiograph…"):
                is_valid, val_message, val_metrics = validate_chest_xray(active_image)

            if not is_valid:
                st.html(html_block(f"""
                    <div class="error-box">
                      <h4>Not a chest X-ray</h4>
                      <p>{val_message}<br><br>
                      This model accepts frontal chest radiographs only (PA or AP view).
                      Documents, photographs, signatures and drawings are rejected automatically.</p>
                    </div>
                """))
                with st.expander("Validation metrics"):
                    st.json(val_metrics)

            else:
                # ── Inference (unchanged logic) ───────────────
                input_tensor = transform(active_image).unsqueeze(0)
                with torch.no_grad():
                    outputs    = model(input_tensor)
                    probs      = F.softmax(outputs, dim=1)
                    conf_t, pred_t = torch.max(probs, 1)

                pred_idx = pred_t.item()
                conf_pct = conf_t.item() * 100.0
                norm_pct = probs[0][0].item() * 100.0
                pneu_pct = probs[0][1].item() * 100.0
                is_pneu  = (pred_idx == 1)
                diag_lbl = "Pneumonia" if is_pneu else "Normal"
                now_str  = datetime.datetime.now().strftime("%H:%M")

                # Opacity slider (under the tab strip, above images)
                blend_alpha = st.slider(
                    "Heatmap opacity",
                    min_value=0.0, max_value=1.0, value=0.45, step=0.05,
                    help="0 = original only · 1 = full heatmap",
                )

                # Grad-CAM (unchanged logic)
                with st.spinner("Computing Grad-CAM…"):
                    blended_img, pure_heat = compute_gradcam_blend(
                        model, active_image, class_idx=pred_idx, alpha=blend_alpha
                    )

                # Tabs: Original | Heatmap | Blend
                tab_orig, tab_heat, tab_blend = st.tabs(
                    ["Original", "Heatmap", "Blend"]
                )

                img_b64  = pil_to_b64(active_image)
                w_px, h_px = active_image.size

                with tab_orig:
                    st.html(html_block(f"""
                        <div class="xray-wrap" role="img" aria-label="Chest radiograph: {image_tag}">
                          <img src="data:image/jpeg;base64,{img_b64}" alt="Chest radiograph — {image_tag}">
                          <div class="scan-line" aria-hidden="true"></div>
                        </div>
                        <div class="xray-caption">{w_px}×{h_px} px &nbsp;·&nbsp; Frontal CXR</div>
                    """))

                with tab_heat:
                    st.image(
                        pure_heat,
                        caption="Raw Grad-CAM attention (JET colormap) — warmer zones drove the prediction.",
                        use_container_width=True,
                    )

                with tab_blend:
                    st.image(
                        blended_img,
                        caption=f"Blend at {int(blend_alpha * 100)}% heatmap opacity.",
                        use_container_width=True,
                    )

                # ── Persist to history (deduplicate) ─────────
                _last_hist = st.session_state.scan_history
                if not _last_hist or _last_hist[-1]["name"] != image_tag:
                    st.session_state.scan_history.append({
                        "name": image_tag,
                        "pid":  patient_id,
                        "pred": pred_idx,
                        "conf": conf_pct,
                        "time": now_str,
                    })

                # ── Store result for right column ─────────────
                _result = {
                    "pred_idx":    pred_idx,
                    "conf_pct":    conf_pct,
                    "norm_pct":    norm_pct,
                    "pneu_pct":    pneu_pct,
                    "is_pneu":     is_pneu,
                    "diag_lbl":    diag_lbl,
                    "blended_img": blended_img,
                    "now_str":     now_str,
                }

# ─────────────────────────────────────────────────────────────
# RIGHT — results panel
# ─────────────────────────────────────────────────────────────
with col_results:
    with st.container(border=True):

        if _result is None and active_image is None:
            render_svg(empty_state_html(
                "Awaiting radiograph",
                "Upload a chest X-ray or pick a benchmark sample to begin.",
                progress=0,
            ))

        elif _result is None and active_image is not None:
            render_svg(empty_state_html(
                "Validation failed",
                "Upload a valid frontal chest X-ray to see results here.",
                progress=1,
            ))

        else:
            # ── Unpack result ─────────────────────────────────
            pred_idx    = _result["pred_idx"]
            conf_pct    = _result["conf_pct"]
            norm_pct    = _result["norm_pct"]
            pneu_pct    = _result["pneu_pct"]
            is_pneu     = _result["is_pneu"]
            diag_lbl    = _result["diag_lbl"]
            blended_img = _result["blended_img"]
            now_str     = _result["now_str"]

            st.html('<p class="card-heading">Diagnosis</p>')

            # ── Verdict badge (tinted, 4px colored left edge) ─
            if is_pneu:
                st.html(html_block("""
                    <div class="verdict verdict-pneumonia fade-up" role="alert" aria-live="assertive">
                      <p class="verdict-label verdict-label-pneumonia">Pneumonia detected</p>
                      <p class="verdict-desc">
                        Radiographic opacities or consolidations identified consistent with
                        pulmonary pneumonia. Recommend immediate clinical correlation.
                      </p>
                    </div>
                """))
            else:
                st.html(html_block("""
                    <div class="verdict verdict-normal fade-up" role="status" aria-live="polite">
                      <p class="verdict-label verdict-label-normal">Normal — no findings</p>
                      <p class="verdict-desc">
                        No significant pulmonary consolidation or focal infiltrates detected.
                        The lungs appear clear on this radiograph.
                      </p>
                    </div>
                """))

            # ── Confidence ring (shown once) ──────────────────
            render_svg(confidence_ring_html(conf_pct, is_pneu))

            # ── Probability bars ──────────────────────────────
            st.html(prob_bars_html(norm_pct, pneu_pct))

            # ── Patient chip row (shown once, here only) ──────
            st.html(patient_chips_html(patient_id, patient_age, patient_gender, physician, now_str))

            # ── Clinical note (single-line, inline) ───────────
            doctor_notes = st.text_input(
                "Clinical note",
                value="Chest radiograph evaluated. Findings correlate with Grad-CAM.",
                placeholder="Add a clinical note…",
                label_visibility="visible",
            )

            # ── PDF report download ───────────────────────────
            probabilities_dict = {"Normal": norm_pct, "Pneumonia": pneu_pct}
            patient_dict = {
                "name":      patient_name,
                "id":        patient_id,
                "age":       patient_age,
                "gender":    patient_gender,
                "physician": physician,
                "date":      datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
            pdf_bytes = generate_pdf_report(
                patient_info=patient_dict,
                diagnosis=diag_lbl,
                confidence=conf_pct,
                probabilities=probabilities_dict,
                original_image=active_image,
                heatmap_image=blended_img,
                clinical_notes=doctor_notes,
            )
            st.download_button(
                label="Download report (PDF)",
                data=pdf_bytes,
                file_name=f"NeuroScan_{patient_id}_{diag_lbl}.pdf",
                mime="application/pdf",
                help="Download a clinical-grade PDF diagnostic report.",
                use_container_width=True,
            )

# ══════════════════════════════════════════════════════════════
# SESSION HISTORY — horizontal scrolling card row at the bottom
# ══════════════════════════════════════════════════════════════
if st.session_state.scan_history:
    with st.container(border=True):
        st.html('<p class="history-section-title">Session history</p>')
        cards_html = "".join(
            history_card_html(e)
            for e in reversed(st.session_state.scan_history[-12:])
        )
        st.html(f'<div class="history-scroll">{cards_html}</div>')

# ══════════════════════════════════════════════════════════════
# FOOTER — medical disclaimer
# ══════════════════════════════════════════════════════════════
st.html(html_block("""
    <div class="app-footer" role="contentinfo">
      <p>
        NeuroScan AI is for educational and research purposes only and is
        <span class="footer-accent">not a substitute for professional medical diagnosis</span>.
        All outputs are generated by a deep learning model and may contain errors.
        Always consult a qualified healthcare provider before making clinical decisions.
      </p>
      <p>
        NeuroScan AI &nbsp;·&nbsp; ResNet-18 &nbsp;·&nbsp; Grad-CAM &nbsp;·&nbsp;
        CLIP Validation Guard &nbsp;·&nbsp; Streamlit
      </p>
    </div>
"""))
