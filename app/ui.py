"""Streamlit interface for the face authenticity API."""

from __future__ import annotations

import base64
import io
import os

import requests
import streamlit as st
from PIL import Image

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="Deepfake Face Detector", page_icon="🔎", layout="wide")
st.title("Deepfake Face Detector")
st.caption("Upload a JPG or PNG face image for a CPU model prediction and Grad-CAM visualization.")

uploaded = st.file_uploader("Choose a JPG or PNG image", type=("jpg", "jpeg", "png"))
if uploaded is not None:
    original = Image.open(uploaded).convert("RGB")
    if st.button("Analyze image", type="primary"):
        with st.spinner("Analyzing image..."):
            try:
                response = requests.post(
                    f"{API_URL}/analyze",
                    files={
                        "image": (
                            uploaded.name,
                            uploaded.getvalue(),
                            uploaded.type or "application/octet-stream",
                        )
                    },
                    timeout=180,
                )
                response.raise_for_status()
                result = response.json()
            except requests.RequestException as error:
                detail = error.response.text if error.response is not None else str(error)
                st.error(f"Could not analyze the image: {detail}")
            else:
                verdict = result["verdict"]
                color = "#c62828" if verdict == "FAKE" else "#2e7d32"
                confidence = max(result["p_fake"], result["p_real"])
                heatmap = Image.open(
                    io.BytesIO(base64.b64decode(result["heatmap_png_base64"]))
                ).convert("RGB")
                original_column, heatmap_column = st.columns(2)
                with original_column:
                    st.subheader("Original")
                    st.image(original, use_container_width=True)
                with heatmap_column:
                    st.subheader("Grad-CAM (fake-class focus)")
                    st.image(heatmap, use_container_width=True)
                st.markdown(
                    f'<h2 style="color:{color}">Verdict: {verdict}</h2>',
                    unsafe_allow_html=True,
                )
                st.write(f"Confidence: {confidence:.1%}")
                st.progress(confidence)
                st.write(f"P(fake): {result['p_fake']:.4f}  |  P(real): {result['p_real']:.4f}")
                st.code(result["sha256"], language=None)

st.info(
    "**Detector limitation:** Trained on StyleGAN faces only; may fail on other generators."
)
