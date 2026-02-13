
import os
import io
from dotenv import load_dotenv
import numpy as np 
import streamlit as st
from PIL import Image



# Load local .env 
load_dotenv()

st.set_page_config(page_title="LandingLens Segmentation App", layout="wide")
st.title("LandingLens Segmentation App")
st.caption("Upload an image → call LandingLens API → view segmentation mask overlay")

# ---  Fetching Credentials from env ---
API_KEY = os.getenv("LANDINGAI_API_KEY", "")
ENDPOINT_ID = os.getenv("LANDINGAI_ENDPOINT_ID", "")

# --- Sidebar controls ---
with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Confidence / Threshold", 0.0, 1.0, 0.50, 0.01)
    display_width = st.slider("Display width (px)", 300, 1200, 700, 10)
    show_debug = st.checkbox("Show debug info", value=False)

    st.markdown("---")
    st.subheader("Credentials status")
    st.success("API Key found ✅" if API_KEY else "API Key missing ❌")
    st.success("Endpoint ID found ✅" if ENDPOINT_ID else "Endpoint ID missing ❌")

# --- Upload ---
uploaded = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])
if not uploaded:
    st.info("Upload an image to run segmentation.")
    st.stop()

img = Image.open(uploaded).convert("RGB")
st.image(img, caption="Input image", width=display_width)

run = st.button("Run Segmentation", type="primary")

# ---------- Helper: Extract mask predictions ----------
def extract_mask_predictions(obj):
    """
    Return a list of prediction objects that have `decoded_boolean_mask`.
    Handles various wrappers: list, tuple, dict, objects with .predictions
    """
    if obj is None:
        return None

    # Case: already a list of mask predictions
    if isinstance(obj, list) and len(obj) > 0 and hasattr(obj[0], "decoded_boolean_mask"):
        return obj

    # Case: tuple wrapper (common in some SDK return shapes)
    if isinstance(obj, tuple):
        for item in obj:
            preds = extract_mask_predictions(item)
            if preds is not None:
                return preds

    # Case: dict wrapper
    if isinstance(obj, dict):
        for v in obj.values():
            preds = extract_mask_predictions(v)
            if preds is not None:
                return preds

    # Case: object that exposes .predictions
    if hasattr(obj, "predictions"):
        return extract_mask_predictions(obj.predictions)

    # Case: single prediction object (rare, but handle)
    if hasattr(obj, "decoded_boolean_mask"):
        return [obj]

    return None

def custom_overlay(image, mask_preds, alpha=0.45):
    """
    Creates overlay with fixed colors for Cat/Dog segmentation masks.
    """
    img_np = np.array(image).astype(np.uint8)

    # Define class colors (RGB)
    COLOR_MAP = {
        "cat": (180, 0, 255),   # purple
        "dog": (255, 255, 0)    # yellow
    }

    overlay_np = img_np.copy()

    for pred in mask_preds:
        label = getattr(pred, "label_name", "unknown").lower()

        # mask is boolean array
        mask = pred.decoded_boolean_mask.astype(bool)

        # pick color (default = red if unknown)
        color = COLOR_MAP.get(label, (255, 0, 0))

        # apply overlay only where mask is True
        overlay_np[mask] = (
            (1 - alpha) * overlay_np[mask] + alpha * np.array(color)
        ).astype(np.uint8)

    return Image.fromarray(overlay_np)



if run:
    if not API_KEY or not ENDPOINT_ID:
        st.error("Missing LANDINGAI_API_KEY or LANDINGAI_ENDPOINT_ID. Put them in your .env and rerun.")
        st.stop()

    with st.spinner("Calling LandingLens API..."):
        try:
            # Import inside run to avoid app crash if package missing
            from landingai.predict import Predictor
            from landingai.visualize import overlay_colored_masks
        except Exception as e:
            st.error("Could not import landingai SDK. Make sure `landingai` is installed in this environment.")
            st.exception(e)
            st.stop()

        try:
            predictor = Predictor(endpoint_id=ENDPOINT_ID, api_key=API_KEY)

            # Try common parameter names; fallback to default if not supported.
            try:
                result = predictor.predict(img, threshold=threshold)
            except TypeError:
                try:
                    result = predictor.predict(img, confidence=threshold)
                except TypeError:
                    result = predictor.predict(img)

            # Some versions return a list even for single image
            result0 = result[0] if isinstance(result, list) and len(result) > 0 else result

            mask_preds = extract_mask_predictions(result0)
            if mask_preds is None:
                st.error("Could not find segmentation mask predictions in the API response.")
                if show_debug:
                    st.write("Type(result):", type(result))
                    st.write("Type(result0):", type(result0))
                    st.write("Raw result0:")
                    st.write(result0)
                st.stop()

            # Build overlay
            overlay = custom_overlay(img, mask_preds, alpha=0.45)

        except Exception as e:
            st.error("Inference failed. Double-check Endpoint ID, deployment status, project type (Segmentation), and credits.")
            if show_debug:
                st.exception(e)
            else:
                st.write(f"{type(e).__name__}: {e}")
            st.stop()

    # --- Results ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Mask Overlay Output")
        st.image(overlay, caption="Segmentation overlay", width=display_width)

        # Download overlay PNG
        buf = io.BytesIO()
        overlay.save(buf, format="PNG")
        st.download_button(
            "Download overlay PNG",
            data=buf.getvalue(),
            file_name="segmentation_overlay.png",
            mime="image/png"
        )

    with col2:
        st.subheader("Reflection helper (use in your PDF)")
        st.write(
            f"**Threshold used:** {threshold:.2f}\n\n"
            "- Higher threshold → cleaner mask but may miss thin/low-confidence regions\n"
            "- Lower threshold → more coverage but can add noise/false positives\n"
        )

        if show_debug:
            st.markdown("### Debug")
            st.write("Type(result):", type(result))
            st.write("Type(result0):", type(result0))
            st.write("Extracted mask preds count:", len(mask_preds))
            st.write("First pred has decoded_boolean_mask:", hasattr(mask_preds[0], "decoded_boolean_mask"))



