import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import io

# --- SESSION STATE INITIALIZATION ---
if 'page' not in st.session_state:
    st.session_state['page'] = "Welcome"

# --- CSS FOR WHITE BACKGROUND (Light Theme Default) AND LARGE UPLOADER ---
page_bg_img = """
<style>
/* 1. Ensure the header/top bar is transparent */
[data-testid="stHeader"] {
     background-color: transparent !important;
}

/* 2. FILE UPLOADER RESIZING (Added !important flags to force the size) */
[data-testid="stFileUploadDropzone"] {
     min-height: 400px !important; /* FORCED vertical size */
     padding: 40px !important;    /* FORCED internal space */
     border-width: 3px; 
     border-style: dashed;
}

/* Increase the font size for text inside the drop zone (e.g., "Drag and drop...") */
[data-testid="stFileUploadDropzone"] p {
     font-size: 1.4em !important; 
}

/* Increase the font size for the "Browse files" button */
[data-testid="baseButton-secondary"] {
     font-size: 1.1em;
     padding: 10px 20px;
}
</style>
"""

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(page_title="Synthetic & Manipulated Image Detector", layout="wide") 
st.markdown(page_bg_img, unsafe_allow_html=True) # Inject the CSS

# Hard-coded thresholds
THRESHOLD_RAW = 0.021722
THRESHOLD_ELA = 0.001001 

IMG_HEIGHT, IMG_WIDTH = 128, 128


# --- 2. MODEL LOADING  ---

@st.cache_resource
def load_models():
    """Loads the models and caches them to prevent reloading on every script rerun."""
    try:
        model_raw = tf.keras.models.load_model("model_raw.h5")
        model_ela = tf.keras.models.load_model("model_ela.h5")
        return model_raw, model_ela
    except Exception as e:
        st.error(f"Error loading models. Please ensure 'model_raw.h5' and 'model_ela.h5' are present. Error: {e}")
        return None, None

model_raw, model_ela = load_models()

# --- 3. HELPER FUNCTIONS ---

def calculate_ela(image_stream, quality=90):
    try:
        image_stream.seek(0)
        original = Image.open(image_stream).convert('RGB')
        buffer = io.BytesIO()
        original.save(buffer, 'JPEG', quality=quality)
        buffer.seek(0)
        resaved = Image.open(buffer)
        ela = ImageChops.difference(original, resaved)
        extrema = ela.getextrema()
        max_diff = max([c[1] for c in extrema])
        scale = 255.0 / (max_diff if max_diff != 0 else 1)
        ela_scaled = ImageEnhance.Brightness(ela).enhance(scale)
        return ela_scaled
    except Exception as e:
        st.error(f"Error in ELA calculation: {e}")
        return None

def preprocess_for_model(image_pil):
    img = image_pil.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch

# --- NEW FUNCTION ---
def get_overall_verdict(error_raw, error_ela, threshold_raw, threshold_ela):
    """Final Verdict"""
    
    is_raw_anomaly = error_raw > threshold_raw
    is_ela_anomaly = error_ela > threshold_ela
    
    if is_raw_anomaly and is_ela_anomaly:
        return "HIGH RISK: Synthetic AND Manipulated"
    elif is_raw_anomaly or is_ela_anomaly:
        # One test failed (either AI or manual edit)
        return "MEDIUM RISK: Possible Forgery Detected"
    else:
        # Both tests passed
        return "LOW RISK: Image Appears Authentic"

# --- 4. MAIN PREDICTION FUNCTION  ---
def predict_image(uploaded_file, model_raw, model_ela):
    image_bytes = uploaded_file.read()
    image_stream = io.BytesIO(image_bytes)
    original_pil = Image.open(image_stream).convert('RGB')
    input_raw = preprocess_for_model(original_pil)
    
    if model_raw:
        reconstructed_raw = model_raw.predict(input_raw, verbose=0)
        error_raw = np.mean(np.square(input_raw - reconstructed_raw))
    else:
        error_raw = 0.0
        
    image_stream.seek(0)
    ela_pil = calculate_ela(image_stream)
    
    if ela_pil is None:
        error_ela = 1.0
    else:
        input_ela = preprocess_for_model(ela_pil)
    
    if model_ela:
        reconstructed_ela = model_ela.predict(input_ela, verbose=0)
        error_ela = np.mean(np.square(input_ela - reconstructed_ela))
    else:
        error_ela = 0.0
     
    # C. COMBINED VERDICT
    overall_verdict = get_overall_verdict(
        error_raw, error_ela, THRESHOLD_RAW, THRESHOLD_ELA
    )
    
    # Ensure you return the new variable
    return original_pil, overall_verdict # <--- UPDATED RETURN



# --- 5. STREAMLIT INTERFACE  ---
st.title("Synthetic & Manipulated Image Detector")

# Create the Navigation Sidebar
st.sidebar.title("Navigation")
st.sidebar.radio( # Uses the session state key 'page'
    "Go to",
    ["Welcome", "Image Scanner"], # Renamed for clarity and consistency
    key='page'
)

# Conditional Page Display Logic (Checks use the new, consistent names)
if st.session_state['page'] == "Welcome":
    # --- WELCOME PAGE ---
    st.markdown("""
    Our website performs a two-step scan for *AI Generation* and *Digital Manipulation*.
    
    * *AI Generation Check:* Identifies images created by AI models (like Midjourney or DALL-E).
    * *Manipulation Check (ELA):* Detects inconsistent compression—the telltale sign of Photoshop or editing software.
    """)
    
    st.markdown("---")
    
    # BUTTON
    st.subheader("Ready to Scan an Image?")
    st.button(
        "Image Scanner", 
        type="primary", 
        use_container_width=True, 
        # This lambda function safely modifies the 'page' key on click
        on_click=lambda: st.session_state.update(page="Image Scanner")) 
    
    st.markdown("---")
    
elif st.session_state['page'] == "Image Scanner":
    # --- SCANNER PAGE ---
    st.markdown("Upload an image to check for **AI Generation** and **Photoshop Manipulation**.")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "webp", "tiff", "tif"])

    if uploaded_file is not None:
        uploaded_file.seek(0) 

        if st.button("Scan Image", type="primary", use_container_width=True):
            if model_raw is None or model_ela is None:
                st.warning("Cannot run scan: Models failed to load. Check console for details.")
            else:
                # Use st.spinner for user feedback during processing
                with st.spinner('Analyzing image... This may take a moment.'):
                    
                    # 1. Capture the two returned variables:
                    original_pil, overall_verdict = predict_image(
                        uploaded_file, model_raw, model_ela
                    )
                
                # --- DISPLAY SINGLE RESULT ---
                st.markdown("---")
                
                st.subheader("Original Image")
                st.image(original_pil, caption="Image Submitted for Analysis", use_column_width=True)
                
                st.markdown("---")

                st.subheader("Final Integrity Assessment")
                st.code(overall_verdict, language=None)
                
                # You can add a brief explanation of the verdict here if needed
                st.info("This verdict combines the results of the Raw Image (AI) and ELA (Manipulation) scans.")