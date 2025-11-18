import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import io

# --- 1. CONFIGURATION & SETUP ---

# IMPORTANT: This must be the very first Streamlit command
st.set_page_config(page_title="Deepfake & Manipulation Detector", layout="wide") 

# Hard-coded thresholds
THRESHOLD_RAW = 0.021722
THRESHOLD_ELA = 0.001001 

IMG_HEIGHT, IMG_WIDTH = 128, 128

# --- 2. MODEL LOADING ---

# Use st.cache_resource to load models only once, which is VITAL for speed
@st.cache_resource
def load_models():
    """Loads the models and caches them to prevent reloading on every script rerun."""
    try:
        # NOTE: Model files must be in the same directory as this script for deployment
        model_raw = tf.keras.models.load_model("model_raw.h5")
        model_ela = tf.keras.models.load_model("model_ela.h5")
        return model_raw, model_ela
    except Exception as e:
        # Display an error if models fail to load
        st.error(f"Error loading models. Please ensure 'model_raw.h5' and 'model_ela.h5' are present. Error: {e}")
        return None, None

model_raw, model_ela = load_models()

# --- 3. HELPER FUNCTIONS ---

def calculate_ela(image_stream, quality=90):
    """Generates an ELA image from a file stream/buffer."""
    try:
        # The stream needs to be readable from the start
        image_stream.seek(0)
        original = Image.open(image_stream).convert('RGB')
        
        # Save as temp JPEG to memory to simulate compression
        buffer = io.BytesIO()
        original.save(buffer, 'JPEG', quality=quality)
        buffer.seek(0)
        resaved = Image.open(buffer)
        
        # Calculate difference and maximize
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
    """Resizes and normalizes a PIL image for the model."""
    img = image_pil.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch

# --- 4. MAIN PREDICTION FUNCTION ---

def predict_image(uploaded_file, model_raw, model_ela):
    """Runs both raw and ELA-based predictions."""
    # Read the file content into a buffer for PIL
    image_bytes = uploaded_file.read()
    image_stream = io.BytesIO(image_bytes)

    # A. RAW ANALYSIS (AI Detection) 
    # Use image_stream for Image.open()
    original_pil = Image.open(image_stream).convert('RGB')
    input_raw = preprocess_for_model(original_pil)
    
    # Check if models are available before prediction
    if model_raw:
        reconstructed_raw = model_raw.predict(input_raw, verbose=0)
        error_raw = np.mean(np.square(input_raw - reconstructed_raw))
    else:
        error_raw = 0.0
        
    if error_raw > THRESHOLD_RAW:
        verdict_raw = f"ANOMALY (Possible AI)\nError: {error_raw:.5f}"
    else:
        verdict_raw = f"REAL (Natural Pixels)\nError: {error_raw:.5f}"

    # B. ELA ANALYSIS (Photoshop Detection)
    # Reset stream for ELA calculation
    image_stream.seek(0)
    ela_pil = calculate_ela(image_stream)
    
    if ela_pil is None:
        # Return partial results if ELA fails
        return original_pil, verdict_raw, None, "ERROR: ELA calculation failed."
        
    input_ela = preprocess_for_model(ela_pil)
    
    # Check if models are available before prediction
    if model_ela:
        reconstructed_ela = model_ela.predict(input_ela, verbose=0)
        error_ela = np.mean(np.square(input_ela - reconstructed_ela))
    else:
        error_ela = 0.0
    
    if error_ela > THRESHOLD_ELA:
        verdict_ela = f"ANOMALY (Manipulated)\nError: {error_ela:.5f}"
    else:
        verdict_ela = f"REAL (Consistent Compression)\nError: {error_ela:.5f}"

    return original_pil, verdict_raw, ela_pil, verdict_ela


# --- 5. STREAMLIT INTERFACE ---

st.title("Deepfake & Manipulation Detector")
st.markdown("# Synthetic & Manipulated Image Detector")
st.markdown("Upload an image to check for **AI Generation** and **Photoshop Manipulation**.")

# --- File Uploader and Button setup (in a single column for clean layout) ---
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# Use Streamlit's flow control: button press triggers the main logic
if uploaded_file is not None:
    # Reset the file stream pointer after st.file_uploader initially reads it
    # This is necessary because the file object state is preserved across reruns
    uploaded_file.seek(0) 

    if st.button("Scan Image", type="primary", use_container_width=True):
        if model_raw is None or model_ela is None:
            st.warning("Cannot run scan: Models failed to load. Check console for details.")
        else:
            # Use st.spinner for user feedback during processing
            with st.spinner('Analyzing image... This may take a moment.'):
                
                # Run prediction
                original_pil, verdict_raw, ela_pil, verdict_ela = predict_image(
                    uploaded_file, model_raw, model_ela
                )
                
                # --- Display Results in two columns ---
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                # Column 1: AI Generation Scan
                with col1:
                    st.subheader("1. AI Generation Scan (Raw Image)")
                    st.image(original_pil, caption="Original Image", use_column_width=True)
                    st.markdown("---")
                    st.markdown("Verdict:")
                    st.code(verdict_raw, language=None)

                # Column 2: Manipulation Scan
                with col2:
                    st.subheader("2. Manipulation Scan (ELA Map)")
                    
                    if ela_pil:
                        st.image(ela_pil, caption="ELA Map (White indicates differences)", use_column_width=True)
                    else:
                        st.warning("ELA Map generation failed.")
                        
                    st.markdown("---")
                    st.markdown("Verdict:")
                    st.code(verdict_ela, language=None)