import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
from PIL import Image, ImageChops, ImageEnhance
import io
import os 

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Threshold for ELA Model (Reconstruction Error)
THRESHOLD_ELA = 0.004158

IMG_HEIGHT, IMG_WIDTH = 128, 128

# Set up the Streamlit page configuration
st.set_page_config(
    page_title="Anomaly-Based Synthetic Human Detector",
    layout="centered",
    initial_sidebar_state="auto"
)

# ==========================================
# 2. MODEL LOADING (Cached for performance)
# ==========================================

@st.cache_resource
def load_models():
    """Loads all ML models and assets using Streamlit caching."""
    
    # Initialize containers for feedback
    loading_messages = []
    
    # A. Load the Isolation Forest
    joblib_filename = 'isolation_forest.joblib'
    try:
        # Check the current working directory path for debugging
        current_dir = os.getcwd()
        full_path = os.path.join(current_dir, joblib_filename)
        
        clf = joblib.load(joblib_filename)
        loading_messages.append(f"✅ Isolation Forest loaded from: {full_path}")
    except Exception as e:
        clf = None
        loading_messages.append(f"   ERROR: '{joblib_filename}' not found. AI detection will be skipped. Tried path: {full_path}")
        loading_messages.append(f"   Details: {e}")

    # B. Load VGG16 Feature Extractor
    vgg_filename = 'vgg16_extractor.h5'
    try:
        full_path = os.path.join(os.getcwd(), vgg_filename)
        feature_extractor = tf.keras.models.load_model(vgg_filename)
        loading_messages.append(f"VGG16 Feature Extractor loaded from: {full_path}")
    except Exception as e:
        # Fallback to re-building if the file is missing/corrupt
        loading_messages.append(f"⚠️ WARNING: '{vgg_filename}' not found/loaded. Re-building VGG16 from Keras defaults.")
        try:
            base_model = tf.keras.applications.VGG16(
                weights='imagenet', 
                include_top=False, 
                input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
            )
            inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
            x = tf.keras.applications.vgg16.preprocess_input(inputs)
            x = base_model(x, training=False)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            feature_extractor = tf.keras.Model(inputs, x)
            loading_messages.append("VGG16 Feature Extractor successfully re-built.")
        except Exception as rebuild_e:
            feature_extractor = None
            loading_messages.append(f"CRITICAL ERROR: Could not load or re-build VGG16. Feature extraction failed: {rebuild_e}")


    # C. Load the ELA Autoencoder
    h5_filename = 'model_ela.h5'
    try:
        full_path = os.path.join(os.getcwd(), h5_filename)
        model_ela = tf.keras.models.load_model(h5_filename)
        loading_messages.append(f"ELA Model loaded from: {full_path}")
    except Exception as e:
        model_ela = None
        loading_messages.append(f"ERROR: '{h5_filename}' not found. Manipulation detection will be skipped. Tried path: {full_path}")
        loading_messages.append(f"Details: {e}")
        
    return clf, feature_extractor, model_ela, loading_messages

clf, feature_extractor, model_ela, loading_feedback = load_models()


# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def calculate_ela(pil_image, quality=90):
    """Generates an ELA image from a PIL Image object."""
    try:
        original = pil_image.convert('RGB')
        
        # Save as temp JPEG to memory (the core ELA step)
        buffer = io.BytesIO()
        original.save(buffer, 'JPEG', quality=quality)
        buffer.seek(0)
        resaved = Image.open(buffer)
        
        # Calculate difference
        ela = ImageChops.difference(original, resaved)
        
        # Enhance
        extrema = ela.getextrema()
        max_diff = max([c[1] for c in extrema])
        if max_diff == 0: max_diff = 1
        scale = 255.0 / max_diff
        ela_scaled = ImageEnhance.Brightness(ela).enhance(scale)
        return ela_scaled
    except Exception as e:
        st.error(f"Error in ELA calculation: {e}")
        return None

def preprocess_image(pil_image):
    """Loads and prepares PIL image for VGG16 (Feature Extractor)"""
    # Resize and convert to array
    img = pil_image.resize((IMG_HEIGHT, IMG_WIDTH))
    img = tf.keras.utils.img_to_array(img)
    
    # Add batch dimension. VGG preprocessing is done inside the model wrapper.
    img_batch = np.expand_dims(img, axis=0)
    return img_batch

def preprocess_ela(ela_pil):
    """Prepares ELA image for Autoencoder"""
    img = ela_pil.resize((IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0 # Autoencoder expects 0-1
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch
    
# ==========================================
# 4. MAIN PREDICTION LOGIC
# ==========================================

def predict_image_streamlit(pil_image):
    
    # Defaults
    is_ai_anomaly = False
    is_manipulated = False
    ela_pil = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), color='gray') 
    error_ela = 0.0

    # --- TEST 1: AI GENERATION CHECK (VGG16 + IF) ---
    if clf is None or feature_extractor is None:
        st.warning("Skipping AI Detection: Model failed to load.")
    else:
        with st.spinner('Running VGG16 Feature Extraction...'):
            img_batch_vgg = preprocess_image(pil_image)
            features = feature_extractor.predict(img_batch_vgg, verbose=0)
            
        with st.spinner('Running Isolation Forest Anomaly Detection...'):
            # Predict returns: 1 (Real) or -1 (Anomaly)
            pred = clf.predict(features)[0]
            if pred == -1:
                is_ai_anomaly = True
            else:
                is_ai_anomaly = False

    # --- TEST 2: MANIPULATION CHECK (ELA + Autoencoder) ---
    if model_ela is None:
        st.warning("Skipping Manipulation Check: Model failed to load.")
        error_ela = 1.0 # Error flag
    else:
        with st.spinner('Calculating Error Level Analysis (ELA)...'):
            ela_pil_temp = calculate_ela(pil_image)
        
        if ela_pil_temp is None:
            error_ela = 1.0
        else:
            ela_pil = ela_pil_temp
            with st.spinner('Running ELA Autoencoder Reconstruction...'):
                input_ela = preprocess_ela(ela_pil)
                reconstructed_ela = model_ela.predict(input_ela, verbose=0)
                error_ela = np.mean(np.square(input_ela - reconstructed_ela))
                
                if error_ela > THRESHOLD_ELA:
                    is_manipulated = True
                else:
                    is_manipulated = False
    
    # --- CONSTRUCT COMBINED VERDICT STRING ---
    # Logic matching your request:
    ela_str = f"{error_ela:.5f}"
    
    if error_ela == 1.0:
        final_verdict = "SCAN FAILURE: ELA Calculation Failed."
    
    # Case 1: BOTH REAL
    elif not is_ai_anomaly and not is_manipulated:
        final_verdict = (
            f"REAL (NATURAL PIXELS)\n"
            f"(CONSISTENT COMPRESSION) ELA ERROR: {ela_str}"
        )
        
    # Case 2: AI ANOMALY ONLY
    elif is_ai_anomaly and not is_manipulated:
        final_verdict = (
            f"ANOMALY (POSSIBLE AI)\n"
            f"(CONSISTENT COMPRESSION) ELA ERROR: {ela_str}"
        )

    # Case 3: MANIPULATION ANOMALY ONLY
    elif not is_ai_anomaly and is_manipulated:
        final_verdict = (
            f"ANOMALY (MANIPULATED)\n"
            f"(NATURAL PIXELS) ELA ERROR: {ela_str}"
        )

    # Case 4: BOTH ANOMALIES
    elif is_ai_anomaly and is_manipulated:
        final_verdict = (
            f"ANOMALY (POSSIBLE AI)\n"
            f"(MANIPULATED) ELA ERROR: {ela_str}"
        )
        
    return final_verdict, ela_pil, is_ai_anomaly, is_manipulated


# ==========================================
# 5. BUILD INTERFACE
# ==========================================

st.title("Anomaly-Based Synthetic Human Detector")
st.markdown("""
This system uses a **Multi-Layered Approach** to detect anomalies:
1.  **Feature Analysis (VGG16 + Isolation Forest):** Checks for AI generation patterns.
2.  **Compression Analysis (ELA + Autoencoder):** Checks for post-processing/manipulation (e.g., Photoshop).
""")

# --- INPUT SECTION ---

st.markdown("## **Image Upload and Scan**")
with st.container(border=True):
    uploaded_file = st.file_uploader(
        "Upload Image (JPG, JPEG, PNG, WEBP, TIFF)", 
        type=["jpg", "jpeg", "png", "webp", "tiff", "tif"],
        help="Drag and drop your image file here to begin analysis."
    )

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file)
        st.image(pil_img, caption="Preview: Input Image Ready for Scan", use_container_width=True) 

        if st.button("🔍 Start Full Scan", type="primary"):
            
            # Run prediction
            final_verdict, ela_pil, is_ai, is_manip = predict_image_streamlit(pil_img)

            st.markdown("---")
            st.header("Final Scan Results")
            
            # --- COMBINED RESULT DISPLAY ---
            # Determine color based on whether ANY anomaly exists
            if is_ai or is_manip:
                st.error(final_verdict)
            elif "FAILURE" in final_verdict:
                st.warning(final_verdict)
            else:
                st.success(final_verdict)
            
            st.markdown("---")
            
            # --- VISUAL DISPLAY ---
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Layer 1: AI Analysis")
                st.image(pil_img, caption="Input Image Features (VGG16)", use_container_width=True)
                

            with col2:
                st.subheader("Layer 2: Manipulation Analysis")
                st.image(ela_pil, caption="ELA Analysis Map", use_container_width=True) 
            
    else:
        st.info("Please upload an image to begin the scan.")