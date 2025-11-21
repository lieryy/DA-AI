import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
from PIL import Image, ImageChops, ImageEnhance
import io

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Threshold for ELA Model (Reconstruction Error)
THRESHOLD_ELA = 0.004158
# Threshold for Isolation Forest (Anomaly Score). Score < Threshold is ANOMALY.
THRESHOLD_IF_ANOMALY = -0.1 

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
    try:
        clf = joblib.load('isolation_forest.joblib')
        loading_messages.append("✅ Isolation Forest loaded.")
    except Exception:
        clf = None
        loading_messages.append("❌ ERROR: 'isolation_forest.joblib' not found. AI detection will be skipped.")

    # B. Re-build VGG16 (Feature Extractor)
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
    loading_messages.append("✅ VGG16 Feature Extractor ready.")

    # C. Load the ELA Autoencoder
    try:
        model_ela = tf.keras.models.load_model("model_ela.h5")
        loading_messages.append("✅ ELA Model loaded.")
    except Exception:
        model_ela = None
        loading_messages.append("❌ ERROR: 'model_ela.h5' not found. Manipulation detection will be skipped.")
        
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
    
# --- NEW FUNCTION FOR COMBINED VERDICT  ---
def get_overall_verdict(error_raw, error_ela, threshold_raw, threshold_ela):
    """Generates a consolidated verdict string matching the user's requested format."""
    
    is_raw_anomaly = error_raw < threshold_raw 
    is_ela_anomaly = error_ela > threshold_ela
    
    raw_score_str = f"{error_raw:.5f}"
    ela_error_str = f"{error_ela:.5f}"
    
    # Handle ELA failure first
    if error_ela == 1.0:
        return "**❌ SCAN FAILURE: ELA Calculation Failed. Check model files.**"
        
    # --- Case 1: BOTH REAL ---
    if not is_raw_anomaly and not is_ela_anomaly:
        # Desired: REAL (NATURAL PIXELS) ERROR: 0.6456 (CONSISTENT COPDSGD) ERROR:
        return (
            f"✅ REAL (NATURAL PIXELS) ERROR: {raw_score_str}\n"
            f"		(CONSISTENT COMPRESSION) ERROR: {ela_error_str}"
        )
        
    # --- Case 2: F RAW, R ELA (AI Anomaly Only) ---
    if is_raw_anomaly and not is_ela_anomaly:
        # Desired: ANOMALY (POSSIBLE AI) ERROR:
        return f"⚠️ ANOMALY (POSSIBLE AI) ERROR: {raw_score_str}"
        
    # --- Case 3: R RAW, F ELA (Manipulation Anomaly Only) ---
    if not is_raw_anomaly and is_ela_anomaly:
        # Desired: ANOMALY (MANIPULATED) ERROR:
        return f"⚠️ ANOMALY (MANIPULATED) ERROR: {ela_error_str}"
        
    # --- Case 4: F BOTH (AI and Manipulation Anomalies) ---
    if is_raw_anomaly and is_ela_anomaly:
        # Desired: ANOMALY (POSSIBLE AI) ERROR: (MANIPULATED) ERROR:
        return (
            f"⚠️ ANOMALY (POSSIBLE AI) ERROR: {raw_score_str}\n"
            f"	(MANIPULATED) ERROR: {ela_error_str}"
        )

    # Fallback (shouldn't happen)
    return "UNKNOWN VERDICT STATE"


# ==========================================
# 4. MAIN PREDICTION LOGIC
# ==========================================

def predict_image_streamlit(pil_image):
    
    # Initialize error values (IF score: higher is more normal, lower is anomaly)
    anomaly_score = 0.5 # Default score for skipping/normal
    ela_error_val = 1.0 # Default to failure flag (as per get_overall_verdict logic)

    # --- TEST 1: AI GENERATION CHECK (VGG16 + IF) ---
    if clf is None:
        anomaly_score = 0.5 
    else:
        with st.spinner('Running VGG16 Feature Extraction...'):
            # 1. Extract Features
            img_batch_vgg = preprocess_image(pil_image)
            features = feature_extractor.predict(img_batch_vgg, verbose=0)
            
        with st.spinner('Running Isolation Forest Anomaly Detection...'):
            # 2. Calculate Anomaly Score (Raw Error)
            anomaly_score = clf.decision_function(features)[0] 

    # --- TEST 2: MANIPULATION CHECK (ELA + Autoencoder) ---
    ela_pil = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), color='gray') # Default placeholder
    if model_ela is None:
        ela_error_val = 1.0 # Failure flag
    else:
        with st.spinner('Calculating Error Level Analysis (ELA)...'):
            ela_pil_temp = calculate_ela(pil_image)
        
        if ela_pil_temp is None:
            ela_error_val = 1.0 # ELA calculation failed flag
        else:
            ela_pil = ela_pil_temp
            with st.spinner('Running ELA Autoencoder Reconstruction...'):
                # 2. Predict with Autoencoder
                input_ela = preprocess_ela(ela_pil)
                reconstructed_ela = model_ela.predict(input_ela, verbose=0)
                error_ela = np.mean(np.square(input_ela - reconstructed_ela))
                ela_error_val = error_ela

    # --- COMBINED VERDICT ---
    final_verdict_str = get_overall_verdict(
        error_raw=anomaly_score, 
        error_ela=ela_error_val, 
        threshold_raw=THRESHOLD_IF_ANOMALY, 
        threshold_ela=THRESHOLD_ELA
    )
        
    return final_verdict_str, ela_pil


# ==========================================
# 5. BUILD INTERFACE
# ==========================================

st.title("🕵️‍♂️ Anomaly-Based Synthetic Human Detector")
st.markdown("""
This system uses a **Multi-Layered Approach** to detect anomalies:
1.  **Feature Analysis (VGG16 + Isolation Forest):** Checks for AI generation patterns.
2.  **Compression Analysis (ELA + Autoencoder):** Checks for post-processing/manipulation (e.g., Photoshop).
""")

with st.expander("Model Loading Status & Configuration"):
    for msg in loading_feedback:
        st.write(msg)
    st.markdown(f"***ELA Error Threshold:*** `{THRESHOLD_ELA}`")
    st.markdown(f"***AI Score Threshold:*** `{THRESHOLD_IF_ANOMALY}` (Score less than this is ANOMALY)")


# --- INPUT SECTION (Made larger using a container and large header) ---

st.markdown("## ⬆️ Image Upload and Scan") # Larger header
with st.container(border=True):
    uploaded_file = st.file_uploader(
        "Upload Image (JPG, JPEG, PNG, WEBP, TIFF)", 
        type=["jpg", "jpeg", "png", "webp", "tiff", "tif"],
        help="Drag and drop your image file here to begin analysis."
    )

    if uploaded_file is not None:
        # Convert Streamlit UploadedFile to PIL Image
        pil_img = Image.open(uploaded_file)
        
        # Display Preview
        st.image(pil_img, caption="Preview: Input Image Ready for Scan", use_column_width=True)

        # Button is placed next to the image preview
        if st.button("🔍 Start Full Scan", type="primary"):
            
            # Run prediction
            final_verdict_str, ela_pil = predict_image_streamlit(pil_img)

            st.markdown("---")
            st.header("Final Scan Results")
            
            # Display the single, combined verdict
            # Check for 'REAL' or 'ANOMALY' to determine the alert box color
            if "ANOMALY" in final_verdict_str:
                st.error(final_verdict_str)
            elif "REAL" in final_verdict_str:
                st.success(final_verdict_str)
            else:
                st.info(final_verdict_str)


            st.markdown("---")
            
            # --- OUTPUT DISPLAY ---
            col1, col2 = st.columns(2)

            # COLUMN 1: AI GENERATION DETAILS
            with col1:
                st.subheader("🤖 Layer 1: AI Generation Details")
                st.markdown(f"VGG16-extracted features are passed to the Isolation Forest model to detect feature-space anomalies associated with synthetic content. (Threshold: `{THRESHOLD_IF_ANOMALY:.4f}`)")


            # COLUMN 2: MANIPULATION CHECK DETAILS
            with col2:
                st.subheader("🖌️ Layer 2: Manipulation Check Details")
                st.markdown(f"Error Level Analysis (ELA) map shows non-uniform compression levels. The Autoencoder measures the reconstruction error to detect tampering. (Threshold: `{THRESHOLD_ELA:.4f}`)")

                st.subheader("ELA Analysis Map")
                st.image(ela_pil, caption="ELA Image (Higher variance/brightness suggests manipulation)", use_column_width=True)
            
    else:
        st.info("Please upload an image to begin the scan.")