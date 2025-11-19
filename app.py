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

/* 2. FILE UPLOADER RESIZING */
[data-testid="stFileUploadDropzone"] > div:first-child { 
    min-height: 380px !important; 
    display: flex !important; 
    flex-direction: column !important;
    justify-content: center !important; 
    align-items: center !important; 
}

[data-testid="stFileUploadDropzone"] {
    min-height: 400px !important;
    height: 400px !important; 
    padding: 40px !important; 
    border-width: 2px;
    border-style: dashed;
    
    background-color: #E4EEF2 !important; 
    border-color: #BBDDE5 !important;    
}

/* Increase the font size for text inside the drop zone (e.g., "Drag and drop...") */
[data-testid="stFileUploadDropzone"] p {
    font-size: 1.4em !important;
    color: #5E8F9B !important;
}

/* SVG */
[data-testid="stFileUploadDropzone"] svg {
    fill: #5E8F9B !important; 
    width: 120px !important; 
    height: 120px !important;
}

/* Increase the font size for the "Browse files" button */
[data-testid="baseButton-secondary"] {
    font-size: 1.1em;
    padding: 10px 20px;
}
</style>
"""

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(
    page_title="Anomaly-Based Detection of Synthetic Human Images", layout="wide")
st.markdown(page_bg_img, unsafe_allow_html=True)  # Inject the CSS

# Hard-coded thresholds
THRESHOLD_RAW = 0.021722
THRESHOLD_ELA = 0.001001
HUMAN_CONFIDENCE_THRESHOLD = 0.8 

IMG_HEIGHT, IMG_WIDTH = 128, 128


# --- 2. MODEL LOADING ---

@st.cache_resource
def load_models():
    """Loads the models and caches them to prevent reloading on every script rerun."""
    try:
        model_raw = tf.keras.models.load_model("model_raw.h5")
        model_ela = tf.keras.models.load_model("model_ela.h5")
        # NOTE: You must provide a human_detector_model.h5 file for the validation to work.
        model_prescreen = tf.keras.models.load_model("human_detector_model.h5")
        
        return model_raw, model_ela, model_prescreen
    
    except Exception as e:
        st.error(f"Error loading models. Please ensure 'model_raw.h5', 'model_ela.h5', and 'human_detector_model.h5' are present. Error: {e}")
        return None, None, None

model_raw, model_ela, model_prescreen = load_models()

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

# --- NEW FUNCTION FOR COMBINED VERDICT ---
def get_overall_verdict(error_raw, error_ela, threshold_raw, threshold_ela):
    """Generates a final, verbose verdict string based on both error scores."""
    
    is_raw_anomaly = error_raw > threshold_raw
    is_ela_anomaly = error_ela > threshold_ela
    
    raw_error_str = f"ERROR: {error_raw:.5f}"
    ela_error_str = f"ERROR: {error_ela:.5f}"
    
    # Calculate the ELA string carefully, handling the None case if ELA failed
    if error_ela == 1.0: # Use 1.0 as the flag for ELA failure
        ela_error_str = "ERROR: ELA calculation failed."
        ela_verdict_prefix = "ERROR"
    elif is_ela_anomaly:
        ela_verdict_prefix = "ANOMALY (MANIPULATED)"
    else:
        ela_verdict_prefix = "REAL (CONSISTENT COMPRESSION)"
        
    # Calculate the RAW string
    if is_raw_anomaly:
        raw_verdict_prefix = "ANOMALY (POSSIBLE AI)"
    else:
        raw_verdict_prefix = "REAL (NATURAL PIXELS)"

    return (
        f"{raw_verdict_prefix} {raw_error_str}\n"
        f"{ela_verdict_prefix} {ela_error_str}"
    )

# --- NEW FUNCTION: HUMAN CHECK (ENHANCED VALIDATION) ---
def is_image_human(image_pil, model_prescreen):
    """
    Uses the prescreen model to check if the image contains a human, 
    and validates that the prediction score isn't too low (e.g., text/blank image).
    """
    try:
        # Preprocess the image for the classification model
        img = image_pil.resize((IMG_WIDTH, IMG_HEIGHT)) 
        img_array = tf.keras.utils.img_to_array(img) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        # Predict confidence score (0 to 1)
        prediction = model_prescreen.predict(img_batch, verbose=0)[0][0]
        
        # 1. Define rejection threshold for non-photographic content
        MIN_PHOTOGRAPHIC_THRESHOLD = 0.05 
        
        # 2. Check if the image meets the MINIMUM confidence required to be a photo
        if prediction < MIN_PHOTOGRAPHIC_THRESHOLD:
            return "NON_PHOTO" 
            
        # 3. Check if the score meets the HIGH confidence required to be a human
        elif prediction >= HUMAN_CONFIDENCE_THRESHOLD:
            return "HUMAN_ACCEPTED"
        
        # 4. If score is in the middle (e.g., 0.1 to 0.79), it's another object/animal
        else:
            return "NON_HUMAN_OBJECT"
        
    except Exception as e:
        print(f"Prescreening check failed: {e}")
        return "CHECK_FAILED"

# --- 4. MAIN PREDICTION FUNCTION ---
def predict_image(uploaded_file, model_raw, model_ela):
    image_bytes = uploaded_file.read()
    image_stream = io.BytesIO(image_bytes)
    original_pil = Image.open(image_stream).convert('RGB')
    input_raw = preprocess_for_model(original_pil)
    
    # A. RAW ANALYSIS (Calculate error_raw first)
    if model_raw:
        reconstructed_raw = model_raw.predict(input_raw, verbose=0)
        error_raw = np.mean(np.square(input_raw - reconstructed_raw))
    else:
        error_raw = 0.0 

    if error_raw > THRESHOLD_RAW:
        verdict_raw = f"ANOMALY (Possible AI)\nError: {error_raw:.5f}"
    else:
        verdict_raw = f"REAL (Natural Pixels)\nError: {error_raw:.5f}"
        
    # B. ELA ANALYSIS (Calculate error_ela and verdict_ela/ela_pil)
    image_stream.seek(0)
    ela_pil = calculate_ela(image_stream)
    
    if ela_pil is None:
        error_ela = 1.0 # Set error_ela high for failure
        verdict_ela = "ERROR: ELA calculation failed."
    else:
        input_ela = preprocess_for_model(ela_pil)
        if model_ela:
            reconstructed_ela = model_ela.predict(input_ela, verbose=0)
            error_ela = np.mean(np.square(input_ela - reconstructed_ela))
        else:
            error_ela = 0.0

        if error_ela > THRESHOLD_ELA:
            verdict_ela = f"ANOMALY (Manipulated)\nError: {error_ela:.5f}"
        else:
            verdict_ela = f"REAL (Consistent Compression)\nError: {error_ela:.5f}"

    # C. GET COMBINED VERDICT STRING (Pass error_raw and error_ela to the new function)
    overall_verdict_string = get_overall_verdict(
        error_raw, error_ela, THRESHOLD_RAW, THRESHOLD_ELA
    )
    
    # D. RETURN ALL NECESSARY ITEMS FOR THE DISPLAY
    return original_pil, verdict_raw, ela_pil, verdict_ela, overall_verdict_string

# --- 5. STREAMLIT INTERFACE ---
st.title("Anomaly-Based Detection of Synthetic Human Image")

# Create the Navigation Sidebar
st.sidebar.title("Navigation")
st.sidebar.radio( 
    "Go to",
    ["Welcome", "Image Scanner"], 
    key='page'
)

# Conditional Page Display Logic
if st.session_state['page'] == "Welcome":
    # --- WELCOME PAGE ---
    st.markdown("""
    Our website performs a two-step scan for **AI Generation** and **Digital Manipulation**.

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
        on_click=lambda: st.session_state.update(page="Image Scanner")) 

    st.markdown("---")

elif st.session_state['page'] == "Image Scanner":
    # --- SCANNER PAGE ---
    st.markdown("Upload an image to check for **AI Generation** and **Photoshop Manipulation**.")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "webp", "tiff", "tif"])

    if uploaded_file is not None:
        
        # Initialize rejection flag and load preview (Level 1)
        is_rejected = False 
        st.subheader("Image Preview:")
        original_pil_check = Image.open(uploaded_file).convert('RGB') 
        st.image(original_pil_check, caption=uploaded_file.name, use_column_width=False, width=300)
        st.markdown("---")
        
        # --- HUMAN CHECK ENFORCEMENT (Level 1) ---
        if model_prescreen: 
            check_result = is_image_human(original_pil_check, model_prescreen)

            if check_result == "NON_PHOTO":
                st.error("🚫 **Upload Rejected:** Please input a human image. This application does not process plain text or non-photographic content.")
                is_rejected = True 
            
            elif check_result == "NON_HUMAN_OBJECT":
                st.error("🚫 **Upload Rejected:** This application is specialized for human images only. Please upload an image containing a human subject.")
                is_rejected = True
        
        # --- 3. RUN SCAN IF CHECK PASSES (Level 1) ---
        # Only show the button and run the scan if the file passed validation
        if not is_rejected:
            uploaded_file.seek(0) # Reset file pointer for predict_image 

            if st.button("Scan Image", type="primary", use_container_width=True):
                
                # Check for detection model loading failures before prediction
                if model_raw is None or model_ela is None:
                    st.warning("Cannot run scan: Detection models failed to load. Check console for details.")
                else:
                    # Use st.spinner for user feedback during processing (Level 2)
                    with st.spinner('Analyzing image... This may take a moment.'):
        
                        # Captures all 5 returned variables (Level 3)
                        original_pil, verdict_raw, ela_pil, verdict_ela, overall_verdict_string = predict_image(
                            uploaded_file, model_raw, model_ela
                        )
                        
                        # --- Display Results in two columns (Level 3) ---
                        st.markdown("---")
                        col1, col2 = st.columns(2)

                        # Column 1: AI Generation Scan (Raw Image)
                        with col1:
                            st.subheader("1. AI Generation Scan (Raw Image)")
                            st.image(original_pil, caption="Original Image", use_column_width=True)

                        # Column 2: Manipulation Scan (ELA)
                        with col2:
                            st.subheader("2. Manipulation Scan (ELA Map)")

                            if ela_pil:
                                st.image(ela_pil, caption="ELA Map (White indicates differences)", use_column_width=True)
                            else:
                                st.warning("ELA Map generation failed.")
                            
                        # --- DISPLAY THE COMBINED VERDICT BELOW THE COLUMNS (Level 3) ---
                        st.markdown("---")
                        st.subheader("Verdict")
                        st.code(overall_verdict_string, language=None)