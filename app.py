import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import io

# CONFIGURATION 
# Hard-coded thresholds
THRESHOLD_RAW = 0.021722
THRESHOLD_ELA = 0.001001  

IMG_HEIGHT, IMG_WIDTH = 128, 128

# LOAD MODELS
print("Loading models...")
model_raw = tf.keras.models.load_model("model_raw.h5")
model_ela = tf.keras.models.load_model("model_ela.h5")
print("Models loaded!")

# HELPER FUNCTIONS
def calculate_ela(image_path, quality=90):
    """Generates an ELA image from a file path."""
    try:
        original = Image.open(image_path).convert('RGB')
        
        # Save as temp JPEG to memory to simulate compression
        buffer = io.BytesIO()
        original.save(buffer, 'JPEG', quality=quality)
        buffer.seek(0)
        resaved = Image.open(buffer)
        
        # Calculate difference
        ela = ImageChops.difference(original, resaved)
        
        # Maximize the difference (scale it up)
        extrema = ela.getextrema()
        max_diff = max([c[1] for c in extrema])
        if max_diff == 0: max_diff = 1
        scale = 255.0 / max_diff
        
        ela_scaled = ImageEnhance.Brightness(ela).enhance(scale)
        return ela_scaled
    except Exception as e:
        print(f"Error in ELA: {e}")
        return None

def preprocess_for_model(image_pil):
    """Resizes and normalizes a PIL image for the model."""
    img = image_pil.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch

# MAIN PREDICTION FUNCTION
def predict_image(image_path):
    if image_path is None:
        return None, "Please upload an image.", None, "Waiting..."

    # A. RAW ANALYSIS (AI Detection) 
    original_pil = Image.open(image_path).convert('RGB')
    input_raw = preprocess_for_model(original_pil)
    
    reconstructed_raw = model_raw.predict(input_raw, verbose=0)
    error_raw = np.mean(np.square(input_raw - reconstructed_raw))
    
    if error_raw > THRESHOLD_RAW:
        verdict_raw = f"ANOMALY (Possible AI)\nError: {error_raw:.5f}"
    else:
        verdict_raw = f"REAL (Natural Pixels)\nError: {error_raw:.5f}"

    # B. ELA ANALYSIS (Photoshop Detection)
    ela_pil = calculate_ela(image_path)
    input_ela = preprocess_for_model(ela_pil)
    
    reconstructed_ela = model_ela.predict(input_ela, verbose=0)
    error_ela = np.mean(np.square(input_ela - reconstructed_ela))
    
    if error_ela > THRESHOLD_ELA:
        verdict_ela = f"ANOMALY (Manipulated)\nError: {error_ela:.5f}"
    else:
        verdict_ela = f"REAL (Consistent Compression)\nError: {error_ela:.5f}"

    return original_pil, verdict_raw, ela_pil, verdict_ela

# BUILD INTERFACE 
with gr.Blocks(title="Deepfake & Manipulation Detector") as demo:
    gr.Markdown("# Synthetic & Manipulated Image Detector")
    gr.Markdown("Upload an image to check for **AI Generation** and **Photoshop Manipulation**.")
    
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="filepath", label="Upload Image")
            submit_btn = gr.Button("Scan Image", variant="primary")
        
        with gr.Column():
            gr.Markdown("### 1. AI Generation Scan")
            out_img_raw = gr.Image(label="Original", height=200)
            out_text_raw = gr.Textbox(label="Verdict")
            
            gr.Markdown("### 2. Manipulation (Photoshop) Scan")
            out_img_ela = gr.Image(label="ELA Map (Hidden Artifacts)", height=200)
            out_text_ela = gr.Textbox(label="Verdict")

    submit_btn.click(
        fn=predict_image, 
        inputs=img_input, 
        outputs=[out_img_raw, out_text_raw, out_img_ela, out_text_ela]
    )

demo.launch()