# app.py
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, current_app
import os
import uuid
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import markdown2
import datetime

# Import preprocessing functions
try:
    from utils.preprocessing import preprocess_image_inference, denoise_image, IMAGE_SIZE
except ImportError:
    print("Error: Could not import preprocessing functions from utils.preprocessing.")
    def preprocess_image_inference(image_path, **kwargs): return None
    def denoise_image(image_path, **kwargs): return None
    IMAGE_SIZE = 256

# Import model loading and ensemble prediction functions
try:
    from utils.models import initialize_models, predict_ensemble
except ImportError:
    print("Error: Could not import model functions from utils.models.")
    def initialize_models(): print("Model functions not available.")
    def predict_ensemble(image_tensor, loaded_models): return None


# Import Grad-CAM functions
try:
    from utils.gradcam import generate_gradcam_overlay
except ImportError:
     print("Error: Could not import Grad-CAM functions from utils.gradcam.")
     def generate_gradcam_overlay(model, image_tensor, original_image_np, target_class_idx):
          print("Grad-CAM function not available.")
          return None

# Import Gemini utility functions
try:
    from utils.gemini_utils import generate_report_from_results, is_fundus_image_with_gemini
except ImportError:
    print("Error: Could not import gemini_utils functions from utils.gemini_utils.")
    def generate_report_from_results(model_results_text, patient_name=None, patient_age=None, analysis_datetime_str=None):
         return {'comprehensive': 'Gemini API not available.', 'synthetic': 'Gemini API not available.'}
    def is_fundus_image_with_gemini(image_path): return False, "Gemini validation unavailable.", "Validation function not available."


app = Flask(__name__)

# Configuration for uploaded files
UPLOAD_FOLDER = 'static/uploaded_images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- Define Normalization Stats (Replace with your calculated values) ---
NORMALIZATION_MEAN = [0.4247607406554276, 0.41477363903977665, 0.40914320244558566] # Replace with your mean
NORMALIZATION_STD = [0.261673335762141, 0.2545360808997545, 0.2384415815175597] # Replace with your std

inference_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD)
])

# --- Define DR Stage Labels (Keep as is) ---
DR_STAGE_LABELS = {
    0: "No Diabetic Retinopathy",
    1: "Mild Non-Proliferative Diabetic Retinopathy (NPDR)",
    2: "Moderate Non-Proliferative Diabetic Retinopathy (NPDR)",
    3: "Severe Non-Proliferative Diabetic Retinopathy (NPDR)",
    4: "Proliferative Diabetic Retinopathy (PDR)"
}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_images():
    loaded_models = current_app.config.get('LOADED_MODELS')

    if loaded_models is None or not isinstance(loaded_models, dict) or not any(model is not None for model in loaded_models.values()):
         print("FATAL ERROR: Models are not available in app.config. This indicates a failure during startup loading.")
         return render_template('index.html', error_message="FATAL ERROR: Models not available. Please check server logs during startup.")

    # --- Get Patient Info from Form ---
    patient_name = request.form.get('patient_name', 'N/A')
    patient_age = request.form.get('patient_age', 'N/A')
    analysis_datetime_obj = datetime.datetime.now() # Get current date and time object
    analysis_datetime_str = analysis_datetime_obj.strftime("%Y-%m-%d %H:%M:%S") # Format it as string


    # --- Handle single image file upload ---
    image_file = request.files.get('fundus_image')

    if not image_file or image_file.filename == '':
         print("Image file is missing or empty.")
         return render_template('index.html', error_message="Please upload an image file.")

    # Initialize variables to None
    original_url = filtered_url = denoised_url = None
    model_results_text = None
    reports = None
    gradcam_data = []


    original_path = None
    filtered_path = None

    # --- Process the single image ---
    if image_file and allowed_file(image_file.filename):
        original_filename = secure_filename(image_file.filename)
        unique_filename_base = str(uuid.uuid4())
        original_filename_saved = f"{unique_filename_base}_original_{original_filename}"
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename_saved)
        image_file.save(original_path)
        original_url = url_for('static', filename=f'uploaded_images/{original_filename_saved}')
        print(f"Saved original image: {original_filename_saved}")

        # --- STEP 1: Validate image type with Gemini ---
        is_fundus, gemini_validation_response, validation_error = is_fundus_image_with_gemini(original_path)

        if not is_fundus:
             print(f"Image validation failed: {validation_error or gemini_validation_response}")
             if os.path.exists(original_path):
                 os.remove(original_path)
             return render_template('index.html', error_message=f"Uploaded image is not a fundus image or validation failed. Gemini response: {gemini_validation_response}")

        print("Image validated as fundus image. Proceeding with analysis.")

        # Load original image numpy array (for Grad-CAM overlay and other preprocessing)
        original_image_np = cv2.imread(original_path, cv2.IMREAD_COLOR)
        if original_image_np is not None:
             original_image_np = cv2.cvtColor(original_image_np, cv2.COLOR_BGR2RGB)

        # --- STEP 2: Apply filters and display ---
        filtered_image_np = preprocess_image_inference(original_path, image_size=IMAGE_SIZE)
        if filtered_image_np is not None:
            filtered_filename_saved = f"{unique_filename_base}_filtered_{original_filename}"
            filtered_path = os.path.join(app.config['UPLOAD_FOLDER'], filtered_filename_saved)
            filtered_image_bgr = cv2.cvtColor(filtered_image_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filtered_path, filtered_image_bgr)
            filtered_url = url_for('static', filename=f'uploaded_images/{filtered_filename_saved}')
            print(f"Saved filtered image: {filtered_filename_saved}")

            # --- Prepare Filtered Image for Model Inference ---
            filtered_tensor = inference_transform(filtered_image_np)
            filtered_tensor = filtered_tensor.unsqueeze(0)

            # --- STEP 3 & 5: Model Evaluation & Overall Result ---
            ensemble_results = predict_ensemble(filtered_tensor, loaded_models)

            if ensemble_results is not None:
                 # --- Format Model Results Text with Labels ---
                 model_results_text = "Individual Model Results:\n"
                 if ensemble_results['individual_results']:
                     for name, res in ensemble_results['individual_results'].items():
                         if 'error' in res:
                             model_results_text += f"- {name}: Error during inference - {res['error']}\n"
                         else:
                              stage_label = DR_STAGE_LABELS.get(res['predicted_class'], "Unknown Stage")
                              # Format with bold label
                              model_results_text += f"- {name}: Stage {res['predicted_class']} (Confidence: {res['confidence']:.4f}) - **{stage_label}**\n"
                 else:
                     model_results_text += "No individual model results available.\n"

                 model_results_text += f"\nEnsemble Prediction (Majority Vote):\n"
                 ensemble_prediction = ensemble_results.get('ensemble_prediction')
                 ensemble_confidence = ensemble_results.get('ensemble_confidence_fraction')

                 if ensemble_prediction is not None:
                     ensemble_stage_label = DR_STAGE_LABELS.get(ensemble_prediction, "Unknown Stage")
                     model_results_text += f"- Predicted Stage: {ensemble_prediction} (Confidence: {ensemble_confidence:.4f} of successful models) - **{ensemble_stage_label}**\n"
                 elif ensemble_results['individual_results']:
                     model_results_text += "- Could not determine (no clear majority or all failed).\n"
                 else:
                      model_results_text += "- No models were available for prediction.\n"


                 # --- STEP 4: Display GradCAM at columns ---
                 # Generate Grad-CAM Overlay for *each* loaded model for the ensemble's predicted class
                 # ONLY GENERATE GRAD-CAM IF ENSEMBLE PREDICTION IS NOT STAGE 0
                 if ensemble_prediction is not None and ensemble_prediction != 0 and original_image_np is not None:
                      print(f"Generating Grad-CAMs for Stage {ensemble_prediction}...")
                      target_class_idx = ensemble_prediction

                      for model_name, model in loaded_models.items():
                           if model is not None:
                                print(f"  Generating Grad-CAM for {model_name}...")
                                gradcam_overlay_np = generate_gradcam_overlay(
                                     model,
                                     filtered_tensor,
                                     original_image_np,
                                     target_class_idx
                                )

                                if gradcam_overlay_np is not None:
                                     gradcam_filename_saved = f"{unique_filename_base}_gradcam_{model_name}_stage{target_class_idx}_{original_filename}"
                                     gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename_saved)
                                     cv2.imwrite(gradcam_path, gradcam_overlay_np)
                                     # Store URL AND model name as a dictionary
                                     gradcam_data.append({'url': url_for('static', filename=f'uploaded_images/{gradcam_filename_saved}'),
                                                          'model_name': model_name})
                                     print(f"  Saved Grad-CAM for {model_name}: {gradcam_filename_saved}")
                                else:
                                     print(f"  Grad-CAM generation failed for {model_name}.")
                           else:
                                print(f"  Skipping Grad-CAM for {model_name}: Model not loaded.")
                 elif ensemble_prediction == 0:
                     print("Skipping Grad-CAM generation because ensemble prediction is Stage 0 (No DR).")
                 else:
                     print("Skipping Grad-CAM generation: Ensemble prediction is None.")


                 # --- STEP 6: Report in columns ---
                 # Call Gemini API (using actual model results and patient info)
                 reports = generate_report_from_results(
                     model_results_text,
                     patient_name=patient_name,
                     patient_age=patient_age,
                     analysis_datetime_str=analysis_datetime_str # Pass the formatted string
                 )


            else: # If ensemble_results is None (model inference failed)
                 model_results_text = "Model inference failed."
                 reports = {'comprehensive': 'Model inference failed.', 'synthetic': 'Model inference failed.'}


        # Apply Denoising (CLAHE Green Channel) - Use original path as input
        if original_image_np is not None:
            denoised_image_np = denoise_image(original_path, image_size=IMAGE_SIZE)
            if denoised_image_np is not None:
                denoised_filename_saved = f"{unique_filename_base}_clahe_{original_filename}"
                denoised_path = os.path.join(app.config['UPLOAD_FOLDER'], denoised_filename_saved)
                denoised_image_bgr = cv2.cvtColor(denoised_image_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(denoised_path, denoised_image_bgr)
                denoised_url = url_for('static', filename=f'uploaded_images/{denoised_filename_saved}')
                print(f"Saved CLAHE Green: {denoised_filename_saved}")


    # --- STEP 7: End --- Render the template with all results
    reports_html = {k: markdown2.markdown(v) for k, v in reports.items()} if reports else None

    return render_template('index.html',
                           original_url=original_url,
                           filtered_url=filtered_url,
                           denoised_url=denoised_url, # This is CLAHE Green
                           # Pass actual model results text
                           model_results_text=model_results_text,
                           # Pass GradCAM data (URLs and model names)
                           gradcam_data=gradcam_data, # <-- Pass the list of dictionaries
                           # Pass generated reports (rendered as HTML)
                           reports_html=reports_html,
                           # Pass patient info to template if needed for display outside reports
                           patient_name=patient_name, # Pass the variables from the form
                           patient_age=patient_age,
                           analysis_datetime=analysis_datetime_str # Pass the formatted string
                           )


if __name__ == '__main__':
    print("Loading models on startup...")
    loaded_models_dict = initialize_models()

    if not loaded_models_dict or not isinstance(loaded_models_dict, dict) or not any(model is not None for model in loaded_models_dict.values()):
        print("FATAL ERROR: Models failed to load on startup.")
        app.config['LOADED_MODELS'] = {}
    else:
         app.config['LOADED_MODELS'] = loaded_models_dict
         print("Models stored on app.config.")

    app.run(debug=True)